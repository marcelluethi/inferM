package inferM.sampler

import inferM.*
import breeze.linalg.DenseVector
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import breeze.stats.{distributions => bdist}

class SMC[A](numberOfSamples: Int) extends DistInterpreter[A, Dist[Samples[A]]]:


  override def pure(value: A): Dist[Samples[A]] = 
    Pure(Samples(Vector.fill(numberOfSamples)(Sample(value, LogProb(0)))))

  override def primitive(dist: PrimitiveDist[A]): Dist[Samples[A]] =
    val sampleFun = () => Samples(Vector.fill(numberOfSamples)(Sample(dist.sample(), LogProb(0))))
    Primitive(SampleDist(sampleFun))
    

  override def conditional(logLikelihood: A => LogProb, dist: Dist[A]): Dist[Samples[A]] = 
    val samples = dist.run(SMC[A](numberOfSamples)).sample().samples
    val reweightedSamples = for sample <- samples 
    yield 
        Sample(sample.value, logLikelihood(sample.value))
    
    SampleDist(() => Samples(resample(reweightedSamples)))

  override def bind[B](dist: Dist[B], bind: B => Dist[A]): Dist[Samples[A]] = 
   
    // prior sample
    val samples = dist.run(SMC[B](numberOfSamples)).sample().samples
    
    // TODO - we should resample at this point
    val newSamples = for sample <- samples
    yield 
      val newSamples : Samples[A] = bind(sample.value).run(SMC[A](numberOfSamples)).sample()
      Samples(newSamples.samples.map(s => Sample(s.value, s.logWeight + sample.logWeight))).samples
    
    val resampled = resample(newSamples.flatten)
    SampleDist(() => Samples(resampled))


    
  private def resample(samples : Vector[Sample[A]]): Vector[Sample[A]] = 
    val logWeights = samples.map (sample => sample.logWeight.toDouble)
    val mx = logWeights.max
    val rw = logWeights.map(lwi => math.exp(lwi - mx))
    val law = mx + math.log(rw.sum/(rw.length))
    val indices = bdist.Multinomial(DenseVector(rw.toArray)).sample(numberOfSamples).toVector
    indices.map (i => samples(i))
    