package inferM.sampler

import inferM.*

class PIMH[A](numberOfSamples : Int) extends DistInterpreter[A, Iterator[Samples[A]]]:

  override def pure(value: A): Iterator[Samples[A]] = pimh(Pure(value), numberOfSamples)

  override def primitive(dist: PrimitiveDist[A]): Iterator[Samples[A]] = pimh(Primitive(dist), numberOfSamples)

  override def conditional(logLikelihood: A => LogProb, dist: Dist[A]) : Iterator[Samples[A]] = pimh(Conditional(logLikelihood, dist), numberOfSamples)

  override def bind[B](dist: Dist[B], bind: B => Dist[A]): Iterator[Samples[A]] = pimh(Bind(dist, bind), numberOfSamples)


  def pimh(dist : Dist[A], numberOfSamples : Int) : Iterator[Samples[A]] = 
    val smc = dist.run(SMC(numberOfSamples))
    MetropolisHastings.mh(smc)
    
   
    //     public static Dist<IEnumerable<Samples<A>>> Create<A>(int numParticles, int chainLen, Dist<A> dist)
    //     {
    //         return MetropolisHastings.MHPrior(dist.Run(new Smc<A>(numParticles)), chainLen);
    //     }
    // }

