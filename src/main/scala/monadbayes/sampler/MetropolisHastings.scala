package monadbayes.sampler

import monadbayes.*

class MetropolisHastings[A](initialSample : A) extends DistInterpreter[A, Iterator[A]]: 

  override def pure(value: A): Iterator[A] = mh(Pure(value))

  override def primitive(dist: PrimitiveDist[A]): Iterator[A] = mh(Primitive(dist))

  override def conditional(lik: A => Prob, dist: Dist[A]): Iterator[A] = mh(Conditional(lik, dist))

  override def bind[B](dist: Dist[B], bind: B => Dist[A]): Iterator[A] =  mh(Bind(dist, bind))


  def mh(dist : Dist[A]): Iterator[A] = 
    val proposals = dist.run(PriorWeightedSampler())
    val proposalIt = Iterator.continually(proposals.sample())
    
    val rng = scala.util.Random()
    
    def oneStep(currentSample : (A, Prob), newProposal : (A, Prob)) : (A, Prob) = 

        val (currentValue, currentP) = currentSample
        val (newValue, newP) = newProposal
        val alpha = scala.math.min(1.0, newP.toDouble / currentP.toDouble)
        val r = rng.nextDouble()
        if (r < alpha) then 
            newProposal 
        else 
            currentSample
        
    Iterator.iterate((initialSample, Prob(1.0)))(currentSample => oneStep(currentSample, proposalIt.next())).map(_._1)


  