package monadbayes.sampler

import monadbayes.*

class MetropolisHastings[A](n : Int, initialSample : A) extends DistInterpreter[A, Seq[A]]: 

  override def pure(value: A): Seq[A] = mh(Pure(value))

  override def primitive(dist: PrimitiveDist[A]): Seq[A] = mh(Primitive(dist))

  override def conditional(lik: A => Prob, dist: Dist[A]): Seq[A] = mh(Conditional(lik, dist))

  override def bind[B](dist: Dist[B], bind: B => Dist[A]): Seq[A] =  mh(Bind(dist, bind))


  def mh(dist : Dist[A]): Seq[A] = 
    val proposals = dist.run(PriorWeightedSampler()).sampleN(n)
    val proposalIt = proposals.iterator
    val initial = proposals.head
    
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
        
    Iterator.iterate((initialSample, Prob(1.0)))(currentSample => oneStep(currentSample, proposalIt.next())).take(proposals.length).toSeq.map(_._1)


  