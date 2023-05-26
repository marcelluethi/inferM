package monadbayes.sampler

import monadbayes.*
import monadbayes.Prob
/**
  * Draw samples from prior and discard likelihood 
  */
class PriorWeightedSampler[A]() extends DistInterpreter[A, Dist[(A, Prob)]]:
  def pure(value : A) : Dist[(A, Prob)] = 
    Pure((value, Prob(1.0)))
  def primitive(dist : PrimitiveDist[A]) : Dist[(A, Prob)] = 
    Primitive(dist.map(a => (a, Prob(1.0))))

  def conditional(lik : A => Prob, dist : Dist[A]) : Dist[(A, Prob)] = 
    val s = dist.run(PriorWeightedSampler[A]())
    (s.map((a, p) => (a, p * lik(a))))

  def bind[B](dist : Dist[B], bind : B => Dist[A] ) : Dist[(A, Prob)] = 
      for 
        x <- dist.run(PriorWeightedSampler[B]())
        y <- bind(x._1)
      yield (y, x._2)

        