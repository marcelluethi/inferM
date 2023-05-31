package inferM.sampler

import inferM.*
import inferM.LogProb

/** Draw samples from prior and discard likelihood
  */
class PriorWeightedSampler[A]() extends DistInterpreter[A, Dist[(A, LogProb)]]:
  def pure(value: A): Dist[(A, LogProb)] =
    Pure((value, LogProb(0.0)))
  def primitive(dist: PrimitiveDist[A]): Dist[(A, LogProb)] =
    Primitive(dist.map(a => (a, LogProb(0.0))))

  def conditional(lik: A => LogProb, dist: Dist[A]): Dist[(A, LogProb)] =
    val s = dist.run(PriorWeightedSampler[A]())
    (s.map((a, p) => (a, p + lik(a))))

  def bind[B](dist: Dist[B], bind: B => Dist[A]): Dist[(A, LogProb)] =
    for
      x <- dist.run(PriorWeightedSampler[B]())
      y <- bind(x._1)
    yield (y, x._2)
