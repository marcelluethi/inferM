package monadbayes.sampler

import monadbayes.*

/**
  * Draw samples from prior and discard likelihood 
  */
class PriorSampler[A]() extends DistInterpreter[A, Dist[A]]:
  def pure(value : A) : Dist[A] = 
    Pure(value)
  def primitive(dist : PrimitiveDist[A]) : Dist[A] = 
    Primitive(dist)
  def conditional(lik : A => Prob, dist : Dist[A]) : Dist[A] = 
    dist.run(PriorSampler[A]())
  def bind[B](dist : Dist[B], bind : B => Dist[A] ) : Dist[A] = 
    Bind[B, A](dist.run(PriorSampler[B]()), bind)
        