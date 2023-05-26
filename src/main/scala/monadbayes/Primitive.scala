package monadbayes

import breeze.stats.distributions.Rand.FixedSeed.randBasis
import breeze.stats.{distributions => bdist}

/**
  * A distribution that can be sampled from
  */
trait PrimitiveDist[A] : 
  self => 
    def sample(): A

    def map[B](f: A => B): PrimitiveDist[B] = SampleDist(() => f(self.sample()))
    def flatMap[B](f: A => PrimitiveDist[B]): PrimitiveDist[B] = 
        SampleDist(() => f(self.sample()).sample())
    def toRV: Dist[A] = Primitive(self)

class SampleDist[A](_sample : () => A) extends PrimitiveDist[A]:
    override def sample(): A = _sample()


case class Normal(mean: Double, stdDev: Double) extends PrimitiveDist[Double]:
    val normal = bdist.Gaussian(mean, stdDev)
    def sample(): Double = normal.draw()
    def logPdf(a: Double): Prob = Prob(normal.logPdf(a))
    def pdf(a: Double): Prob = Prob(normal.pdf(a))
