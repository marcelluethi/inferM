package inferM.dists

import inferM.*

import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode
import BreezeDoubleForwardMode.given
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.{algebraT => alg}

import spire.implicits.DoubleAlgebra
import spire.algebra.Trig
import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import breeze.linalg.DenseVector
import spire.syntax.numeric.partialOrderOps

class Uniform(from: alg.Scalar, to: alg.Scalar) extends Dist[Double]:

  def value(s : alg.Scalar) = s.value

  def logPdf(x: alg.Scalar): alg.Scalar =
    if x.value >= from.value && x.value <= to.value then alg.trig.log(alg.liftToScalar(1.0) / (to - from))
    else alg.liftToScalar(Double.MinValue)

  def draw(): Double =
    val dist = bdists.Uniform(alg.unliftToDouble(from), alg.unliftToDouble(to))
    dist.draw()
