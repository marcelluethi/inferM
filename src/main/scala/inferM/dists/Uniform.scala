package inferM.dists

import inferM.*

import scalagrad.auto.forward.BreezeDoubleForwardDualMode.derive as d
import scalagrad.auto.forward.BreezeDoubleForwardDualMode.algebra.*  // import syntax
import scalagrad.auto.forward.BreezeDoubleForwardDualMode.algebraDSL as alg

import spire.implicits.DoubleAlgebra
import spire.algebra.Trig
import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import breeze.linalg.DenseVector
import spire.syntax.numeric.partialOrderOps

class Uniform(from: alg.Scalar, to: alg.Scalar) extends Dist:

  def value(v : alg.Scalar) = v

  def logPdf(x: alg.Scalar): alg.Scalar =
    if x.value >= from.value && x.value <= to.value then alg.trig.log(alg.lift(1.0) / (to - from))
    else alg.lift(Double.MinValue)

  def draw(): Double =
    val dist = bdists.Uniform(from.value, to.value)
    dist.draw()
