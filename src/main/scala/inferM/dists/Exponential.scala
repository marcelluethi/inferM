package inferM.dists

import inferM.*

import scalagrad.auto.forward.BreezeDoubleForwardDualMode.derive as d
import scalagrad.auto.forward.BreezeDoubleForwardDualMode.algebra.*  // import syntax
import scalagrad.auto.forward.BreezeDoubleForwardDualMode.algebraDSL as alg

import spire.implicits.DoubleAlgebra
import spire.algebra.Trig
import spire.algebra.NRoot
import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import breeze.linalg.DenseVector
import scalagrad.api.matrixalgebra.MatrixAlgebra

class Exponential(lambda: alg.Scalar) extends Dist:

  def value(s : alg.Scalar) : alg.Scalar = s

  def logPdf(x: alg.Scalar): alg.Scalar =  
    if x.value < 0 then alg.lift(Double.NegativeInfinity)
    else lambda * alg.lift(-1) * x + alg.trig.log(lambda)

  def draw(): Double =
    val dist = bdists.Exponential(lambda.value)
    dist.draw()

