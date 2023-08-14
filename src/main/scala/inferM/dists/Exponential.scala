package inferM.dists


import inferM.* 

import scalagrad.api.matrixalgebra.MatrixAlgebraDSL
import scalagrad.api.spire.trig.DualScalarIsTrig.given
import scalagrad.api.spire.numeric.DualScalarIsNumeric.given
import spire.implicits.DoubleAlgebra 
import spire.algebra.Trig
import spire.algebra.NRoot
import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.{algebraT as alg}
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.given
import breeze.linalg.DenseVector


class Exponential(lambda : alg.Scalar) extends Dist[Double]:
  
  def logPdf(x : Double): alg.Scalar = 
    import alg.* 
    val trig = summon[Trig[alg.Scalar]]
    if x < 0 then alg.liftToScalar(Double.NegativeInfinity) 
    else lambda * alg.lift(-1) * alg.lift(x) + trig.log(lambda)

  def draw() : Double = 
    val dist = bdists.Exponential(lambda.value)
    dist.draw()
