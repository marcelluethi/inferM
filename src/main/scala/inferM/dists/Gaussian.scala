package inferM.dists

import inferM.* 


import scalagrad.api.matrixalgebra.MatrixAlgebraT
import scalagrad.api.spire.trig.DualScalarIsTrig.given
import spire.implicits.DoubleAlgebra 
import spire.algebra.Trig
import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import scalagrad.auto.forward.breeze.DeriverBreezeDoubleForwardPlan.{algebraT as alg}
import scalagrad.auto.forward.breeze.DeriverBreezeDoubleForwardPlan.given

class Gaussian(mu : alg.Scalar, sdev : alg.Scalar) extends Dist[Double]:
  
  def logPdf(x : Double): alg.Scalar = 
    import alg.* 
    val trig = summon[Trig[alg.Scalar]]
    val PI = alg.liftToScalar(Math.PI)
    val logSqrt2Pi = alg.liftToScalar(.5) * trig.log(alg.liftToScalar(2.0) * PI)
    val logSdev = trig.log(sdev)
    val a = (alg.liftToScalar(x) - mu) / sdev
    alg.liftToScalar(-0.5) * a * a - logSqrt2Pi - logSdev

  def draw() : Double = 
    val dist = bdists.Gaussian(mu.value, sdev.value)
    dist.draw()


