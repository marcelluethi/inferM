package inferM.dists

import inferM.* 


import scalagrad.api.matrixalgebra.MatrixAlgebraT
import scalagrad.api.spire.trig.DualScalarIsTrig.given
import spire.implicits.DoubleAlgebra 
import spire.algebra.Trig
import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import scalagrad.api.ScalaGrad
import scalagrad.auto.forward.breeze.DeriverBreezeDoubleForwardPlan
import DeriverBreezeDoubleForwardPlan.{algebraT as alg}

class Gaussian(mu : alg.Scalar, sdev : alg.Scalar)(using trig: Trig[alg.Scalar]) extends Dist:
  
  def logPdf(x : alg.Scalar): alg.Scalar = 
    import alg.* 
    
    val PI = alg.liftToScalar(Math.PI)
    val logSqrt2Pi = alg.liftToScalar(.5) * trig.log(alg.liftToScalar(2.0) * PI)
    val logSdev = trig.log(sdev)
    val a = (x - mu) / sdev
    alg.liftToScalar(-0.5) * a * a - logSqrt2Pi - logSdev

  def draw() : Double = 
    val dist = bdists.Gaussian(mu.value, sdev.value)
    dist.draw()


