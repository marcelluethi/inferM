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


class Gaussian(mu : alg.Scalar, sdev : alg.Scalar) extends Dist[Double]:
  
  def logPdf(x : Double): alg.Scalar = 
    import alg.* 
    val trig = summon[Trig[alg.Scalar]]
    val PI = alg.lift(Math.PI)
    val logSqrt2Pi = alg.liftToScalar(.5) * trig.log(alg.liftToScalar(2.0) * PI)
    val logSdev = trig.log(sdev)
    val a = (alg.liftToScalar(x) - mu) / sdev
    alg.liftToScalar(-0.5) * a * a - logSqrt2Pi - logSdev

  def draw() : Double = 
    val dist = bdists.Gaussian(mu.value, sdev.value)
    dist.draw()


class MultivariateGaussian(mean : alg.ColumnVector, cov : alg.Matrix)  extends Dist[DenseVector[Double]]:

  def logPdf(x : DenseVector[Double]) = 
    import alg.*
    val trig = summon[Trig[alg.Scalar]]
    val nroot = summon[NRoot[alg.Scalar]]

    val centered = alg.lift(x) - mean
    val precision = cov.inv

    val k = mean.length
        
    val logNormalizer = trig.log(alg.lift(Math.pow(Math.PI * 2, -k  / 2.0)) * alg.lift(1.0) / nroot.sqrt(cov.det))
    
    alg.lift(- 0.5) * centered.t * (precision * centered)  + logNormalizer


  def draw() : DenseVector[Double] = 
    val dist = bdists.MultivariateGaussian(mean.value, cov.value)
    dist.draw()