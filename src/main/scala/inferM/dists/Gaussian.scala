package inferM.dists

import inferM.*
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode
import BreezeDoubleForwardMode.given
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.{algebraT => alg}

import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import breeze.linalg.DenseVector
import scalagrad.api.matrixalgebra.MatrixAlgebra
import spire.algebra.NRoot

class Gaussian(mu: alg.Scalar, sdev: alg.Scalar) extends Dist[Double]:

  def value(s : alg.Scalar) : Double = alg.unliftToDouble(s)

  def logPdf(x: alg.Scalar): alg.Scalar =
    
    val norm = alg.trig.log(
      alg.lift(1.0) / (sdev * alg.lift(Math.sqrt(2.0 * Math.PI)))
    )
    val a = (x - mu) / sdev
    norm + alg.lift(-0.5) * a * a

  def draw(): Double =
    val dist = bdists.Gaussian(alg.unliftToDouble(mu), alg.unliftToDouble(sdev))
    dist.draw()

class MultivariateGaussian[S](mean: alg.ColumnVector, cov: alg.Matrix)(using nroot : NRoot[alg.Scalar]) extends MvDist[DenseVector[Double]]:

  def value(vec : alg.ColumnVector) : DenseVector[Double] = vec.value

  def logPdf(x: alg.ColumnVector): alg.Scalar =

    val centered = x - mean
    val precision = cov.inv

    val k = mean.length

    val logNormalizer = alg.trig.log(
      alg.lift(Math.pow(Math.PI * 2, -k / 2.0)) * (alg.lift(
        1.0
      ) / nroot.sqrt(cov.det))
    )

    alg.lift(-0.5) * centered.t * (precision * centered) + logNormalizer

  def draw(): DenseVector[Double] =
    // TODO don't know how to unlift
    // val dist = bdists.MultivariateGaussian(mean, cov.value)
    // dist.draw()
    ???

