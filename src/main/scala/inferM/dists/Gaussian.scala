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

import breeze.linalg.DenseVector
import scalagrad.api.matrixalgebra.MatrixAlgebra

class Gaussian[S, CV](mu: S, sdev: S)(using
    alg: MatrixAlgebra[S, CV, _, _],
    trig: Trig[S]
) extends Dist[Double, S, CV]:

  def logPdf(x: S): S =
    import alg.*

    val norm = trig.log(
      alg
        .liftToScalar(1.0) / (sdev * alg.liftToScalar(Math.sqrt(2.0 * Math.PI)))
    )
    val a = (x - mu) / sdev
    norm + alg.liftToScalar(-0.5) * a * a

  def draw(): Double =
    val dist = bdists.Gaussian(alg.unliftToDouble(mu), alg.unliftToDouble(sdev))
    dist.draw()

  def toRV(name: String): RV[Double, S, CV] =
    RV[Double, S, CV](
      s => alg.unliftToDouble(s(name).asInstanceOf[S]),
      s => logPdf(s(name).asInstanceOf[S])
    )

class MultivariateGaussian[S, CVec, RVec, M](mean: CVec, cov: M)(using
    alg: MatrixAlgebra[S, CVec, RVec, M],
    trig: Trig[S],
    nroot: NRoot[S]
) extends MvDist[DenseVector[Double], S, CVec]:

  def logPdf(x: CVec): S =

    val centered = x - mean
    val precision = cov.inv

    val k = mean.length

    val logNormalizer = trig.log(
      alg.liftToScalar(Math.pow(Math.PI * 2, -k / 2.0)) * (alg.liftToScalar(
        1.0
      ) / nroot.sqrt(cov.det))
    )

    alg.liftToScalar(-0.5) * centered.t * (precision * centered) + logNormalizer

  def draw(): DenseVector[Double] =
    // TODO don't know how to unlift
    // val dist = bdists.MultivariateGaussian(mean, cov.value)
    // dist.draw()
    ???

  def toRV(name: String): RV[DenseVector[Double], S, CVec] =
    // TODO - unlift seems missing in scala-grad. But casting should work
    RV(
      s => s(name).asInstanceOf[CVec].asInstanceOf[DenseVector[Double]],
      s => logPdf(s(name).asInstanceOf[CVec])
    )
