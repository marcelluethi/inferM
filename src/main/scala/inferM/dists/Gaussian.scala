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
import org.apache.commons.math3.fraction.Fraction

class Gaussian[S: Fractional, CV](mu: S, sdev: S)(using
    alg: MatrixAlgebra[S, CV, _, _],
) extends Dist[Double, S, CV]:

  def logPdf(x: S): S =
    val norm = alg.trig.log(
      alg.lift(1.0) / (sdev * alg.lift(Math.sqrt(2.0 * Math.PI)))
    )
    val a = (x - mu) / sdev
    norm + alg.lift(-0.5) * a * a

  def draw(): Double =
    bdists.Gaussian(mu.toDouble, sdev.toDouble).draw()

  def toNewRV(name: String)(using nameOf: ValueOf[name.type]): NewRV[S, S, CV, Tuple1[(name.type, S)]] =
    NewRV(
      s => s._1._2,
      s => logPdf(s._1._2)
    )

class MultivariateGaussian[S, CVec, RVec, M](mean: CVec, cov: M)(using
    alg: MatrixAlgebra[S, CVec, RVec, M],
    trig: Trig[S],
    nroot: NRoot[S]
) extends MvDist[DenseVector[Double], S, CVec]:

  import alg.given

  def logPdf(x: CVec): S =

    val centered = x - mean
    val precision = cov.inv

    val k = mean.length

    val logNormalizer = trig.log(
      alg.lift(Math.pow(Math.PI * 2, -k / 2.0)) * (alg.lift(
        1.0
      ) / nroot.sqrt(cov.det))
    )

    alg.lift(-0.5) * centered.t * (precision * centered) + logNormalizer

  def draw(): DenseVector[Double] =
    bdists.MultivariateGaussian(alg.unlift(mean), alg.unlift(cov)).draw()

  def toRV(name: String): RV[CVec, S, CVec] =
    // TODO - unlift seems missing in scala-grad. But casting should work
    RV(
      s => s(name).asInstanceOf[CVec],
      s => logPdf(s(name).asInstanceOf[CVec])
    )
