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

class Exponential[S, CV](lambda: S)(using
    alg: MatrixAlgebra[S, CV, _, _],
    trig: Trig[S],
    numeric: Numeric[S]
) extends Dist[Double, S, CV]:

  def logPdf(x: S): S =
    import numeric.*
    if x < alg.liftToScalar(0) then alg.liftToScalar(Double.NegativeInfinity)
    else lambda * alg.liftToScalar(-1) * x + trig.log(lambda)

  def draw(): Double =
    val dist = bdists.Exponential(alg.unliftToDouble(lambda))
    dist.draw()

  def toRV(name: String): RV[Double, S, CV] =
    RV(
      s => alg.unliftToDouble(s(name).asInstanceOf[S]),
      s => logPdf(s(name).asInstanceOf[S])
    )
