package inferM.dists

import inferM.*

import scalagrad.api.matrixalgebra.MatrixAlgebraDSL
import scalagrad.api.spire.trig.DualScalarIsTrig.given
import spire.implicits.DoubleAlgebra
import spire.algebra.Trig
import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import breeze.linalg.DenseVector
import scalagrad.api.matrixalgebra.MatrixAlgebra
import spire.syntax.numeric.partialOrderOps

class Uniform[S, CV](from: S, to: S)(using
    alg: MatrixAlgebra[S, CV, _, _],
    trig: Trig[S],
    numeric: Numeric[S]
) extends Dist[S, S, CV]:

  def logPdf(x: S): S =
    import numeric.*
    if x >= from && x <= to then trig.log(alg.liftToScalar(1.0) / (to - from))
    else alg.liftToScalar(Double.MinValue)

  def draw(): S =
    val dist = bdists.Uniform(alg.unlift(from), alg.unlift(to))
    alg.lift(dist.draw())

  def toRV(name: String): RV[S, S, CV] =
    RV(
      s => s(name).asInstanceOf[S],
      s => logPdf(s(name).asInstanceOf[S])
    )
