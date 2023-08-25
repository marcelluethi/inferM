package inferM.dists

import inferM.*

import scalagrad.api.spire.trig.DualScalarIsTrig.given
import spire.implicits.DoubleAlgebra
import spire.algebra.Trig
import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import scalagrad.api.matrixalgebra.MatrixAlgebra
import spire.implicits.trigOps

class Bernoulli[S, CV](p: S)(using
    alg: MatrixAlgebra[S, CV, _, _],
) extends Dist[Boolean, S, CV]:

  def logPdf(x: S): S =
    if (Math.abs(alg.unliftToDouble(x)) < 1e-5) then
      alg.trig.log(alg.liftToScalar(1.0) - p)
    else alg.trig.log(p)

  def draw(): Boolean =
    val dist = bdists.Bernoulli(alg.unliftToDouble(p))
    dist.draw()

  def toRV(name: String): RV[Boolean, S, CV] =
    def isCloseToZero(v: S) =
      Math.abs(alg.unliftToDouble(v)) < 1e-10
    RV(
      s => if isCloseToZero(s(name).asInstanceOf[S]) then false else true,
      s => logPdf(s(name).asInstanceOf[S])
    )
