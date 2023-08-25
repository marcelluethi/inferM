package inferM.dists

import inferM.*

import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode
import BreezeDoubleForwardMode.given
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.{algebraT => alg}

import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.given
import scalagrad.api.matrixalgebra.MatrixAlgebra

class Bernoulli(p: alg.Scalar) extends Dist[Boolean]:

  def value(s : alg.Scalar) : Boolean =    
    def isCloseToZero(v: alg.Scalar) =
      Math.abs(alg.unliftToDouble(v)) < 1e-10
    
    if isCloseToZero(s) then false else true


  def logPdf(x: alg.Scalar): alg.Scalar =
    if Math.abs(x.value) < 1e-5 then
      alg.trig.log(alg.liftToScalar(1.0) - p)
    else alg.trig.log(p)

  def draw(): Boolean =
    val dist = bdists.Bernoulli(alg.unliftToDouble(p))
    dist.draw()

