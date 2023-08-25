package inferM.dists

import inferM.*

import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode
import BreezeDoubleForwardMode.given
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.{algebraT => alg}

import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.given
import scalagrad.api.matrixalgebra.MatrixAlgebra

class Bernoulli(p: alg.Scalar) extends Dist:

  def value(s : alg.Scalar) : alg.Scalar =  if (s.value <= p.value) alg.lift(1.0) else alg.lift(0.0)

  def logPdf(x: alg.Scalar): alg.Scalar =
    if Math.abs(x.value) < 1e-5 then
      alg.trig.log(alg.liftToScalar(1.0) - p)
    else alg.trig.log(p)

  def draw(): Boolean =
    val dist = bdists.Bernoulli(alg.unliftToDouble(p))
    dist.draw()

