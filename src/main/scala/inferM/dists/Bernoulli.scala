package inferM.dists

import inferM.*

import scalagrad.auto.forward.BreezeDoubleForwardDualMode.derive as d
import scalagrad.auto.forward.BreezeDoubleForwardDualMode.algebra.*  // import syntax
import scalagrad.auto.forward.BreezeDoubleForwardDualMode.algebraDSL as alg

import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis


class Bernoulli(p: alg.Scalar) extends Dist:

  def value(s : alg.Scalar) : alg.Scalar =  if (s.value <= p.value) alg.lift(1.0) else alg.lift(0.0)

  def logPdf(x: alg.Scalar): alg.Scalar =
    if Math.abs(x.value) < 1e-5 then
      alg.trig.log(alg.lift(1.0) - p)
    else alg.trig.log(p)

  def draw(): Boolean =
    val dist = bdists.Bernoulli(p.value)
    dist.draw()

