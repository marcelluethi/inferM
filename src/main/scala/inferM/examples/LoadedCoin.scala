package inferM.examples

import inferM.*
import inferM.dists.*
import inferM.sampler.*

import scalagrad.api.spire.trig.DualScalarIsTrig.given
import spire.implicits.DoubleAlgebra
import scalagrad.api.spire.numeric.DualScalarIsNumeric.given
import spire.compat.numeric
import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.VariableSeed.randBasis

import scalagrad.auto.forward.BreezeDoubleForwardDualMode.derive as d
import scalagrad.auto.forward.BreezeDoubleForwardDualMode.algebra.*  // import syntax
import scalagrad.auto.forward.BreezeDoubleForwardDualMode.algebraDSL as alg

object LoadedCoin extends App:

  // example data
  val pGroundTruth = 0.1
  val data = bdists.Bernoulli(pGroundTruth).sample(100)

  val prior =
    for p <- Gaussian(alg.lift(0.5), alg.lift(1.0)).toRV("p")
    yield p

  val likelihood = (p: alg.Scalar) =>
    val targetDist = Bernoulli(p)

    data.foldLeft(alg.zeroScalar)((sum, x) =>
      sum + targetDist.logPdf(
        if x then alg.lift(1.0) else alg.lift(0.0)
      )
    )

  val posterior = prior.condition(likelihood)

  // Sampling
  val hmc = HMC[alg.Scalar](
    initialValue = Map("p" -> 0.5),
    epsilon = 0.01,
    numLeapfrog = 20
  )

  val samples = posterior.sample(hmc).take(1000).toSeq

  println("mean p: " + samples.map(_.value.toDouble).sum / samples.size)
