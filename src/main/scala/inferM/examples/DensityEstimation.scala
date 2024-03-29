package inferM.examples

import inferM.*
import inferM.dists.*
import inferM.sampler.*

import scalagrad.api.spire.trig.DualScalarIsTrig.given
import spire.implicits.DoubleAlgebra

import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import scalagrad.auto.forward.BreezeDoubleForwardDualMode.derive as d
import scalagrad.auto.forward.BreezeDoubleForwardDualMode.algebra.*  // import syntax
import scalagrad.auto.forward.BreezeDoubleForwardDualMode.algebraDSL as alg

object DensityEstimation extends App:

  // example data
  val muGroundTruth = 5.0
  val sigmaGroundTruth = 2.0

  val data = bdists.Gaussian(muGroundTruth, sigmaGroundTruth).sample(100)

  case class Parameters(mu: alg.Scalar, sigma: alg.Scalar)

  val prior = for
    mu <- Gaussian(alg.lift(0.0), alg.lift(10)).toRV("mu")
    sigma <- Exponential(alg.lift(1.0)).toRV("sigma")
  yield Parameters(mu, sigma)

  val likelihood = (parameters: Parameters) =>
    val targetDist = Gaussian(parameters.mu, parameters.sigma)
    data.foldLeft(alg.zeroScalar)((sum, point) =>
      sum + targetDist.logPdf(alg.lift(point))
    )

  val posterior = prior.condition(likelihood)

  // Sampling
  val hmc = HMC[Parameters](
    initialValue = Map("mu" -> 0.0, "sigma" -> 1.0),
    epsilon = 0.01,
    numLeapfrog = 20
  )

  val samples = posterior.sample(hmc).drop(1000).take(1000).toSeq

  println("mean mu: " + samples.map(s => s.value.mu.toDouble).sum / samples.size)
  println("mean sigma: " + samples.map(s => s.value.sigma.value).sum / samples.size)
