package inferM.examples

import inferM.*
import inferM.dists.*
import inferM.sampler.*

import scalagrad.api.spire.trig.DualScalarIsTrig.given
import spire.implicits.DoubleAlgebra

import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.given
import BreezeDoubleForwardMode.{algebraT as alg}
import scalagrad.api.matrixalgebra.MatrixAlgebra

object DensityEstimation extends App:

  // example data
  val muGroundTruth = 5.0
  val sigmaGroundTruth = 2.0

  val data = bdists.Gaussian(muGroundTruth, sigmaGroundTruth).sample(100)

  case class Parameters(mu: alg.Scalar, sigma: alg.Scalar)

  val prior = for
    mu <- Gaussian(alg.liftToScalar(0.0), alg.liftToScalar(10)).toRV("mu")
    sigma <- Exponential(alg.liftToScalar(1.0)).toRV("sigma")
  yield Parameters(mu, sigma)

  val likelihood = (parameters: Parameters) =>
    val targetDist = Gaussian(parameters.mu, parameters.sigma)
    data.foldLeft(alg.zeroScalar)((sum, point) =>
      sum + targetDist.logPdf(alg.liftToScalar(point))
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
