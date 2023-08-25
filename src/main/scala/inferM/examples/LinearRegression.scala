package inferM.examples
/*
import inferM.*
import inferM.dists.*
import inferM.sampler.*

import scalagrad.api.spire.trig.DualScalarIsTrig.given
import spire.implicits.DoubleAlgebra
import spire.compat.numeric
import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import scalagrad.api.spire.numeric.DualScalarIsNumeric.given
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.given
import BreezeDoubleForwardMode.{algebraT as alg}

object LinearRegression extends App:

  // example data
  val aGroundTruth = 3.0
  val bGroundTruth = 2.0
  val noiseSigma = 1.0

  val data = Seq
    .range(0, 10)
    .map(x =>
      (
        x.toDouble,
        aGroundTruth * x + bGroundTruth + bdists
          .Gaussian(0.0, noiseSigma)
          .sample()
      )
    )

  case class Parameters(a: Double, b: Double, sigma: Double)

  val prior = for
    a <- Gaussian(alg.lift(1.0), alg.lift(10)).toRV("a")
    b <- Gaussian(alg.lift(2.0), alg.lift(10)).toRV("b")
    sigma <- Exponential(alg.lift(1.0)).toRV("sigma")
  yield Parameters(a, b, sigma)

  val likelihood = (parameters: Parameters) =>
    data.foldLeft(alg.zeroScalar)((sum, point) =>
      val (x, y) = point
      sum + Gaussian(
        alg.lift(parameters.a) * alg.lift(x) + alg.lift(parameters.b),
        alg.lift(parameters.sigma)
      ).logPdf(alg.lift(y))
    )

  val posterior = prior.condition(likelihood)

  // Sampling
  val hmc = HMC[Parameters](
    initialValue = Map("a" -> 1.0, "b" -> 0.0, "sigma" -> 1.0),
    epsilon = 0.1,
    numLeapfrog = 20
  )

  val samples = posterior.sample(hmc).drop(10000).take(10000).toSeq

  println("mean a: " + samples.map(_._1).sum / samples.size)
  println("mean b: " + samples.map(_._2).sum / samples.size)
  println("mean sigma: " + samples.map(_._3).sum / samples.size)
*/