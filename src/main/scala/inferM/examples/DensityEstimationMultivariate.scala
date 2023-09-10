package inferM.examples

import inferM.*
import inferM.dists.*
import inferM.sampler.*

import scalagrad.api.spire.trig.DualScalarIsTrig.given
import spire.implicits.DoubleAlgebra
import spire.compat.numeric
import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import scalagrad.auto.forward.BreezeDoubleForwardDualMode.derive as d
import scalagrad.auto.forward.BreezeDoubleForwardDualMode.algebra.*  // import syntax
import scalagrad.auto.forward.BreezeDoubleForwardDualMode.algebraDSL as alg

import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix

object DensityEstimationMultiVariate extends App:

  // example data
  val muGroundTruth = DenseVector(0.0, 0.0)
  val covGroundTruth = DenseMatrix((3.0, 0.0), (0.0, 3.0))

  val data =
    bdists.MultivariateGaussian(muGroundTruth, covGroundTruth).sample(100)

  case class Parameters(mu: alg.ColumnVector, sigma2: alg.Scalar)

  val prior = for
    mu <-
      MultivariateGaussian(
        alg.lift(DenseVector.ones[Double](2)),
        alg.lift(DenseMatrix.eye[Double](2) * 10.0)
      ).toRV("mean")
    sigma2 <- Exponential(alg.lift(0.1)).toRV("sigma2")
  yield Parameters(mu, sigma2)

  val likelihood = (parameters: Parameters) =>
    val targetDist = MultivariateGaussian(
      parameters.mu,
      alg.lift(DenseMatrix.eye[Double](2)) * parameters.sigma2
    )
    

    data.foldLeft(alg.zeroScalar)((sum, point) =>
      sum + targetDist.logPdf(alg.lift(point))
    )

  val posterior = prior.condition(likelihood)

  // Sampling
  val hmc = HMC[Parameters](
    initialValue = Map("mean" -> DenseVector.zeros[Double](2), "sigma2" -> 3.0),
    epsilon = 0.1,
    numLeapfrog = 20
  )

  val samples = posterior.sample(hmc).drop(500).take(1000).toSeq
  val estimatedMean = samples.map(_.value.mu.value).reduce(_ + _) * (1.0 / samples.size)
  val estimatedSigma2 =
    samples.map(_.value.sigma2.value).reduce(_ + _) * (1.0 / samples.size)
  val varSigma2 = samples
    .map(_.value.sigma2.value)
    .map(x => (x - estimatedSigma2) * (x - estimatedSigma2))
    .reduce(_ + _) * (1.0 / samples.size)
  println("mean " + estimatedMean)
  println("sigma2 " + estimatedSigma2)
  println("var sigma2 " + varSigma2)
