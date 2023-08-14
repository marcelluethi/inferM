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
import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix

object DensityEstimationMultiVariate extends App:  

  // example data
  val muGroundTruth = DenseVector(1.0, 2.0)
  val covGroundTruth = DenseMatrix((1.0, 0.0), (0.0, 1.0))

  val data = bdists.MultivariateGaussian(muGroundTruth, covGroundTruth).sample(100)
  
  case class Parameters(mu : DenseVector[Double])

  val prior = for  
    mu <- RV.fromPrimitiveMultivariate(
      MultivariateGaussian(
        alg.lift(DenseVector.zeros[Double](2)), 
        alg.lift(DenseMatrix.eye[Double](2))
      ), "mean")
  yield Parameters(mu)

  
  val likelihood =  (parameters : Parameters) => 
    val targetDist = MultivariateGaussian(
        alg.lift(parameters.mu),
        alg.lift(covGroundTruth)
    )
      
    data.foldLeft(alg.zeroScalar)((sum, point) =>
      sum + targetDist.logPdf(point)
    )
    
  val posterior = prior.condition(likelihood)

  // Sampling
  val hmc = HMC[Parameters](
    initialValue = Map("mean" -> DenseVector.zeros[Double](2)),
    epsilon = 0.05,
    numLeapfrog = 20
  )

  val samples = posterior.sample(hmc).take(1000).toSeq
  val estimatedMean = samples.map(_.mu).reduce(_ + _) * (1.0 /  samples.size)
  println("mean " + estimatedMean)


