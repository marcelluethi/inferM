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

object DensityEstimation extends App:  

  // example data
  val muGroundTruth = 3.0
  val sigmaGroundTruth = 2.0

  val data = bdists.Gaussian(muGroundTruth, sigmaGroundTruth).sample(100)
  
  case class Parameters(mu : Double, sigma : Double)

  val prior = for  
    mu <- RV.fromPrimitive(Gaussian(alg.liftToScalar(0.0), alg.liftToScalar(10)), "mu")
    sigma <- RV.fromPrimitive(Gaussian(alg.liftToScalar(1.0), alg.liftToScalar(1.0)), "sigma")
  yield Parameters(mu, sigma)

  
  val likelihood =  (parameters : Parameters) => 
    val targetDist = Gaussian(
        alg.liftToScalar(parameters.mu),
        alg.liftToScalar(parameters.sigma)
      )
    data.foldLeft(alg.zeroScalar)((sum, point) =>
      sum + targetDist.logPdf(point)
    )
    
  val posterior = prior.condition(likelihood)

  // Sampling
  val hmc = HMC[Parameters](
    initialValue = Map("mu" -> 0.0, "sigma" -> 1.0),
    epsilon = 0.05,
    numLeapfrog = 20
  )

  val samples = posterior.sample(hmc).take(1000).toSeq
//  println(samples)
  println("mean mu: " +samples.map(_.mu).sum / samples.size)
  println("mean sigma: " +samples.map(_.sigma).sum / samples.size)

