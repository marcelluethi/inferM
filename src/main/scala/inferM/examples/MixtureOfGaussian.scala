// package inferM.examples


// import inferM.*
// import inferM.dists.*
// import inferM.sampler.*

// import scalagrad.api.spire.trig.DualScalarIsTrig.given
// import spire.implicits.DoubleAlgebra 

// import breeze.stats.{distributions => bdists}
// import breeze.stats.distributions.Rand.FixedSeed.randBasis

// import scalagrad.auto.forward.breeze.DeriverBreezeDoubleForwardPlan
// import scalagrad.auto.forward.breeze.DeriverBreezeDoubleForwardPlan.given
// import DeriverBreezeDoubleForwardPlan.{algebraT as alg}

// object MixtureOfGaussian extends App:  

//   // example data
//   val mu1GroundTruth = -5.0
//   val sigma1GroundTruth = 1.0
//   val gaussian1 = bdists.Gaussian(mu1GroundTruth, sigma1GroundTruth)

//   val mu2GroundTruth = 5.0
//   val sigma2GroundTruth = 1.0
//   val gaussian2 = bdists.Gaussian(mu2GroundTruth, sigma2GroundTruth)
//   val wGroundTruth = 0.1
//   val bernulli = bdists.Bernoulli(wGroundTruth)

//   val data = for i <- 0 until 100 yield
//     if bernulli.draw() then gaussian1.draw() else gaussian2.draw()  



//   case class Parameters(mu1 : Double, mu2 : Double, w : Double)

//   val prior = for  
//     mu1 <- RV.fromPrimitive(Gaussian(alg.liftToScalar(-3.0), alg.liftToScalar(10)), "mu")
//     mu2 <- RV.fromPrimitive(Gaussian(alg.liftToScalar(4.0), alg.liftToScalar(10)), "sigma1")
//     w <- RV.fromPrimitive(Gaussian(alg.liftToScalar(0.5), alg.liftToScalar(1.0)), "w")
//   yield Parameters(mu1, mu2, w)

//   val likelihood =  (parameters : Parameters) => 
//     val gauss1 = Gaussian(
//         alg.liftToScalar(parameters.mu1),
//         alg.liftToScalar(sigma1GroundTruth)
//       )
//     val gauss2 = Gaussian(
//         alg.liftToScalar(parameters.mu2),
//         alg.liftToScalar(sigma1GroundTruth)
//       )
//     val bernoulli = Bernoulli(
//         alg.liftToScalar(parameters.w)
//       )
//     data.foldLeft(alg.zeroScalar)((sum, point) =>
//       sum + gauss1.logPdf
//     )
    
//   val posterior = prior.condition(likelihood)

//   // Sampling
//   val hmc = HMC[Parameters](
//     initialValue = Map("mu" -> 0.0, "sigma" -> 1.0),
//     epsilon = 0.05,
//     numLeapfrog = 20
//   )

//   val samples = posterior.sample(hmc).take(1000).toSeq
// //  println(samples)
//   println("mean mu: " +samples.map(_.mu).sum / samples.size)
//   println("mean sigma: " +samples.map(_.sigma).sum / samples.size)


