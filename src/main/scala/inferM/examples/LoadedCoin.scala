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

object LoadedCoin extends App:  

  // example data
  val pGroundTruth = 0.8
  val data = bdists.Bernoulli(pGroundTruth).sample(100)

  val prior = for  
    p <- Uniform(alg.liftToScalar(0.0), alg.liftToScalar(1.0)).toRV("p")
  yield p

  
  val likelihood =  (p : Double) => 
    val targetDist = Bernoulli(alg.liftToScalar(p))
    
    data.foldLeft(alg.zeroScalar)((sum, x) =>
      sum + targetDist.logPdf(x)
    )
    
  val posterior = prior.condition(likelihood)

  // Sampling
  val hmc = HMC[Double](
    initialValue = Map("p" -> 0.5),
    epsilon = 0.05,
    numLeapfrog = 20
  )

  val samples = posterior.sample(hmc).take(1000).toSeq
  println("mean p: " +samples.sum / samples.size)

