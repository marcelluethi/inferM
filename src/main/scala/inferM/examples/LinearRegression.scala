package inferM.examples

import inferM.*
import inferM.dists.*
import inferM.sampler.*


import scalagrad.api.spire.trig.DualScalarIsTrig.given
import spire.implicits.DoubleAlgebra 

import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import scalagrad.auto.forward.breeze.DeriverBreezeDoubleForwardPlan
import scalagrad.auto.forward.breeze.DeriverBreezeDoubleForwardPlan.given
import DeriverBreezeDoubleForwardPlan.{algebraT as alg}

object LinearRegression extends App:  

  // example data
  val aGroundTruth = 3.0
  val bGroundTruth = 2.0
  val noiseSigma = 1.0

  val data = Seq.range(0, 10).map(x => 
    (x.toDouble, aGroundTruth * x + bGroundTruth + bdists.Gaussian(0.0, noiseSigma).sample())
  )

  case class Parameters(a : Double, b : Double)

  val prior = for  
    a <- RV.fromPrimitive(Gaussian(alg.liftToScalar(1.0), alg.liftToScalar(10)), "a")
    b <- RV.fromPrimitive(Gaussian(alg.liftToScalar(2.0), alg.liftToScalar(10)), "b")
  yield Parameters(a, b)

  
  val likelihood =  (parameters : Parameters) => 
    data.foldLeft(alg.zeroScalar)((sum, point) =>
      val (x, y) = point 
      sum + Gaussian(
        alg.liftToScalar(parameters.a)  * alg.liftToScalar(x) +alg.liftToScalar(parameters.b), 
        alg.liftToScalar(1.0)).logPdf(alg.liftToScalar(y)
      )
    )

  val posterior = prior.condition(likelihood)

  // Sampling
  val hmc = HMC[Parameters](
    initialValue = Map("a" -> 0.0, "b" -> 0.0),
    epsilon = 0.5,
    numLeapfrog = 20
  )

  val samples = posterior.sample(hmc).take(10000).toSeq

  println("mean a: " +samples.map(_._1).sum / samples.size)
  println("mean b: " +samples.map(_._2).sum / samples.size)

