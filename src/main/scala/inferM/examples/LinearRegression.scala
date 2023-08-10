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
  val bGroundTruth = 10.0
  val noiseSigma = 1.0

  val data = Seq.range(0, 10).map(x => 
    (x.toDouble, aGroundTruth * x + bGroundTruth + bdists.Gaussian(0.0, noiseSigma).sample())
  )

  // defining the regression model 
  val prior = for  
    a <- RV.fromPrimitive(Gaussian(alg.liftToScalar(1.0), alg.liftToScalar(10)), "a")
    b <- RV.fromPrimitive(Gaussian(alg.liftToScalar(2.0), alg.liftToScalar(10)), "b")
  yield (a, b)

  def addPoint(x : Double, y : Double)(prior : RV[(Double, Double)]) : RV[(Double, Double)] = 
    prior.condition((a, b) => 
      Gaussian(alg.liftToScalar(a)  * alg.liftToScalar(x) +alg.liftToScalar(b), alg.liftToScalar(1.0)).logPdf(alg.liftToScalar(y)))
  
  val posterior = data.foldLeft(prior)((prior, point) => addPoint(point._1, point._2)(prior))

  // Sampling
  val hmc = HMC[(Double, Double)](
    initialValue = Map("a" -> 0.0, "b" -> 0.0),
    epsilon = 0.5,
    numLeapfrog = 20
  )

  val samples = posterior.sample(hmc).take(10000).toSeq

  println("mean a: " +samples.map(_._1).sum / samples.size)
  println("mean b: " +samples.map(_._2).sum / samples.size)

