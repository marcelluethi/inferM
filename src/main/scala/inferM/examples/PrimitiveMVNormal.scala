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

object PrimitiveMVNormal extends App:  

  // example data
  val mean = alg.lift(DenseVector(1.0, 2.0))
  val cov = alg.lift(DenseMatrix((1.0, 0.5), (0.5, 1.0)))

  val prior = for  
    x <- RV.fromPrimitiveMultivariate(MultivariateGaussian(mean,  cov), "x")
  yield x

  
  // Sampling
  val hmc = HMC[DenseVector[Double]](
    initialValue = Map("x" -> DenseVector.zeros[Double](2)),
    epsilon = 0.05,
    numLeapfrog = 20
  )

  val samples = prior.sample(hmc).drop(1000).take(10000).toSeq

  val estimatedMean = samples.reduce(_ + _) * (1.0 /  samples.size)
  val estimatedCov = samples.foldLeft(DenseMatrix.zeros[Double](2, 2))((sum, sample) => 
    sum + (sample - estimatedMean) * (sample - estimatedMean).t
  ) * (1.0 / samples.size)

  println("mean " + estimatedMean)
  println("cov " + estimatedCov)