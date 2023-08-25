package inferM.examples

import inferM.*
import inferM.dists.*
import inferM.sampler.*

import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.given
import BreezeDoubleForwardMode.{algebraT as alg}

import breeze.stats.distributions.Rand.FixedSeed.randBasis
import scaltair.Chart
import scaltair.Channel
import scaltair.FieldType
import scaltair.Mark

import scaltair.PlotTargetBrowser.plotTargetBrowser

@main
def jointGaussian(): Unit = 
  
  val prior = 
    for 
      x <- Uniform(alg.lift(0.0), alg.lift(1.0)).toRV("x")
      y <- Gaussian(x, alg.lift(1.0)).toRV("y")
    yield (x, y)

  val hmc = HMC[(alg.Scalar, alg.Scalar)](
    initialValue = Map("x" -> 0.0, "y" -> 0.0),
    epsilon = 0.1,
    numLeapfrog = 20
  )


  val samples = prior.sample(hmc).drop(1000).take(1000).toSeq
  
  Chart(Map("x" -> samples.map(_._1.toDouble), "y" -> samples.map(_._2.toDouble)))
  .encode(
    Channel.X("x", FieldType.Quantitative),
    Channel.Y("y", FieldType.Quantitative)
  ).mark(Mark.Point())
  .show()
