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


object LogisticRegression extends App:

  // example data
  val aGroundTruth = 1.0
  val bGroundTruth = 0.0
  
  def invlogit(x: alg.Scalar): alg.Scalar =
    alg.lift(1.0) / (alg.lift(1.0) + alg.trig.exp(-x))

  val data = Seq.range(-50, 50).map(_ / 10.0).map(s => if s  < 3 then (s, 0) else (s, 1))

  println("data: " + data)

  case class Parameters(a: alg.Scalar, b: alg.Scalar)

  val prior = for
    a <- Gaussian(alg.lift(0.0), alg.lift(10)).toRV("a")
    b <- Gaussian(alg.lift(0.0), alg.lift(10)).toRV("b")
  yield Parameters(a, b)


  val likelihood = (parameters: Parameters) =>
    data.foldLeft(alg.zeroScalar)((sum, point) =>
      val (x, y) = point
      sum + Binomial(1, 
        invlogit(parameters.a * alg.lift(x) + parameters.b)
      ).logPdf(alg.lift(y))
    )

  val posterior = prior.condition(likelihood)

 
  // Sampling
  val hmc = HMC[Parameters](
    initialValue = Map("a" -> 0.0, "b" -> 0.0),
    epsilon = 0.01,
    numLeapfrog = 10
  )

  val samples = posterior.sample(hmc).drop(10000).take(10000).toSeq

  println(for x <- data.map(_._1) yield (x, invlogit(samples.last.value.a * alg.lift(x) + samples.last.value.b).value.round))

  println("mean a: " + samples.map(_.value._1.toDouble).sum / samples.size)
  println("mean b: " + samples.map(_.value._2.toDouble).sum / samples.size)
  
