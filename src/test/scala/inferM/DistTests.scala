package inferM

import inferM.dists.Uniform
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.{algebraT as alg}
import inferM.sampler.HMC
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import inferM.dists.Exponential
import inferM.dists.Gaussian
import inferM.dists.Bernoulli


class MyTests extends munit.FunSuite:
  test("generates correct prior samples from uniform distribution") {
    val (a, b) = (2.0, 4.0)
    val uniform = Uniform(alg.lift(a), alg.lift(b)).toRV("x")
    val samples = uniform.sample(HMC(Map("x" -> 0.0), 0.1, 10)).drop(1000).take(10000).toSeq
    val mean = samples.sum / samples.length
    val expectedMean = a + (b - a) / 2
    val variance = samples.map(x => (x - mean) * (x - mean)).sum / samples.length
    val expectedVariance = 1.0 / 12 * Math.pow(b - a, 2)

    assert(Math.abs(mean - expectedMean) < 1e-1)
    assert(Math.abs(variance - expectedVariance) < 1e-1)
  }

  test("generates correct prior samples from an exponential distribution") {
    val lambda = 2.0
    val exponential =Exponential(alg.lift(lambda)).toRV("lambda")
    val samples = exponential.sample(HMC(Map("lambda" -> 1.0), 0.01, 10)).drop(1000).take(10000).toSeq
    val mean = samples.sum / samples.length
    val expectedMean = 1.0 / lambda
    val variance = samples.map(x => (x - mean) * (x - mean)).sum / samples.length
    val expectedVariance = 1.0 / (lambda * lambda)
    assert(Math.abs(mean - expectedMean) < 1e-1)
    assert(Math.abs(variance - expectedVariance) < 1e-1)
  }

  test("generates correct prior samples from a gaussian distribution") {
    val mu = 5.0
    val sigma = 2.0
    val gaussian = Gaussian(alg.lift(mu), alg.lift(sigma)).toRV("x")
    val samples = gaussian.sample(HMC(Map("x" -> 3.0), 1e-1, 10)).drop(5000).take(100000).toSeq
    val mean = samples.sum / samples.length
    val expectedMean = mu
    val variance = samples.map(x => (x - mean) * (x - mean)).sum / samples.length
    val expectedVariance = sigma * sigma
    assert(Math.abs(mean - expectedMean) < 1e-1)
    assert(Math.abs(variance - expectedVariance) < 5e-1)
  }
