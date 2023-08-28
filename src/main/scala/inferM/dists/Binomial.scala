package inferM.dists
import inferM.Dist

import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode
import BreezeDoubleForwardMode.given
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.{algebraT => alg}

import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.given
import scalagrad.api.matrixalgebra.MatrixAlgebra


class Binomial(n : Integer, p: alg.Scalar) extends Dist:

  def value(s : alg.Scalar) : alg.Scalar =  if (s.value <= p.value) alg.lift(1.0) else alg.lift(0.0)

  def logPdf(x: alg.Scalar): alg.Scalar =
    // it is okay to use value here because the gradient as it does not make sense to do gradient based inference on discrete variables
    val k = x.value.toInt 
    alg.trig.log(alg.liftToScalar(combinations(n, k).toInt)) + alg.trig.log(alg.num.pow(p, k)) + alg.trig.log(alg.num.pow(alg.lift(1.0) - p, n - k))
  
  def draw(): Int =
    val dist = bdists.Binomial(n, alg.unliftToDouble(p))
    dist.draw()


  private def permutations(n: Int): BigInt =
  (1 to n).map(BigInt(_)).foldLeft(BigInt(1))(_ * _)

  private def combinations(n: Int, k: Int): BigInt =
  permutations(n) / (permutations(k) * permutations(n - k))
