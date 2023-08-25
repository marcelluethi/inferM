package inferM.examples

import inferM.*
import inferM.dists.*
import inferM.sampler.*

import scalagrad.api.spire.trig.DualScalarIsTrig.given
import spire.implicits.DoubleAlgebra
import scalagrad.api.spire.numeric.DualScalarIsNumeric.given
import spire.compat.numeric
import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import scalagrad.api.matrixalgebra.MatrixAlgebra
import scalagrad.api.matrixalgebra.MatrixAlgebraDSL
import scalagrad.auto.breeze.BreezeDoubleMatrixAlgebra
import scalagrad.auto.breeze.BreezeDoubleMatrixAlgebraDSL

object LoadedCoin extends App:

  // example data
  val pGroundTruth = 0.8
  val data = bdists.Bernoulli(pGroundTruth).sample(100)

  def posteriorF(alg: MatrixAlgebraDSL): RV[Double, alg.Scalar, alg.ColumnVector] = {
    given MatrixAlgebra[alg.Scalar, alg.ColumnVector, _, _] = alg.innerAlgebra
    
    // P(w)
    val prior =
      for p <- Uniform(alg.lift(0.0), alg.lift(1.0)).toRV("p")
      yield p

    // P(D|w)
    val likelihood = (p2: alg.Scalar) =>
      val p = alg.lift(alg.unlift(p2)) // stop gradients => why?
      val targetDist = Bernoulli(p)

      data.foldLeft(alg.zeroScalar)((sum, x) =>
        sum + targetDist.logPdf(
          if x == false then alg.lift(0.0) else alg.lift(1.0)
        )
      )

    // P(w|D) = P(w)P(D|w)
    val posterior = prior.condition(likelihood)
    posterior.map(_.toDouble)
  }

  // Sampling
  val hmc = HMC[Double](
    initialValue = Map("p" -> 0.5),
    epsilon = 0.05,
    numLeapfrog = 20
  )

  val samples = hmc.sample(posteriorF).take(1000).toSeq
  println("mean p: " + samples.sum / samples.size)

  /*val metro = MetropolisHastings[Double](
    initialValue = Map("p" -> 0.5),
    proposal = x => x.updatedWith("p")(p => p.map(_.asInstanceOf[Double] + Gaussian(0.0, 0.1)(using BreezeDoubleMatrixAlgebra).draw())),
  )(using BreezeDoubleMatrixAlgebra)

  val samples2 = metro.sample(posteriorF(BreezeDoubleMatrixAlgebraDSL)).take(1000).toSeq
  println("mean p: " + samples2.sum / samples2.size)*/