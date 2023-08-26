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

object MyGaussianOnGaussian extends App:

  def posteriorF(alg: MatrixAlgebraDSL): RV[(Double, Double), alg.Scalar, alg.ColumnVector] = {

    given MatrixAlgebra[alg.Scalar, alg.ColumnVector, _, _] = alg.innerAlgebra

    for {
      x <- Gaussian(alg.lift(0.0), alg.lift(1.0)).toRV("x")
      y <- Gaussian(x, alg.lift(0.1)).toRV("y")
    } yield (x.toDouble, y.toDouble)
  }

  // Sampling
  val hmc = HMC[(Double, Double)](
    initialValue = Map("x" -> 0.5, "y" -> 0.5),
    epsilon = 0.05,
    numLeapfrog = 20
  )

  val samples = hmc.sample(posteriorF).drop(100).take(10).toSeq
  println("mean p: " + samples)
