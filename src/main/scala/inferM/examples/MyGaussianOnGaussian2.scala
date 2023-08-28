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
import scalagrad.auto.breeze.BreezeDoubleMatrixAlgebraDSL

object MyGaussianOnGaussian2 extends App:

  type ->[K <: String, V] = (K, V)

  def posteriorF(alg: MatrixAlgebraDSL): NewRV[(Double, Double), alg.Scalar, alg.ColumnVector, (("x" -> alg.Scalar), ("y" -> alg.Scalar))] = {

    given MatrixAlgebra[alg.Scalar, alg.ColumnVector, _, _] = alg.innerAlgebra

    val test = for {
      x <- Gaussian(alg.lift(0.0), alg.lift(1.0)).toNewRV("x")
      y <- Gaussian(x, alg.lift(0.1)).toNewRV("y")
    } yield (x.toDouble, y.toDouble)

    test
  }

  println(posteriorF(BreezeDoubleMatrixAlgebraDSL).logDensity(("x" -> 0.5, "y" -> 0.5)))
