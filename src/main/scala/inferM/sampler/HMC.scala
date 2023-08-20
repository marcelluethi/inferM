package inferM.sampler

import inferM.*

import breeze.stats.{distributions => bdists}
import scalagrad.api.ScalaGrad
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.given

import inferM.RV.{LatentSample}
import breeze.linalg.DenseVector
import scalagrad.auto.breeze.BreezeDoubleMatrixAlgebra.given
import scalagrad.api.matrixalgebra.MatrixAlgebra
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode
import scalagrad.api.dual.DualColumnVector
import scalagrad.auto.breeze.BreezeDoubleMatrixAlgebraDSL
import scalagrad.api.matrixalgebra.MatrixAlgebraDSL
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.given
import BreezeDoubleForwardMode.{algebraT as alg}

/** Implementation of Hamiltonian Monte Carlo
  *
  * @param initialValue
  * @param epsilon
  * @param numLeapfrog
  * @param rng
  */
class HMC[A](using rng: breeze.stats.distributions.RandBasis)(
    initialValue: LatentSample[Double, DenseVector[Double]],
    epsilon: Double,
    numLeapfrog: Int
) extends Sampler[A, alg.Scalar, alg.ColumnVector]:

  /** Given values for the parameters (as a Map), the function returns a Map
    * with the same keys, but with the values replaced by the gradient
    */
  def gradDensity(rv: RV[A, alg.Scalar, alg.ColumnVector]): (
      latent: LatentSample[Double, DenseVector[Double]]
  ) => LatentSample[Double, DenseVector[Double]] = latentSample =>
    latentSample.map {
      case (name, _: Double) =>
        def density(s: alg.Scalar): alg.Scalar =
          rv.logDensity(liftArgsToDual(latentSample).updated(name, s))
        val grad: Double => Double = ScalaGrad.derive(density _)
        val pointToEvalute = latentSample(name)
        pointToEvalute match
          case s: Double => (name, grad(s))
          case _         => throw new Exception("Should not happen")
      case (name, _: DenseVector[Double]) =>
        def density(s: alg.ColumnVector): alg.Scalar =
          rv.logDensity(liftArgsToDual(latentSample).updated(name, s))
        val grad: DenseVector[Double] => DenseVector[Double] =
          ScalaGrad.derive((density))
        val pointToEvalute = latentSample(name)
        pointToEvalute match
          case v: DenseVector[Double] => (name, grad(v))
          case _ => throw new Exception("Should not happen")
    }

  def liftArgsToDual(
      latentSample: LatentSample[Double, DenseVector[Double]]
  ): LatentSample[alg.Scalar, alg.ColumnVector] =
    latentSample.map((name, value) =>
      value match
        case s: Double => (name, alg.liftToScalar(s))
        case v: DenseVector[Double] =>
          (
            name,
            alg.createColumnVectorFromElements(
              v.toScalaVector.map(alg.liftToScalar)
            )
          )
    )

  def sample(rv: RV[A, alg.Scalar, alg.ColumnVector]): Iterator[A] =
    import alg.*
    def U = (latentSample: LatentSample[Double, DenseVector[Double]]) =>
      val x = rv.logDensity(liftArgsToDual(latentSample))
      x * alg.liftToScalar(-1.0)

    def gradU(
        current: LatentSample[Double, DenseVector[Double]]
    ): LatentSample[Double, DenseVector[Double]] =
      gradDensity(rv)(current)
        .map(
          (name, value) => // make it negative, as U is also negated (see above)
            value match
              case s: Double =>
                (name, s * -1.0)
              case v: DenseVector[Double] => (name, v * -1.0)
        )

    def pStep(
        p: LatentSample[Double, DenseVector[Double]],
        q: LatentSample[Double, DenseVector[Double]],
        halfStep: Boolean
    ): LatentSample[Double, DenseVector[Double]] =
      if halfStep then
        p.zip(gradU(q))
          .map {
            case ((name, p: Double), (_, dUq: Double)) =>
              (name, p - epsilon * dUq / 2.0)
            case (
                  (name, p: DenseVector[Double]),
                  (_, dUq: DenseVector[Double])
                ) =>
              (name, p - epsilon * dUq / 2.0)
            case _ => throw new Exception("Should not happen")
          }
          .toMap
      else
        p.zip(gradU(q))
          .map {
            case ((name, p: Double), (_, dUq: Double)) =>
              (name, p - epsilon * dUq)
            case (
                  (name, p: DenseVector[Double]),
                  (_, dUq: DenseVector[Double])
                ) =>
              (name, p - epsilon * dUq)
            case _ => throw new Exception("Should not happen")
          }
          .toMap

    def qStep(
        p: LatentSample[Double, DenseVector[Double]],
        q: LatentSample[Double, DenseVector[Double]]
    ): LatentSample[Double, DenseVector[Double]] =
      q.zip(p)
        .map {
          case ((name, q: Double), (_, p: Double)) => (name, q + epsilon * p)
          case ((name, q: DenseVector[Double]), (_, p: DenseVector[Double])) =>
            (name, q + epsilon * p)
          case _ => throw new Exception("Should not happen")
        }
        .toMap

    def logProbP(p: LatentSample[Double, DenseVector[Double]]): Double =
      p.map {
        case (name, value: Double) => (name, value * value / 2.0)
        case (name, value: DenseVector[Double]) =>
          (name, breeze.linalg.sum(value *:* value) / 2.0)
        case _ => throw new Exception("Should not happen")
      }.values
        .reduce(_ + _)

    def oneStep(
        currentQ: LatentSample[Double, DenseVector[Double]]
    ): LatentSample[Double, DenseVector[Double]] =

      var q = currentQ
      var currentP = currentQ.map {
        case (name, _: Double) => (name, bdists.Gaussian(0.0, 1.0).sample())
        case (name, v: DenseVector[Double]) =>
          (name, DenseVector.rand(v.length, bdists.Gaussian(0.0, 1.0)))
      }

      // make half step
      var p = pStep(currentP, q, halfStep = true)

      for i <- 0 until numLeapfrog do
        q = qStep(p, q)
        if i <= numLeapfrog - 1 then p = pStep(p, q, halfStep = false)

      p = pStep(p, q, halfStep = true)
      p = p
        .map((name, p) =>
          p match
            case p: Double              => (name, -p)
            case p: DenseVector[Double] => (name, -p)
        )
        .toMap

      val currentqProb = U(currentQ).toDouble
      val currentkProb = logProbP(currentP)
      val proposedQProb = U(q).toDouble
      val proposedkProb = logProbP(p)

      val a = currentqProb - proposedQProb + currentkProb - proposedkProb
      val r = rng.uniform.draw()

      if (r < Math.exp(a.toDouble)) then q else currentQ

    Iterator
      .iterate(initialValue)(currentSample => oneStep(currentSample))
      .map(params => rv.value(liftArgsToDual(params)))
