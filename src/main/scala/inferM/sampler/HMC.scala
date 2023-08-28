package inferM.sampler

import inferM.*

import breeze.stats.{distributions => bdists}
import scalagrad.api.forward.ForwardMode

import inferM.RV.{LatentSample}
import breeze.linalg.DenseVector
import scalagrad.auto.breeze.BreezeDoubleMatrixAlgebra.given
import scalagrad.api.matrixalgebra.MatrixAlgebra
import scalagrad.api.dual.DualColumnVector
import scalagrad.auto.breeze.BreezeDoubleMatrixAlgebraDSL
import scalagrad.api.matrixalgebra.MatrixAlgebraDSL
import breeze.linalg.*

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
):

  private val primaryAlg = BreezeDoubleMatrixAlgebraDSL

  /** Given values for the parameters (as a Map), the function returns a Map
    * with the same keys, but with the values replaced by the gradient
    */
  def gradDensity(rvF: (alg: MatrixAlgebraDSL) => RV[A, alg.Scalar, alg.ColumnVector]):
    LatentSample[Double, DenseVector[Double]] => LatentSample[Double, DenseVector[Double]] = 
      latentSample =>
        latentSample.map {
          case (name, s: Double) =>
            def density(alg: MatrixAlgebraDSL)(s: alg.Scalar): alg.Scalar =
              rvF(alg).logDensity(liftArgsToDual(alg)(latentSample).updated(name, s))
            val grad: Double => Double = ForwardMode.derive(density)(primaryAlg)
            (name, grad(s))
          case (name, cv: DenseVector[Double]) =>
            def density(alg: MatrixAlgebraDSL)(s: alg.ColumnVector): alg.Scalar =
              rvF(alg).logDensity(liftArgsToDual(alg)(latentSample).updated(name, s))
            val grad: DenseVector[Double] => DenseVector[Double] =
              ForwardMode.derive(density)(primaryAlg)
            (name, grad(cv))
        }

  def liftArgsToDual(alg: MatrixAlgebraDSL)(
      latentSample: LatentSample[Double, DenseVector[Double]]
  ): LatentSample[alg.Scalar, alg.ColumnVector] =
    latentSample.map((name, value) =>
      value match
        case s: Double => (name, alg.lift(s))
        case v: DenseVector[Double] => (name, alg.lift(v))
    )

  // def sample(rv: RV[A, alg.Scalar, alg.ColumnVector]): Iterator[A] =
  def sample(rvF: (alg: MatrixAlgebraDSL) => RV[A, alg.Scalar, alg.ColumnVector]): Iterator[A] =
    val rv = rvF(BreezeDoubleMatrixAlgebraDSL)

    def U = (latentSample: LatentSample[Double, DenseVector[Double]]) =>
      val x = rv.logDensity(latentSample)
      x * -1.0

    def gradU(
        current: LatentSample[Double, DenseVector[Double]]
    ): LatentSample[Double, DenseVector[Double]] =
      gradDensity(rvF)(current)
        .map(
          (name, value) => // make it negative, as U is also negated (see above)
            value match
              case s: Double => (name, s * -1.0)
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
              (name, p -  dUq * epsilon / 2.0)
            case (
                  (name, p: DenseVector[Double]),
                  (_, dUq: DenseVector[Double])
                ) =>
              (name, p -  dUq * epsilon / 2.0)
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
              (name, p -  dUq * epsilon)
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
            (name, q +  p * epsilon)
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
      .map(params => rv.value(params))
