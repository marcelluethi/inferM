package inferM.sampler

import inferM.*

import inferM.RV.{LatentSample, LatentSampleDouble}
import breeze.linalg.DenseVector

import breeze.stats.{distributions => bdists}


import scalagrad.auto.forward.BreezeDoubleForwardDualMode.derive as d
import scalagrad.auto.forward.BreezeDoubleForwardDualMode.algebra.*  // import syntax
import scalagrad.auto.forward.BreezeDoubleForwardDualMode.algebraDSL as alg
import scalagrad.api.forward.ForwardMode
import scalagrad.auto.breeze.BreezeDoubleMatrixAlgebraDSL

/** Implementation of Hamiltonian Monte Carlo
  *
  * @param initialValue
  * @param epsilon
  * @param numLeapfrog
  * @param rng
  */
class HMC[A](using rng: breeze.stats.distributions.RandBasis)(
    initialValue: LatentSampleDouble,
    epsilon: Double,
    numLeapfrog: Int
) extends Sampler[A]:

  /** Given values for the parameters (as a Map), the function returns a Map
    * with the same keys, but with the values replaced by the gradient
    */
  def gradDensity(rv: RV[A]): (
      latent: LatentSampleDouble
  ) => LatentSampleDouble = latentSample =>
    latentSample.map {
      case (name, _: Double) =>
        def density(s: alg.Scalar): alg.Scalar =
          rv.logDensity(liftArgsToDual(latentSample).updated(name, s))
        val grad: Double => Double = d(density _)
        val pointToEvalute = latentSample(name)
        pointToEvalute match
          case s: Double => (name, grad(s))
          case _         => throw new Exception("Should not happen")
      case (name, _: DenseVector[Double]) =>
        def density(s: alg.ColumnVector): alg.Scalar =
          rv.logDensity(liftArgsToDual(latentSample).updated(name, s))
        val grad: DenseVector[Double] => DenseVector[Double] =
          d(density)
        val pointToEvalute = latentSample(name)
        pointToEvalute match
          case v: DenseVector[Double] => (name, grad(v))
          case _ => throw new Exception("Should not happen")
    }

  def liftArgsToDual(
      latentSample: LatentSampleDouble
  ): LatentSample =
    latentSample.map((name, value) =>
      value match
        case s: Double => (name, alg.lift(s))
        case v: DenseVector[Double] =>
          (
            name,
            alg.createColumnVectorFromElements(
              v.toScalaVector.map(alg.lift)
            )
          )
    )

  def sample(rv: RV[A]): Iterator[Sample[A]] =
    import alg.*
    def U = (latentSample: LatentSampleDouble) =>
      val x = rv.logDensity(liftArgsToDual(latentSample))
      x * alg.lift(-1.0)

    def gradU(
        current: LatentSampleDouble
    ): LatentSampleDouble =
      gradDensity(rv)(current)
        .map(
          (name, value) => // make it negative, as U is also negated (see above)
            value match
              case s: Double =>
                (name, s * -1.0)
              case v: DenseVector[Double] => (name, v * -1.0)
        )

    def pStep(
        p: LatentSampleDouble,
        q: LatentSampleDouble,
        halfStep: Boolean
    ): LatentSampleDouble =
      if halfStep then
        p.zip(gradU(q))
          .map {
            case ((name, p: Double), (_, dUq: Double)) =>
              (name, p -  dUq * epsilon / 2.0)
            case (
                  (name, p: DenseVector[Double]),
                  (_, dUq: DenseVector[Double])
                ) =>
              (name, p -  dUq * epsilon/ 2.0)
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
        p: LatentSampleDouble,
        q: LatentSampleDouble
    ): LatentSampleDouble =
      q.zip(p)
        .map {
          case ((name, q: Double), (_, p: Double)) => (name, q + epsilon * p)
          case ((name, q: DenseVector[Double]), (_, p: DenseVector[Double])) =>
            (name, q +  p * epsilon)
          case _ => throw new Exception("Should not happen")
        }
        .toMap

    def logProbP(p: LatentSampleDouble): Double =
      p.map {
        case (name, value: Double) => (name, value * value / 2.0)
        case (name, value: DenseVector[Double]) =>
          (name, breeze.linalg.sum(value *:* value) / 2.0)
        case _ => throw new Exception("Should not happen")
      }.values
        .reduce(_ + _)

    def oneStep(
        currentQ: LatentSampleDouble
    ): LatentSampleDouble =

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
      .map(params => 
        val paramsLifted = liftArgsToDual(params)
        Sample(rv.value(paramsLifted), rv.logDensity(paramsLifted).value))
