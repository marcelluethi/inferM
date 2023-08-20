package inferM.sampler

import inferM.*
import inferM.RV.{LatentSample}
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.given
import breeze.linalg.DenseVector
import scalagrad.api.matrixalgebra.MatrixAlgebra

/** Implementation of the Metropolis Hastings algorithm
  */
class MetropolisHastings[A, S, CV](
    initialValue: LatentSample[S, CV],
    proposal: LatentSample[S, CV] => LatentSample[S, CV]
)(using
    alg: MatrixAlgebra[S, CV, _, _],
    rng: breeze.stats.distributions.RandBasis
) extends Sampler[A, S, CV]:
  def sample(rv: RV[A, S, CV]): Iterator[A] =

    def liftSample(sample: LatentSample[S, CV]): LatentSample[S, CV] =
      sample.map {
        case (name, value: Double) => (name, alg.liftToScalar(value))
        case (name, value: DenseVector[Double @unchecked]) =>
          (
            name,
            alg.createColumnVectorFromElements(
              value.toScalaVector.map(alg.liftToScalar)
            )
          )
        case (_, _) => throw new Exception("Should not happen")
      }

    def oneStep(
        currentSample: LatentSample[S, CV],
        newProposal: LatentSample[S, CV]
    ): LatentSample[S, CV] =

      val currentP = rv.logDensity(liftSample(currentSample))
      val newP = rv.logDensity(liftSample(newProposal))
      val a = newP - currentP
      val r = rng.uniform.draw()

      if (a.toDouble > 0.0 || r < Math.exp(a.toDouble)) then newProposal
      else currentSample

    Iterator
      .iterate(initialValue)(currentSample =>
        oneStep(currentSample, proposal(currentSample))
      )
      .map(params => rv.value(liftSample(params)))
