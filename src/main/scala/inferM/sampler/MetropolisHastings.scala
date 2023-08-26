package inferM.sampler

import inferM.*
import inferM.RV.{LatentSample}
import breeze.linalg.DenseVector
import scalagrad.api.matrixalgebra.MatrixAlgebra

/** Implementation of the Metropolis Hastings algorithm
  */
class MetropolisHastings[A](
    initialValue: LatentSample[Double, DenseVector[Double]],
    proposal: LatentSample[Double, DenseVector[Double]] => LatentSample[Double, DenseVector[Double]]
)(using
    alg: MatrixAlgebra[Double, DenseVector[Double], _, _],
    rng: breeze.stats.distributions.RandBasis
):
  def sample(rv: RV[A, Double, DenseVector[Double]]): Iterator[A] =

    def liftSample(sample: LatentSample[Double, DenseVector[Double]]): LatentSample[Double, DenseVector[Double]] =
      sample.map {
        case (name, value: Double) => (name, alg.lift(value))
        case (name, value: DenseVector[Double @unchecked]) =>
          (
            name,
            alg.createColumnVectorFromElements(
              value.toScalaVector.map(alg.lift)
            )
          )
        case (_, _) => throw new Exception("Should not happen")
      }

    def oneStep(
        currentSample: LatentSample[Double, DenseVector[Double]],
        newProposal: LatentSample[Double, DenseVector[Double]]
    ): LatentSample[Double, DenseVector[Double]] =

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
