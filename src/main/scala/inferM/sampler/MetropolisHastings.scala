package inferM.sampler

import inferM.*
import inferM.RV.{LatentSample, LatentSampleDouble}
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode
import BreezeDoubleForwardMode.given
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.{algebraT => alg}
import breeze.linalg.DenseVector
import scalagrad.api.matrixalgebra.MatrixAlgebra

/** Implementation of the Metropolis Hastings algorithm
  */
class MetropolisHastings[A](
    initialValue: LatentSampleDouble,
    proposal: LatentSampleDouble => LatentSampleDouble
)(using rng: breeze.stats.distributions.RandBasis
) extends Sampler[A]:
  def sample(rv: RV[A]): Iterator[A] =

    def liftSample(sample: LatentSampleDouble): LatentSample =
      sample.map {
        case (name, value: Double) => (name, alg.lift(value))
        case (name, value: DenseVector[Double @unchecked]) =>
          ( name,  alg.lift(value) )
      }

    def oneStep(
        currentSample: LatentSampleDouble,
        newProposal: LatentSampleDouble
    ): LatentSampleDouble =

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
