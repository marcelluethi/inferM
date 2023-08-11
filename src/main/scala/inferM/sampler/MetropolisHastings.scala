package inferM.sampler

import inferM.* 
import inferM.RV.LatentSampleDouble
import scalagrad.auto.forward.breeze.DeriverBreezeDoubleForwardPlan
import scalagrad.auto.forward.breeze.DeriverBreezeDoubleForwardPlan.given
import DeriverBreezeDoubleForwardPlan.{algebraT as alg}

/**
  * Implementation of the Metropolis Hastings algorithm
  *
  */
class MetropolisHastings[A](initialValue : LatentSampleDouble, proposal :LatentSampleDouble => LatentSampleDouble)(using rng : breeze.stats.distributions.RandBasis) extends Sampler[A]:
  def sample(rv : RV[A]) : Iterator[A] = 
    
    def oneStep(
        currentSample: LatentSampleDouble,
        newProposal:LatentSampleDouble
    ): LatentSampleDouble =

      val currentP = rv.logDensity(currentSample.map((name, value) => (name, alg.liftToScalar(value))))
      val newP = rv.logDensity(newProposal.map((name, value) => (name, alg.liftToScalar(value))))
      val a = newP - currentP
      val r = rng.uniform.draw()

      if (a.toDouble > 0.0 || r < Math.exp(a.toDouble)) then newProposal
      else currentSample

    Iterator
      .iterate(initialValue)(currentSample =>
        oneStep(currentSample, proposal(currentSample))
    ).map(params => rv.value(params.map((name, value) => (name, alg.liftToScalar(value)))))


