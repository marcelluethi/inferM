package inferM.sampler

import inferM.* 
import scalagrad.auto.forward.breeze.DeriverBreezeDoubleForwardPlan
import scalagrad.auto.forward.breeze.DeriverBreezeDoubleForwardPlan.given
import DeriverBreezeDoubleForwardPlan.{algebraT as alg}

class MetropolisHastings[A](initialValue : Map[String, Double], proposal : Map[String, Double] => Map[String, Double])(using rng : breeze.stats.distributions.RandBasis) extends Sampler[A]:
  def sample(rv : RV[A]) : Iterator[A] = 
    
    def oneStep(
        currentSample: Map[String, Double],
        newProposal: Map[String, Double]
    ):  Map[String, Double] =

      val currentP = rv.density(currentSample.map((name, value) => (name, alg.liftToScalar(value))))
      val newP = rv.density(newProposal.map((name, value) => (name, alg.liftToScalar(value))))
      val a = newP - currentP
      val r = rng.uniform.draw()

      if (a.toDouble > 0.0 || r < Math.exp(a.toDouble)) then newProposal
      else currentSample

    Iterator
      .iterate(initialValue)(currentSample =>
        oneStep(currentSample, proposal(currentSample))
    ).map(params => rv.value(params.map((name, value) => (name, alg.liftToScalar(value)))))


