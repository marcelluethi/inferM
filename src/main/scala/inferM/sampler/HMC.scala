package inferM.sampler

import inferM.*


import breeze.stats.{distributions => bdists}
import scalagrad.api.ScalaGrad
import scalagrad.auto.forward.breeze.DeriverBreezeDoubleForwardPlan.{algebraT as alg}
import scalagrad.auto.forward.breeze.DeriverBreezeDoubleForwardPlan.given


class HMC[A](initialValue : Map[String, Double], epsilon : Double, numLeapfrog : Int)(using rng : breeze.stats.distributions.RandBasis) extends Sampler[A]:

  /**
    * Given values for the parameters (as a Map), the function returns 
    * a Map with the same keys, but with the values replaced by the gradient
    */
  def gradDensity(rv : RV[A]) : (params : Latent) => Map[String, Double] = params =>

    params.map((name, _) => 
      val grad = ScalaGrad.derive((s : alg.Scalar) => rv.density(params.updated(name, s)))
      val pointToEvalute = params(name)
      (name, grad(pointToEvalute.toDouble))
    )          
    

  def sample(rv : RV[A]) : Iterator[A] = 
    
    def U = (params : Map[String, Double]) => 
      -rv.density(params.map((name, value) => (name, alg.liftToScalar(value)))).toDouble
    
    def gradU(current : Map[String, Double]) : Map[String, Double] = 
      val liftedArg = current.map((name, value) => (name, alg.liftToScalar( - value))) // ! see the minus sign here
      gradDensity(rv)(liftedArg).map((name, value) => (name, value.toDouble))

    def oneStep(currentQ : Map[String, Double]) : Map[String, Double] =       
      
      var q = currentQ
      var currentP = currentQ.map((name, _) => (name, bdists.Gaussian(0.0, 1.0).sample()))
      
      // make half step
      var p = currentP.zip(gradU(q)).map{case ((name, p), (_, dUq)) => (name, p - epsilon * dUq / 2.0)}.toMap
      
      for i <- 0 until numLeapfrog do
        q = q.zip(p).map{case ((name, q), (_, p)) => (name, q + epsilon * p)}.toMap
        if i <= numLeapfrog - 1 then           
           p = p.zip(gradU(q)).map{case ((name, p), (_, dUq)) => (name, p - epsilon * dUq)}.toMap

      p = p.zip(gradU(q)).map{case ((name, p), (_, dUq)) => (name, p - epsilon * dUq / 2.0)}.toMap
      p = p.map((name, p) => (name, -p)).toMap

      val currentqProb = U(currentQ).toDouble
      val currentkProb = currentP.map((name, value) => (name, value * value / 2.0)).values.reduce(_ + _)
      val proposedQProb = U(q).toDouble
      val proposedkProb = p.map((name, value) => (name, value * value / 2.0)).values.reduce(_ + _)

      val a = currentqProb - proposedQProb + currentkProb - proposedkProb
      val r = rng.uniform.draw()

      if (r < Math.exp(a.toDouble)) then q else currentQ      

    Iterator
      .iterate(initialValue)(currentSample =>
        oneStep(currentSample)
    ).map(params => rv.value(params.map((name, value) => (name, alg.liftToScalar(value)))))



