package inferM

import scalagrad.api.matrixalgebra.MatrixAlgebraT
import spire.implicits.DoubleAlgebra 
import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis


import scalagrad.api.ScalaGrad
import scalagrad.auto.forward.breeze.DeriverBreezeDoubleForwardPlan
import scalagrad.auto.forward.breeze.DeriverBreezeDoubleForwardPlan.given
import DeriverBreezeDoubleForwardPlan.{algebraT as alg}

type Latent = Map[String, alg.Scalar]

class RV[A](val value : Latent => A, val density : Latent => alg.Scalar):


  /**
    * Given values for the parameters (as a Map), the function returns 
    * a Map with the same keys, but with the values replaced by the gradient
    */
  def gradDensity : (params : Latent) => Map[String, Double] = params =>

    params.map((name, _) => 
      val grad = ScalaGrad.derive((s : alg.Scalar) => density(params.updated(name, s)))
      val pointToEvalute = params(name)
      (name, grad(pointToEvalute.toDouble))
    )          
    

  def sample(sampler : Sampler[A]) : Iterator[A] = 
    sampler.sample(this)
    
  def map[B](f : A => B) : RV[B] = RV(params => f(value(params)), density)

  def flatMap[B](f : A => RV[B]) : RV[B] = RV(
    params => f(value(params)).value(params),    
    params => f(value(params)).density(params) + density(params)
  )
  
  def condition(likelihood : A => alg.Scalar) : RV[A] = RV(
    params => value(params),
    params => density(params) + likelihood(value(params))
  )
  

object RV:
  def fromPrimitive(p : Dist, paramName : String) : RV[Double] =  
      RV( params =>params(paramName).toDouble, params => p.logPdf(params(paramName)))

