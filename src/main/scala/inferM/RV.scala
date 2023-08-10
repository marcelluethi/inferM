package inferM

import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import scalagrad.api.ScalaGrad
import scalagrad.auto.forward.breeze.DeriverBreezeDoubleForwardPlan.{algebraT as alg}
import scalagrad.auto.forward.breeze.DeriverBreezeDoubleForwardPlan.given

type Latent = Map[String, alg.Scalar]

class RV[A](val value : Latent => A, val density : Latent => alg.Scalar):

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

