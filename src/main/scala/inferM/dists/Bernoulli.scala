package inferM.dists

import inferM.* 


import scalagrad.api.spire.trig.DualScalarIsTrig.given
import spire.implicits.DoubleAlgebra 
import spire.algebra.Trig
import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.{algebraT as alg}
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.given

class Bernoulli(p : alg.Scalar) extends UvDist[Boolean]:
  
  def logPdf(x : alg.Scalar): alg.Scalar = 

    val trig = summon[Trig[alg.Scalar]]
    if DistUtils.scalarIsZero(x) == false then
      trig.log(alg.liftToScalar(1.0) - p)
    else
      trig.log(p)

  def draw() : Boolean = 
    val dist = bdists.Bernoulli(p.value)
    dist.draw()

