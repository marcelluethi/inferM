package inferM.dists

import inferM.* 


import scalagrad.api.matrixalgebra.MatrixAlgebraT
import scalagrad.api.spire.trig.DualScalarIsTrig.given
import spire.implicits.DoubleAlgebra 
import spire.algebra.Trig
import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import scalagrad.auto.forward.breeze.DeriverBreezeDoubleForwardPlan.{algebraT as alg}
import scalagrad.auto.forward.breeze.DeriverBreezeDoubleForwardPlan.given

class Bernoulli(p : alg.Scalar) extends Dist[Boolean]:
  
  def logPdf(x : Boolean): alg.Scalar = 

    val trig = summon[Trig[alg.Scalar]]
    if x == false then
      trig.log(alg.liftToScalar(1.0) - p)
    else
      trig.log(p)

  def draw() : Boolean = 
    val dist = bdists.Bernoulli(p.value)
    dist.draw()

