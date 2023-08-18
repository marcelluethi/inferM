package inferM.dists

import inferM.* 


import scalagrad.api.matrixalgebra.MatrixAlgebraDSL
import scalagrad.api.spire.trig.DualScalarIsTrig.given
import spire.implicits.DoubleAlgebra 
import spire.algebra.Trig
import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.{algebraT as alg}
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.given

class Uniform(from : alg.Scalar, to : alg.Scalar) extends Dist[Double]:
  
  def logPdf(x : Double): alg.Scalar = 
    val trig = summon[Trig[alg.Scalar]]

    import alg.* 
    if x >= from.value && x <= to.value then trig.log(alg.lift(1.0) / (to - from)) else alg.lift(Double.MinValue)
  

  def draw() : Double = 
    val dist = bdists.Uniform(from.value, to.value)
    dist.draw()

  def toRV(name : String) : RV[Double] = 
    RV[Double](s => s(name).asInstanceOf[alg.Scalar].value, s => logPdf(s(name).asInstanceOf[alg.Scalar].value))