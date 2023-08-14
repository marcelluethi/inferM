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
    import alg.* 
    alg.liftToScalar(1.0) / (to - from)
  

  def draw() : Double = 
    val dist = bdists.Uniform(from.value, to.value)
    dist.draw()


