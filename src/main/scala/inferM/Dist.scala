package inferM


import scalagrad.auto.forward.breeze.DeriverBreezeDoubleForwardPlan
import DeriverBreezeDoubleForwardPlan.{algebraT as alg}


/** 
 * A distribution that can be sampled from and whose log-pdf can be computed
*/
trait Dist:
  def draw(): alg.PrimaryScalar
  def logPdf(a: alg.Scalar): alg.Scalar

