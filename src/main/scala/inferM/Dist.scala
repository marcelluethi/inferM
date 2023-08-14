package inferM


import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode
import BreezeDoubleForwardMode.{algebraT as alg}


/** 
 * A distribution that can be sampled from and whose log-pdf can be computed
*/
trait Dist[A]:
  def draw(): A
  def logPdf(a: A): alg.Scalar

