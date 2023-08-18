package inferM


import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode
import BreezeDoubleForwardMode.{algebraT as alg}


/** 
 * A distribution that can be sampled from and whose log-pdf can be computed
*/
trait UvDist[A]:
  def draw(): A
  def logPdf(a: alg.Scalar): alg.Scalar

trait MvDist[A]:
  def draw(): A
  def logPdf(a : alg.ColumnVector): alg.Scalar


