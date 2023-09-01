package inferM

import breeze.numerics.log
import inferM.RV.LatentSample

case class Sample[A](value : A, logDensity : Double)

/** Base trait for implementing sampling algorithms.
  */
trait Sampler[A]:
  def sample(rv: RV[A]): Iterator[Sample[A]]
