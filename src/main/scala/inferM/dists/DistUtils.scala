package inferM.dists

import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.{algebraT as alg}

object DistUtils:
  def scalarIsZero(a : alg.Scalar) : Boolean =     
    Math.abs(a.value) < 1e-10
    
