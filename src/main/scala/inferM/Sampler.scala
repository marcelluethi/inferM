package inferM

/**
  * Base trait for implementing sampling algorithms. 
  */
trait Sampler[A]:
  def sample(rv : RV[A]) : Iterator[A]
