package inferM

trait Sampler[A]:
  def sample(rv : RV[A]) : Iterator[A]
