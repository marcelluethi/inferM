package inferM

/** Base trait for implementing sampling algorithms.
  */
trait Sampler[A, S, CV]:
  def sample(rv: RV[A, S, CV]): Iterator[A]
