package inferM

/** A distribution that can be sampled from and whose log-pdf can be computed
  */
trait Dist[A, S, CV]:
  def logPdf(value: S): S
  def toRV(name: String): RV[A, S, CV]
  def draw(): A

trait MvDist[A, S, CV]:
  def logPdf(value: CV): S
  def toRV(name: String): RV[A, S, CV]
  def draw(): A
