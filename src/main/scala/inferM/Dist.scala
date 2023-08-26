package inferM

import scalagrad.api.matrixalgebra.MatrixAlgebra

/** A distribution that can be sampled from and whose log-pdf can be computed
  */
trait Dist[A, S, CV](using alg: MatrixAlgebra[S, CV, _, _]):

  import alg.given

  def logPdf(value: S): S
  def draw(): A

  def toRV(name: String): RV[S, S, CV] =
    RV(
      s => s(name).asInstanceOf[S],
      s => logPdf(s(name).asInstanceOf[S])
    )

trait MvDist[A, S, CV]:
  def logPdf(value: CV): S
  def toRV(name: String): RV[CV, S, CV]
  def draw(): A
