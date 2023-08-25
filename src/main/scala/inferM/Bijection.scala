package inferM

import scalagrad.api.matrixalgebra.MatrixAlgebra

trait Bijection[S, CV](using alg : MatrixAlgebra[S, CV, _, _]):
  def apply(s: S): S
  def inverse(a: S): S
