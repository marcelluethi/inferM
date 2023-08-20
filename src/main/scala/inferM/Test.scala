package inferM

import scalagrad.api.matrixalgebra.MatrixAlgebra

@main
def forwardSample2 =
  // import ScalaGrad and the forward plan
  import scalagrad.api.ScalaGrad
  import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode
  import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.given
  import BreezeDoubleForwardMode.{algebraT as alg}

  class Foo[S]:
    def f(x1: S, x2: S)(using MatrixAlgebra[S, _, _, _]): S =
      x2 + x1

  def g[S](using MatrixAlgebra[S, _, _, _])(x1: S, x2: S): S =
    x2 + x1

  // derive the function
  val foo = Foo[alg.Scalar]
  val df = ScalaGrad.derive(foo.f) // HERE!

  val df2 = ScalaGrad.derive(g[alg.Scalar]) // HERE!
  println(df(3.0, 3.0))

  // derive the function
  // val df = ScalaGrad.derive(f[alg.Scalar])
  // val dg = ScalaGrad.derive(g)
