package inferM

import scalagrad.api.matrixalgebra.MatrixAlgebra
import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import scalagrad.auto.breeze.BreezeDoubleMatrixAlgebra
import breeze.linalg.Transpose


// def main() : Unit = 
//   for 
//     a <- Gaussian(alg.lift(1.0), alg.lift(2.0))
//     b <- Gaussian(a, alg.lift(1.0))
//   yield (a, b)

def f(x : Double, y : Double)(using alg : MatrixAlgebra[Double, DenseVector[Double], Transpose[DenseVector[Double]], DenseMatrix[Double]]) : Double = 
  alg.plusSS(x, y)

// def forwardSample2 =
//   // import ScalaGrad and the forward plan
//   import scalagrad.api.ScalaGrad
//   import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode
//   import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.given
//   import BreezeDoubleForwardMode.{algebraT as alg}

//   class Foo[S]:
//     def f(x1: S, x2: S)(using MatrixAlgebra[S, _, _, _]): S =
//       x2 + x1

//   def g[S](using MatrixAlgebra[S, _, _, _])(x1: S, x2: S): S =
//     x2 + x1

//   def h(alg : BreezeDoubleMatrixAlgebra.type)(a : Double, b : Double) : Double = 
//     alg.plusSS(a, b)

//   val dh = ScalaGrad.derive[Double => Double](h(BreezeDoubleMatrixAlgebra))

//   // derive the function
//   val foo = Foo[alg.Scalar]
//   val df = ScalaGrad.derive(foo.f) // HERE!

//   val df2 = ScalaGrad.derive(g[alg.Scalar]) // HERE!
//   println(df(3.0, 3.0))

  // derive the function
  // val df = ScalaGrad.derive(f[alg.Scalar])
  // val dg = ScalaGrad.derive(g)
