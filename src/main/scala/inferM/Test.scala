// package inferM

// import scalagrad.api.matrixalgebra.MatrixAlgebra
// import breeze.linalg.DenseVector
// import breeze.linalg.DenseMatrix
// import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode
// import BreezeDoubleForwardMode.given
// import breeze.linalg.Transpose
// import scalagrad.api.ScalaGrad
// import scalagrad.api.matrixalgebra.MatrixAlgebraDSL
// import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode

// sealed trait Algs
// final class BreezeDualAlg  extends Algs
// final class BreezeDoubleAlg extends Algs

// type Scalar[X <: Algs] = X match 
//   case BreezeDualAlg => BreezeDoubleForwardMode.algebraT.Scalar
//   case BreezeDoubleAlg => Double

// type ColumnVec[X <: Algs] = X match 
//   case BreezeDualAlg => BreezeDoubleForwardMode.algebraT.ColumnVector
//   case BreezeDoubleAlg => DenseVector[Double]

// type RowVec[X <: Algs] = X match 
//   case BreezeDualAlg => BreezeDoubleForwardMode.algebraT.RowVector
//   case BreezeDoubleAlg => Transpose[DenseVector[Double]]

// type Matrix[X <: Algs] = X match 
//   case BreezeDualAlg => BreezeDoubleForwardMode.algebraT.Matrix
//   case BreezeDoubleAlg => DenseMatrix[Double]


// type MatrixAlg[X <: Algs] = X match 
//   case _ => MatrixAlgebra[Scalar[X], ColumnVec[X], RowVec[X], Matrix[X]]

// object MatrixAlg:
//   given MatrixAlg[BreezeDualAlg] = BreezeDoubleForwardMode.algebra

// def f[A <: Algs](using algebra : MatrixAlg[A])(s : Scalar[A], v : Scalar[A]) : Scalar[A] = 
//   import algebra.*
//   s + s



// def main(args : Array[String]) : Unit =
//   import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.{algebraT => alg}
//   import MatrixAlg.given
//   val x = f(alg.lift(3.0), alg.lift(4.0))
//   val y = ScalaGrad.derive(f)