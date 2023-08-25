package inferM

import scalagrad.api.matrixalgebra.MatrixAlgebra
import scalagrad.api.ScalaGrad


import breeze.linalg.DenseVector
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode
import BreezeDoubleForwardMode.given
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.{algebraT => alg}

/** A distribution that can be sampled from and whose log-pdf can be computed
  */
trait Dist:

  self => 

    def logPdf(s: alg.Scalar): alg.Scalar
    def value(s : alg.Scalar) : alg.Scalar
    def toRV(name: String): RV[alg.Scalar] = 
      RV[alg.Scalar](
        s => self.value(s(name).asInstanceOf[alg.Scalar]),
        s => self.logPdf(s(name).asInstanceOf[alg.Scalar])
      )
    

  //def transform(bij : Bijection[forwardAlg.Scalar, forwardAlg.ColumnVector]) : Dist[A]= ???
    // new Dist[A, S, CV]:
    //   def alg : MatrixAlgebra[S, CV, _, _] = forwardAlg
    //   def logPdf(value: S): S = 

    //     val f = ScalaGrad.derive[forwardAlg.Scalar, forwardAlg.Scalar](bij.inverse)
    //     self.logPdf(bij.inverse(value)) * (value)
    //   def value(v : S) : A = self.value(bij.apply(v))



trait MvDist:
  self =>

    def logPdf(value: alg.ColumnVector): alg.Scalar
    def value(v : alg.ColumnVector) : alg.ColumnVector
    def toRV(name: String): RV[alg.ColumnVector] = 
      RV[alg.ColumnVector](
          s => self.value(s(name).asInstanceOf[alg.ColumnVector]),
          s => self.logPdf(s(name).asInstanceOf[alg.ColumnVector])
        )

   // def transform(bij : Bijection[S, CV]) : Dist[A]= ???
