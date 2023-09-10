package inferM

import scalagrad.auto.forward.BreezeDoubleForwardDualMode.derive as d
import scalagrad.auto.forward.BreezeDoubleForwardDualMode.algebra.*  // import syntax
import scalagrad.auto.forward.BreezeDoubleForwardDualMode.algebraDSL as alg


import breeze.linalg.DenseVector
import scalagrad.auto.forward.BreezeDoubleForwardDualMode.derive as d
import scalagrad.auto.forward.BreezeDoubleForwardDualMode.algebra.*  // import syntax
import scalagrad.auto.forward.BreezeDoubleForwardDualMode.algebraDSL as alg
import breeze.linalg.DenseMatrix

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
    

  def transform(bij : Bijection[alg.Scalar, alg.Scalar]) : Dist= 
    new Dist:      
      def logPdf(value: alg.Scalar): alg.Scalar = 
        val dInv = d(bij.inverse)
        self.logPdf(bij.inverse(value)) + alg.trig.log(alg.lift(dInv(value.value)))
      def value(v : alg.Scalar) : alg.Scalar = self.value(v)



trait MvDist:
  self =>

    def logPdf(value: alg.ColumnVector): alg.Scalar
    def value(v : alg.ColumnVector) : alg.ColumnVector
    def toRV(name: String): RV[alg.ColumnVector] = 
      RV[alg.ColumnVector](
          s => self.value(s(name).asInstanceOf[alg.ColumnVector]),
          s => self.logPdf(s(name).asInstanceOf[alg.ColumnVector])
        )


    def transform(bij : Bijection[alg.ColumnVector, alg.ColumnVector]) : MvDist = 
      new MvDist:      
        def logPdf(value: alg.ColumnVector): alg.Scalar = 
          val dInv : DenseVector[Double] => DenseMatrix[Double] = d(bij.inverse)
          self.logPdf(bij.inverse(value)) + alg.trig.log(alg.lift(breeze.linalg.det(dInv(value.dv))))
        def value(v : alg.ColumnVector) : alg.ColumnVector = self.value(v)
