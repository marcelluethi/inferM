package inferM.sampler

import inferM.*


import breeze.stats.{distributions => bdists}
import scalagrad.api.ScalaGrad
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.{algebraT as alg}
import scalagrad.auto.forward.breeze.BreezeDoubleForwardMode.given
import inferM.RV.{LatentSample, LatentSampleDouble}
import breeze.linalg.DenseVector
import scala.tools.nsc.doc.html.HtmlTags.P

/**
  * Implementation of Hamiltonian Monte Carlo
  *
  * @param initialValue
  * @param epsilon
  * @param numLeapfrog
  * @param rng
  */
class HMC[A](initialValue : LatentSampleDouble, epsilon : Double, numLeapfrog : Int)(using rng : breeze.stats.distributions.RandBasis) extends Sampler[A]:

  /**
    * Given values for the parameters (as a Map), the function returns 
    * a Map with the same keys, but with the values replaced by the gradient
    */
  def gradDensity(rv : RV[A]) : (latent : LatentSample) => LatentSampleDouble = latentSample =>

    latentSample.map{
      case (name, _ : alg.Scalar) => 
        val grad = ScalaGrad.derive((s : alg.Scalar) => rv.logDensity(latentSample.updated(name, s)))
        val pointToEvalute = latentSample(name)
        pointToEvalute match 
          case s : alg.Scalar => (name, grad(s.value))
          case _ => throw new Exception("Should not happen")
      case (name, _ : alg.ColumnVector) => 
        val grad = ScalaGrad.derive((s : alg.ColumnVector) => rv.logDensity(latentSample.updated(name, s)))
        val pointToEvalute = latentSample(name)
        pointToEvalute match 
          case v : alg.ColumnVector => (name, grad(v.value))
          case _ => throw new Exception("Should not happen")
    }          
    
  def liftArgsToDual(latentSample: LatentSampleDouble): LatentSample = 
    latentSample.map((name, value) => 
        value match 
          case s : Double => (name, alg.liftToScalar(s))
          case v : DenseVector[Double]  => (name, alg.createColumnVectorFromElements(v.toScalaVector.map(alg.liftToScalar)))
          case _ => throw new Exception("Should not happen")
      )

  def sample(rv : RV[A]) : Iterator[A] = 
    import alg.*
    def U = (latentSample : LatentSampleDouble) => 
      val x = rv.logDensity(liftArgsToDual(latentSample)) 
      x * alg.liftToScalar(-1.0)
    
    def gradU(current : LatentSampleDouble) : LatentSampleDouble = 
      val liftedArg = liftArgsToDual(current)
      gradDensity(rv)(liftedArg)
      .map((name, value) =>  // make it negative, as U is also negated (see above)
        value match 
          case s : Double => (name, s * -1.0)
          case v : DenseVector[Double] => (name, v * -1.0)
          case _ => throw new Exception("Should not happen")
      )

    def pStep(p : LatentSampleDouble, q : LatentSampleDouble, halfStep : Boolean) : LatentSampleDouble = 
      if halfStep then
        p.zip(gradU(q))
        .map{
          case ((name, p : Double), (_, dUq : Double)) =>
            (name, p - epsilon * dUq / 2.0)
          case ((name, p : DenseVector[Double]), (_, dUq : DenseVector[Double])) =>
           (name, p - epsilon * dUq / 2.0)
          case _ => throw new Exception("Should not happen") 
        }.toMap
      else
        p.zip(gradU(q))
        .map{
          case ((name, p : Double), (_, dUq : Double)) =>
            (name, p - epsilon * dUq )
          case ((name, p : DenseVector[Double]), (_, dUq : DenseVector[Double])) =>
           (name, p - epsilon * dUq )
          case _ =>  throw new Exception("Should not happen") 
        }.toMap        

    def qStep(p : LatentSampleDouble, q : LatentSampleDouble) : LatentSampleDouble =
      q.zip(p).map{
        case ((name, q : Double), (_, p : Double)) => (name, q + epsilon * p)
        case ((name, q : DenseVector[Double]), (_, p : DenseVector[Double])) => (name, q + epsilon * p)
        case _ => throw new Exception("Should not happen")
      }.toMap 

    def logProbP(p : LatentSampleDouble) : Double = 
      p.map{
        case (name, value: Double) => (name, value * value / 2.0)
        case (name, value: DenseVector[Double]) => (name, breeze.linalg.sum(value *:* value) / 2.0)
        case _ => throw new Exception("Should not happen")
      }.values.reduce(_ + _)

    def oneStep(currentQ : LatentSampleDouble) : LatentSampleDouble =       
      
      var q = currentQ
      var currentP = currentQ.map{
        case (name, _ : Double) => (name, bdists.Gaussian(0.0, 1.0).sample())
        case (name, v : DenseVector[Double]) => (name, DenseVector.rand(v.length, bdists.Gaussian(0.0, 1.0)))
      }
      
      // make half step
      var p = pStep(currentP, q, halfStep = true)
      
      for i <- 0 until numLeapfrog do
        q = qStep(p, q)
        if i <= numLeapfrog - 1 then           
           p = pStep(p, q, halfStep = false)

      p = pStep(p, q, halfStep = true)
      p = p.map((name, p) => p match 
        case p : Double => (name, -p)
        case p : DenseVector[Double] => (name, -p)
        case _ => throw new Exception("Should not happen")
      ).toMap

      val currentqProb = U(currentQ).toDouble
      val currentkProb = logProbP(currentP)
      val proposedQProb = U(q).toDouble
      val proposedkProb = logProbP(p)

      val a = currentqProb - proposedQProb + currentkProb - proposedkProb
      val r = rng.uniform.draw()

      if (r < Math.exp(a.toDouble)) then q else currentQ      

    Iterator
      .iterate(initialValue)(currentSample =>
        oneStep(currentSample)
    ).map(params => rv.value(liftArgsToDual(params)))



