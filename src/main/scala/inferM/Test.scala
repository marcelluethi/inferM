package inferM



trait Alg:
  type Scalar
  def zero : Scalar

object Alg1 extends Alg:
  type Scalar = Double
  def zero : Scalar =  1.0




class RV1[A](val value : Alg ?=> Alg#Scalar => A, val density : Alg ?=> Alg#Scalar => Alg#Scalar):


  def map[B](f : A => B) : RV1[B] = RV1(params => f(value(params)), density)

  def flatMap[B](f : A => RV1[B]) : RV1[B] = RV1(
    params => f(value(params)).value(params),    
    params => f(value(params)).density(params) //+ density(params)
  )
  
  def condition(likelihood : Alg ?=> Alg => A => Alg#Scalar) : RV1[A] = RV1(
    params => value(params),
    params => density(params) 
  )
  

object RV1:
  def fromPrimitive()(using alg : Alg) : RV1[Double] =  
      RV1[Double]( value => 1.0, params => alg.zero)

object Test:

  given algGiven : Alg = Alg1
  val rv = RV1.fromPrimitive()


