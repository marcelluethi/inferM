package inferM

import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import breeze.linalg.DenseVector
import scalagrad.api.matrixalgebra.MatrixAlgebra
import scalagrad.api.matrixalgebra.MatrixAlgebraDSL
import math.Fractional.Implicits.infixFractionalOps

// Define the type class
trait TupleSize[T <: Tuple]:
  def value: Int


object TupleSize:
  given TupleSize[EmptyTuple.type] with {
    def value: Int = 0
  }
  given [H, T <: Tuple](using ts: TupleSize[T]) : TupleSize[H *: T] with {
    def value: Int = 1 + ts.value
  }
  given [T1 <: Tuple, T2 <: Tuple](using t1: TupleSize[T1], t2: TupleSize[T2]) : TupleSize[Tuple.Concat[T1, T2]] with {
    def value: Int = t1.value + t2.value
  }

/** A random variable, described by the logDensity function, from which samples
  * of type S can be drawn.
  *
  * @tparam A
  *   The type of samples produced by sampling from this random variable
  * @tparam S
  *   The Scalar type used (for autodiff)
  * @tparam CV
  *   The ColumnVector type used (for autodiff)
  */
class NewRV[A, S: Fractional, CV, LS <: Tuple : TupleSize](
  val value: LS => A,
  val logDensity: LS => S
):

  val self = this

  /** Map the sampled values of the random variable with a function f (the
    * pushforward of the random variable)
    */
  def map[B](f: A => B): NewRV[B, S, CV, LS] =
    NewRV(sample => f(value(sample)), logDensity)

  /** Map the sampled values of the random variable into a new random variable.
    * This produces a joint distribution of the original random variable and the
    * new one. (if this NewRV has density p(x) , the function f represents the
    * conditional distribution x-> p(y | x))
    */
  def flatMap[B, LS2 <: Tuple : TupleSize](f: A => NewRV[B, S, CV, LS2]): NewRV[B, S, CV, Tuple.Concat[LS, LS2]] = NewRV[B, S, CV, Tuple.Concat[LS, LS2]](
    latentSample => 
      val lsSize = summon[TupleSize[LS]].value
      val (ls: LS, ls2: LS2) = latentSample.splitAt(lsSize).asInstanceOf[(LS, LS2)]
      f(value(ls)).value(ls2),
    latentSample =>
      val lsSize = summon[TupleSize[LS]].value
      val (ls: LS, ls2: LS2) = latentSample.splitAt(lsSize).asInstanceOf[(LS, LS2)]
      f(value(ls)).logDensity(ls2) + logDensity(ls)
  )

  /** Create a new random variable (the posterior), using the current one as a
    * prior and the given likelihood function as a conditional distribution. The
    * likelihood needs to provide the log density for each given sample S
    */
  def condition(likelihood: A => S): NewRV[A, S, CV, LS] = NewRV(
    latentSample => value(latentSample),
    latentSample => logDensity(latentSample) + likelihood(value(latentSample))
  )

object NewRV:

  type TupleOf[S, CV] = [T <: Tuple] =>> TupleOfAux[S, CV, T]

  type TupleOfAux[S, CV, T <: Tuple] = T match 
    case EmptyTuple => EmptyTuple
    case S *: t => S *: TupleOfAux[S, CV, t]
    case CV *: t => CV *: TupleOfAux[S, CV, t]
