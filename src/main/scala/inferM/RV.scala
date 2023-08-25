package inferM

import breeze.stats.{distributions => bdists}
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import scalagrad.api.ScalaGrad
import inferM.RV.LatentSample
import breeze.linalg.DenseVector
import scalagrad.api.matrixalgebra.MatrixAlgebra

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
class RV[A, S, CV](
    val value: LatentSample[S, CV] => A,
    val logDensity: LatentSample[S, CV] => S
)(using alg: MatrixAlgebra[S, CV, _, _]):

  /** Map the sampled values of the random variable with a function f (the
    * pushforward of the random variable)
    */
  def map[B](f: A => B): RV[B, S, CV] =
    RV(sample => f(value(sample)), logDensity)

  /** Map the sampled values of the random variable into a new random variable.
    * This produces a joint distribution of the original random variable and the
    * new one. (if this RV has density p(x) , the function f represents the
    * conditional distribution x-> p(y | x))
    */
  def flatMap[B](f: A => RV[B, S, CV]): RV[B, S, CV] = RV(
    latentSample => f(value(latentSample)).value(latentSample),
    latentSample =>
      f(value(latentSample)).logDensity(latentSample) + logDensity(latentSample)
  )

  /** Create a new random variable (the posterior), using the current one as a
    * prior and the given likelihood function as a conditional distribution. The
    * likelihood needs to provide the log density for each given sample S
    */
  def condition(likelihood: A => S): RV[A, S, CV] = RV(
    latentSample => value(latentSample),
    latentSample => logDensity(latentSample) + likelihood(value(latentSample))
  )

object RV:

  /** A LatentSample is a sample produced by a sampler within the RV class. It
    * is latent in the sense that after the sampler runs, it is transformed to
    * another sample type by applying a function to it.
    *
    * The type of the sample is a multivariate random variable. It is
    * represented as a map, which makes it possible to give each variable a name
    * and to refer to it by name, when we for example need to pass initial
    * values to a sampler.
    */
  type LatentSample[S, CV] = Map[String, S | CV]
