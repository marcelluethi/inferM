# InferM

InferM is an experimental library for Bayesian inference in Scala. It is inspired by [Rainier](https://github.com/stripe/rainier). In contrast to Rainier, it is (at least at this point in time) not meant for practical use, but rather as a playground to experiment with the idea of specifying probabilistic models using a monadic interface. Care has been taken to make the code as explicit as possible, so that it is easy to understand what is going on.

InferM allows to define probabilistic models by combining using univariate and multivariate random variables. Sampling is done using Hamiltonian Monte Carlo (HMC).

For automatic differentiation, it makes use of [Scala-grad](https://github.com/benikm91/scala-grad).

## Example

The following example shows how to define a linear regression model:
```scala 
case class Parameters(a : Double, b : Double, sigma : Double)

val prior = for  
  a <- RV.fromPrimitive(Gaussian(alg.lift(1.0), alg.lift(10)), "a")
  b <- RV.fromPrimitive(Gaussian(alg.lift(2.0), alg.lift(10)), "b")
  sigma <- RV.fromPrimitive(Exponential(alg.lift(1.0)), "sigma")
yield Parameters(a, b, sigma)


val likelihood =  (parameters : Parameters) => 
  data.foldLeft(alg.zeroScalar)((sum, point) =>
    val (x, y) = point 
    sum + Gaussian(
      alg.lift(parameters.a)  * alg.lift(x) +alg.lift(parameters.b), 
      alg.lift(parameters.sigma)
    ).logPdf(y)
  )

val posterior = prior.condition(likelihood)

// Sampling
val hmc = HMC[Parameters](
  initialValue = Map("a" -> 0.0, "b" -> 0.0, "sigma" -> 1.0),
  epsilon = 0.5,
  numLeapfrog = 20
)

val samples = posterior.sample(hmc).drop(10000).take(10000).toSeq
```

See the [examples](src/main/scala/inferM/examples) for more examples.


## Design

The core mechanism for specifying probabilistic models is the `RV` class. The `RV` class is a monad, which allows us to build up complex models from simple ones. An explanation of the `RV` class can be found [here](doc/Mechanism.md).

The core idea, of using Monads for specifying probabilistic programs has been explored by many other libraries. The most prominent example is [monad-bayes](https://github.com/tweag/monad-bayes). In Scala, the most prominent example is [Rainier](https://github.com/stripe/rainier), from which the core ideas have been borrowed. 



## State of the project

The project is in a very early stage. The API is not stable yet and performance is not a priority at this stage.
No packages have been published and, as some of its dependencies are also not published, it is not possible to use it as a library yet.

