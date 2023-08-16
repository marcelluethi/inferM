# How does inferM work?

The core mechanism used to specify probabilistic is the [RV class](../src/main/scala/inferM/RV.scala). `RV` stands for *Random Variable*. Notice that the implementation itself is extermely concise. The implementation is merely a few lines of code. However, 
it can be difficult to understand without further explanation. 
Let's unpack it step by step.


A random variable is a function, which maps from a space of samples (the sample space) to a set of possible outcomes.
How likely each outcome is, is determined by a density function, which assigns to each outcome a number that is proportional to its probability, so it is 
a function $Sample \rightarrow \mathbb{R}$.

A simple Scala implementation of a random variable could look as follows:

```scala
type Sample = Double
type LogProb = Double
class RV(logDensity : Sample => LogProb):
  def sample(sampler : Sampler[Sample]) : Sample = sampler.sampleFrom(logDensity) 
```

There are several things to note here. First, the density function is a function from the sample to a log-probability. This is often preferable, as the probabilities can be very small numbers, which can lead to numerical problems.
Second, we have defined the type `Sample` to be a `Double`. 
Third, we have delegated the main work of drawing a sample from the random variable to a special sampler class. There are different algorithms that can be used to generate samples, given a density function that can be
evaluated point wise. Examples are the Metropolis-Hastings and Hamiltonian Monte Carlo. Which sample method we use is irrelevant to the random variable itself.

The biggest restriction in our definition is that the sample space is fixed to be the type `Double`. Many random variables, however, are not of this type.
Instead of changing the Sample type and consequently the `sampler`, we can, leave the `sampler` untouched and instead introduce a value function, which maps the 
`Sample` value to a value of type `S`. This function is called the `value` function. The resulting random variable does then not produce a sample of type `Sample`, but of some other type `S`. To make the distinction clear, 
we call this internal sample the `LatentSample` as it is not observed from the outside.  The new definition looks as follows:

```scala
type LatentSample = Double
type LogProb = Double
class RV[S](value : LatentSample => S, logDensity : LatentSample => LogProb):
  def sample(sampler :Sampler[LatentSample]) : A = value(sampler.sampleFrom(logDensity))
```

Note that we simply pass the sample from the density through the value function. 
If we want to preserve the original density, we can define the value function as the identity function. 
```scala 
  RV[Double](identity, x => if x >= 0 and x <= 1 then 0.0 else Double.NegativeInfinity
```
But we can also drastically change the behavior of the random variable, without having to change the underlying sampler, by simply providing a different value function. 
For example, we can define a coin flip as follows:
  
  ```scala
  val coinFlip = RV[Boolean](x => x < 0.5, x => if x >= 0 and x <= 1.0 then 0.0 else Double.NegativeInfinity)
  ```
Note that the density is still a continuous - it is just a uniform distribution on the interval [0, 1]. In particular, the sampler can still make use of the fact that the density is continuous, even though the values produced by the 
random variable are discrete.

This type of transforming one random variable into another by applying a function is useful. We can make it even more convenient by defining the 
`map` method on the `RV` class.

```scala
class RV[S](value : LatentSample => S, logDensity : LatentSample => LogProb):
  def sample(sampler :Sampler[Sample]) : S = value(sampler.sampleFrom(logDensity))
  def map[B](f : S => T) : RV[T] = RV(event => f(value(event)), logDensity)  
```

## From prior to posterior

The random variable is unconstrained. In a Bayesian setting, we would say that his random variable corresponds to a prior distribution. Let's denote it by $p(S)$.
To go from a prior distribution to a posterior distribution, we need to condition on some data $D$. The dependency on the data is give by the likelihood function $p(D | S=s)$, which specifies how 
likely we are to observe the data $D$ given a particular value of the random variable $S=s$. In our implementation, it becomes a function of type `S -> LogProb`.
The density of the conditioned random variable (the posterior) $p(S | D = d)$ is then given by the product of the prior and the likelihood, divided by the probability of the data $p(D = d)$, which is a constant.
This means that in our implementation, where we work in log space, we can simply add the log-likelihood to the log-density of the prior. 

Our new random variable class looks as follows:

```scala
class RV[S](value : LatentSample => S, logDensity : LatentSample => LogProb):
  def sample(sampler :Sampler[LatentSample]) : S = value(sampler.sampleFrom(logDensity))
  def map[T](f : S => T) : RV[T] = RV(sample => f(value(sample)), logDensity)  
  def condition(likelihood : S => LogProb) : RV[S] = RV(value, sample => logDensity(sample) + likelihood(value(sample)))
```

As an example, let's consider the case where we have a coin, which we flip $n$ times and observe $k$ heads. We can then define the likelihood function as follows:

```scala
def likelihood(k : Int, n : Int) : Boolean => LogProb = 
  x => if x then k * math.log(0.5) else (n - k) * math.log(0.5)
```

We can then define the posterior distribution as follows:

```scala
val posterior = coinFlip.condition(likelihood(k, n))
```

## Joint distributions

So far, samples of the random variable are constrained to be univariate. However, we often want to work with multivariate random variables.
To allow for this possibility, we need to redefine the type of the sample. It would be most straight-forward to use a vector of Doubles. 
We will use, however, the type `Map[String, Double]` to represent a random vector. This is because we want to be able to give names to the different dimensions of the vector.
The density function remains a function from the sample to a log-probability, but it is now a joint density function, as the sample has multiple dimensions.

The construction we use to build up joint random variables is the following:
We start with a random variable that produces a single value, lets call it $p(S)$. We then define a function $f : S \rightarrow p(T | S)$, which maps the value of the random variable to a new random variable.
The joint probability is then given by $p(S, T) = p(S) p(T | S)$. 

In Scala, we can encode this using the `flatMap`` method on the `RV`` class. 

```scala
type Sample = Map[String, Double]
class RV[S](value : Sample => S, logDensity : Sample => LogProb):
  def sample(sampler :Sampler[Sample]) : S = value(sampler.sampleFrom(logDensity))
  def map[T](f : S => T) : RV[T] = RV(sample => f(value(sample)), logDensity)  
  def flatMap[T](f : S => RV[T]) : RV[T] = 
    RV(
      sample => f(value(sample)).value(sample), 
      sample => logDensity(sample) + f(value(sample)).logDensity(sample)
    )  
  def condition(likelihood : S => LogProb) : RV[T] = RV(value, Sample => logDensity(Sample) + likelihood(value(Sample)))

```

To see this in action, lets come back to the coin flip example. We can define a random variable that flips a coin $n$ times and returns the number of heads as follows:

```scala 
coinFlip.flatMap(coin1 => coinFlip.map(coin2 => (coin1, coin2)))
```

or, using the `for`-comprehension syntax:

```scala
val twoFlipsRV = 
  for 
    coin1 <- coinFlip
    coin2 <- coinFlip
  yield (coin1, coin2)
```

## More about sampling

The random variable class, described about is used to define a model (the joint density). It is  a description, which individual random variables a model is comprised of and and how these random variables depend on each other. The random variable itself has no code to produce any samples. In fact, this is the beauty of the Bayesian approach. Modelling can be completely separated from inference. 
However, at some point, we want to produce samples from the model. To do this, we need to use a sampler. The standard choice in inferM is the Hamiltonian Monte Carlo (HMC) sampler. The HMC sampler assumes that the density function is continuous and differentiable. In particular, it needs access to the derivative of the density function. This is why in the real definition of the `RV` class, the density function is not of type `Double` but is a dual type, which in addition to the value also has information about the derivative at each point. However, as previously mentioned, any sampler that can produce samples from a density function can be used. To implement another sampler, simply implement the [Sampler](../src/main/scala/inferM/Sampler.scala) trait.

## Putting it all together

At this point, we have discussed all the main components of the inferM library. The best way to get a feel for how to use the `RV` class to define models is to look at the examples in the [examples](../src/main/scala/examples) folder. To really understand how the RV class works, it might be helpful to define a few simple models yourself and to think about how map and flatMap are used to build up the joint density.


