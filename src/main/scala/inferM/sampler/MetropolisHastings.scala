package inferM.sampler

import inferM.*

class MetropolisHastings[A](initialSample: A)
    extends DistInterpreter[A, Iterator[A]]:

  override def pure(value: A): Iterator[A] = mh(Pure(value))

  override def primitive(dist: PrimitiveDist[A]): Iterator[A] = mh(
    Primitive(dist)
  )

  override def conditional(
      logLikelihood: A => LogProb,
      dist: Dist[A]
  ): Iterator[A] =
    mh(Conditional(logLikelihood, dist))

  override def bind[B](dist: Dist[B], bind: B => Dist[A]): Iterator[A] = mh(
    Bind(dist, bind)
  )

  def mh(dist: Dist[A]): Iterator[A] =
    val proposals = dist.run(PriorWeightedSampler())
    val proposalIt = Iterator.continually(proposals.sample())

    val rng = scala.util.Random()

    def oneStep(
        currentSample: (A, LogProb),
        newProposal: (A, LogProb)
    ): (A, LogProb) =

      val (currentValue, currentP) = currentSample
      val (newValue, newP) = newProposal
      val a = newP.toDouble - currentP.toDouble
      val r = rng.nextDouble()

      if (a > 0.0 || r < a.toProb) then newProposal
      else currentSample

    Iterator
      .iterate((initialSample, LogProb(0.0)))(currentSample =>
        oneStep(currentSample, proposalIt.next())
      )
      .map(_._1)
