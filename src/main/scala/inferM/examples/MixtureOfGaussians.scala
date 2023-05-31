package inferM.examples

import inferM.*
import inferM.sampler.*
import scaltair.*
import scaltair.PlotTargetBrowser.given

object MixtureOfGaussians extends App:

  val prior = for
    x <- Bernoulli(0.5)
    y <- Normal(if x then -3 else 3, 1)
  yield y

  val model = prior.condition(x => Normal(x, 2.0).logPdf(1.0))

  val samples = model
    .run(MetropolisHastings(initialSample = 0))
    .drop(1000)
    .take(10000)
    .toSeq
  val plotData = Map("x" -> samples)
  Chart(plotData)
    .encode(
      Channel.X("x", FieldType.Quantitative).binned(maxBins = 40),
      Channel.Y("x", FieldType.Quantitative).count()
    )
    .markBar()
    .show()
