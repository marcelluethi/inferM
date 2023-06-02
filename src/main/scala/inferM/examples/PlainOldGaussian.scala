package inferM.examples
import inferM.sampler.SMC
import inferM.sampler.PriorWeightedSampler
import inferM.*
import scaltair.* 
import scaltair.PlotTargetBrowser.given
import inferM.sampler.PIMH

object PlainOldGaussian extends App:

  val prior = for
    x <- Normal(3.0, 1.0)
    y <- Normal(x, 0.1)
  yield (x, y)

  val model = prior.condition((x, y) => if x > 1 then LogProb(0) else LogProb.MinValue)

//   val samples = model.run(SMC(1000)).sample()
//   println(samples.samples.length)
  val samples = model.run(PIMH(1000)).take(100).toSeq.last.samples

  val plotData = Map("x" -> samples.map(_.value._1), "y"-> samples.map(_.value._2))
  Chart(plotData)
    .encode(
      Channel.X("x", FieldType.Quantitative),
      Channel.Y("y", FieldType.Quantitative)
    )
    .markPoint()
    .show()

