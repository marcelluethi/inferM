package inferM.examples


import scaltair.*
import scaltair.PlotTargetBrowser.given

import inferM.*
import inferM.sampler.* 

object LinearRegression extends App:

    case class Params(a : Double, b : Double, sigma : Double)
    val prior = for
        a <- Normal(0, 3)
        b <- Normal(0, 1)
        sigma <- Gamma(1, 1)
    yield Params(a, b, sigma)

    // training data
    val data = Seq((1.0, 1.1), (2.0, 1.9), (3.0, 3.0))

    def addDataPoint(dist : Dist[Params], x : Double, y : Double): Dist[Params] =
        dist.condition(params => Normal(params.a * x + params.b, params.sigma).logPdf(y))

    val model = data.foldLeft[Dist[Params]](prior)((dist, point) => addDataPoint(dist, point._1, point._2))

    val samples = model.run(MetropolisHastings(initialSample = Params(0, 0, 1))).drop(1000).take(10000).toSeq
    
    // plot a histogram of the marginal distribution of a
    val columnData = Map("a" -> samples.map(_.a))
    Chart(columnData).encode(
      Channel.X("a", FieldType.Quantitative).binned(),
      Channel.Y("a", FieldType.Quantitative).count()
    ).markBar()    
    .show()

    
