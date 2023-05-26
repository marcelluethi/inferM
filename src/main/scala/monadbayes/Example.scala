package monadbayes


import scaltair.*
import scaltair.PlotTargetBrowser.given

import sampler.PriorSampler
import monadbayes.sampler.PriorSampler
import monadbayes.sampler.PriorWeightedSampler
import monadbayes.sampler.MetropolisHastings

object Example extends App:

    val prior = for
        x <- Primitive(Normal(0, 3))
        y <- Primitive(Normal(x, 1))
    yield (x, y)

    val posterior = prior.condition((x, y) => if (x > 0) then Prob(1.0) else Prob(1e-10))    
    val samples = posterior.run(MetropolisHastings(100, initialSample = (0.1, 0.0)))

    println(samples.length)
    val columnData = Map("x" -> samples.map(_._1), "y" -> samples.map(_._2))
    Chart(columnData).encode(
      Channel.X("x", FieldType.Quantitative),
      Channel.Y("y", FieldType.Quantitative)
    ).markCircle()    
    .show()

    
