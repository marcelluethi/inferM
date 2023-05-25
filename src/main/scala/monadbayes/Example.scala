package monadbayes


import scaltair.*
import scaltair.PlotTargetBrowser.given

import sampler.PriorSampler

object Example extends App:

    val prior = for
        x <- Primitive(Normal(0, 3))
        y <- Primitive(Normal(x, 1))
    yield (x, y)


    println("samples: " +prior.sampleN(5))
    val samples = prior.sampleN(100)


    println(samples.length)
    val columnData = Map("x" -> samples.map(_._1), "y" -> samples.map(_._2))
    Chart(columnData).encode(
      Channel.X("x", FieldType.Quantitative),
      Channel.Y("y", FieldType.Quantitative)
    ).markCircle()    
    .show()

    
