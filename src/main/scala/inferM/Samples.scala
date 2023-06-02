package inferM

  /**
  * Representation a sample with associated probability. 
  *
  * @param value
  * @param logWeight
  */
case class Sample[A](value: A, logWeight: LogProb):
  
  def map[B](f: A => B): Sample[B] = 
    Sample(f(value), logWeight)

  def flatMap[B](f : A => Sample[B]): Sample[B] = 
    val ps = f(value)
    Sample(ps.value, ps.logWeight + logWeight)


case class Samples[A](samples : Vector[Sample[A]]):

    def map[B](f: A => B): Samples[B] = 
      Samples(samples.map(_.map(f)))
    
    def flatMap[B](f : A => Samples[B]): Samples[B] =
        Samples(samples.flatMap(s => f(s.value).samples))
