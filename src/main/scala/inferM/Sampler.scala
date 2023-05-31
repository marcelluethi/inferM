package inferM

class Sampler[A]() extends DistInterpreter[A, A]:

    def bind[B](dist : Dist[B],  bind : B => Dist[A]) : A =
        val x = dist.run(Sampler[B]())        
        bind(x).run[A](Sampler[A]())


    /// <summary>
    /// All conditionals must be removed before sampling.
    /// </summary>
    def conditional(logLikelihood : A => LogProb, dist : Dist[A]) : A =
        throw new Exception("All conditionals must be revmoved before sampling.")
        

    def primitive(dist : PrimitiveDist[A]) : A = 
        dist.sample();        

    def pure(value : A) : A = value

extension [A](dist : Dist[A])
    def sample() : A = dist.run(Sampler[A]())

    def sampleN(n : Int) : Seq[A] = 
        (1 to n).map(_ => dist.sample())