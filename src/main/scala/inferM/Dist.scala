package inferM

opaque type LogProb = Double
object LogProb:
    def apply(value : Double) : LogProb = value

extension (p : LogProb)
    def toDouble : Double = p
    def +(p2 : LogProb) : LogProb = p + p2
    def toProb : Double = math.exp(p)

trait Dist[A]:
    def run[X](interpreter: DistInterpreter[A, X]): X

    def map[B](f: A => B): Dist[B] = flatMap(a => Pure(f(a)))
    def flatMap[B](f: A => Dist[B]): Dist[B] = Bind(this, f)
    
    def condition(logLikelihood : A => LogProb) : Dist[A] = Conditional(logLikelihood, this)
        
trait DistInterpreter[A, X]:
    def pure(value : A) : X
    def primitive(dist : PrimitiveDist[A]) : X
    def conditional(logLikelihood : A => LogProb, dist : Dist[A]) : X
    def bind[B](dist : Dist[B], bind : B => Dist[A] ) : X


case class Pure[A](value : A) extends Dist[A]:
    def run[X](interpreter: DistInterpreter[A, X]): X = 
        interpreter.pure(value)

case class Primitive[A](dist : PrimitiveDist[A]) extends Dist[A]:
    def run[X](interpreter: DistInterpreter[A, X]): X = interpreter.primitive(dist)

case class Conditional[A](logLikelihood : A => LogProb, dist : Dist[A]) extends Dist[A]:
    def run[X](interpreter: DistInterpreter[A, X]): X = interpreter.conditional(logLikelihood, dist)

case class Bind[Y, A](dist : Dist[Y], bind : Y => Dist[A] ) extends Dist[A]:
    def run[X](interpreter: DistInterpreter[A, X]): X = interpreter.bind(dist, bind)