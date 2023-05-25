package monadbayes

opaque type Prob = Double
object Prob:
    def apply(value : Double) : Prob = value

trait Dist[A]:
    def run[X](interpreter: DistInterpreter[A, X]): X

    def map[B](f: A => B): Dist[B] = flatMap(a => Pure(f(a)))
    def flatMap[B](f: A => Dist[B]): Dist[B] = Bind(this, f)
    
        
trait DistInterpreter[A, X]:
    def pure(value : A) : X
    def primitive(dist : PrimitiveDist[A]) : X
    def conditional(lik : A => Prob, dist : Dist[A]) : X
    def bind[B](dist : Dist[B], bind : B => Dist[A] ) : X


case class Pure[A](value : A) extends Dist[A]:
    def run[X](interpreter: DistInterpreter[A, X]): X = 
        interpreter.pure(value)

case class Primitive[A](dist : PrimitiveDist[A]) extends Dist[A]:
    def run[X](interpreter: DistInterpreter[A, X]): X = interpreter.primitive(dist)

case class Conditional[A](lik : A => Prob, dist : Dist[A]) extends Dist[A]:
    def run[X](interpreter: DistInterpreter[A, X]): X = interpreter.conditional(lik, dist)

case class Bind[Y, A](dist : Dist[Y], bind : Y => Dist[A] ) extends Dist[A]:
    def run[X](interpreter: DistInterpreter[A, X]): X = interpreter.bind(dist, bind)