// package monadbayes


  
// trait Monad[M[_]]:
//     def pure[A](a: A): M[A]
//     def flatMap[A, B](ma: M[A])(f: A => M[B]): M[B]


// object Monad:
//     def apply[M[_]](using monad: Monad[M]): Monad[M] = monad


// trait ~>[F[_], G[_]]:
//     def apply[A](fa: F[A]): G[A]


// trait Free[M[_], A]:
//     import Free.*

//     def flatMap[B](f: A => Free[M, B]): Free[M, B] = FlatMap(this, f)
//     def map[B](f: A => B): Free[M, B] = flatMap(a => pure(f(a)))
//     def foldMap[G[_]: Monad](natTrans: M ~> G): G[A] = this match 
//         case Pure(a) => Monad[G].pure(a)
//         case Suspend(ma) => natTrans.apply(ma)
//         case FlatMap(fa, f) => // need a G[B]
//             Monad[G].flatMap(fa.foldMap(natTrans))(a =>     f(a)      .foldMap(natTrans) )



// object Free:
//     def pure[M[_], A](a: A): Free[M, A] = Pure(a)
//     def liftM[M[_], A](ma: M[A]): Free[M, A] = Suspend(ma)

//     case class Pure[M[_], A](a: A) extends Free[M, A]
//     case class FlatMap[M[_],A,B](fa: Free[M, A], f: A => Free[M, B]) extends Free[M, B]
//     case class Suspend[M[_], A](ma: M[A]) extends Free[M, A]

