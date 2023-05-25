// package monadbayes.sampler

// import monadbayes.*

// def mhsampler[X](initialValue : X) :  DistOps ~> MHSampler = new (DistOps ~> MHSampler):
//   override def apply[A](fa: DistOps[A]): MHSampler[A] = 
//     println(fa)
//     MHSampler(() => Seq.empty)

  

// case class MHSampler[A](sample: () => Seq[A]) 
    

// given mh: Monad[MHSampler] with 
//     override def pure[A](a: A) = MHSampler(() => Seq(a))
//     override def flatMap[A, B](ma: MHSampler[A])(f: A => MHSampler[B]) =
//       MHSampler(() => (ma.sample().map(f)).flatMap(_.sample()))
