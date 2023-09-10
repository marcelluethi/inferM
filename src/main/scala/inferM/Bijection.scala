package inferM



trait Bijection[A, B] extends Function1[A, B]:
  override def apply(s : A) : B = forward(s)
  def forward(s: A): B
  def inverse(a: B): A



