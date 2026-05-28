## dlasv2

*     .. Scalar Arguments ..
      DOUBLE PRECISION   CSL, CSR, F, G, H, SNL, SNR, SSMAX, SSMIN
*     ..
*
* =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO
      parameter( zero = 0.0d0 )
      DOUBLE PRECISION   HALF
      parameter( half = 0.5d0 )
      DOUBLE PRECISION   ONE
      parameter( one = 1.0d0 )
      DOUBLE PRECISION   TWO
      parameter( two = 2.0d0 )
      DOUBLE PRECISION   FOUR
      parameter( four = 4.0d0 )
*     ..
*     .. Local Scalars ..
      LOGICAL            GASMAL, SWAP
      INTEGER            PMAX
      DOUBLE PRECISION   A, CLT, CRT, D, FA, FT, GA, GT, HA, HT, L, M,
     $                   MM, R, S, SLT, SRT, T, TEMP, TSIGN, TT
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          abs, sign, sqrt
*     ..
*     .. External Functions ..
      DOUBLE PRECISION   DLAMCH
      EXTERNAL           dlamch
*     ..
*     .. Executable Statements ..
*
      ft = f
      fa = abs( ft )
      ht = h
      ha = abs( h )
*
*     PMAX points to the maximum absolute element of matrix
*       PMAX = 1 if F largest in absolute values
*       PMAX = 2 if G largest in absolute values
*       PMAX = 3 if H largest in absolute values
*
      pmax = 1
      swap = ( ha.GT.fa )
      IF( swap ) THEN
         pmax = 3
         temp = ft
         ft = ht
         ht = temp
         temp = fa
         fa = ha
         ha = temp
*
*        Now FA .ge. HA
*
      END IF
      gt = g
      ga = abs( gt )
      IF( ga.EQ.zero ) THEN
*
*        Diagonal matrix
*
         ssmin = ha
         ssmax = fa
         clt = one
         crt = one
         slt = zero
         srt = zero
      ELSE
         gasmal = .true.
         IF( ga.GT.fa ) THEN
            pmax = 2
            IF( ( fa / ga ).LT.dlamch( 'EPS' ) ) THEN
*
*              Case of very large GA
*
               gasmal = .false.
               ssmax = ga
               IF( ha.GT.one ) THEN
                  ssmin = fa / ( ga / ha )
               ELSE
                  ssmin = ( fa / ga )*ha
               END IF
               clt = one
               slt = ht / gt
               srt = one
               crt = ft / gt
            END IF
         END IF
         IF( gasmal ) THEN
*
*           Normal case
*
            d = fa - ha
            IF( d.EQ.fa ) THEN
*
*              Copes with infinite F or H
*
               l = one
            ELSE
               l = d / fa
            END IF
*
*           Note that 0 .le. L .le. 1
*
            m = gt / ft
*
*           Note that abs(M) .le. 1/macheps
*
            t = two - l
*
*           Note that T .ge. 1
*
            mm = m*m
            tt = t*t
            s = sqrt( tt+mm )
*
*           Note that 1 .le. S .le. 1 + 1/macheps
*
            IF( l.EQ.zero ) THEN
               r = abs( m )
            ELSE
               r = sqrt( l*l+mm )
            END IF
*
*           Note that 0 .le. R .le. 1 + 1/macheps
*
            a = half*( s+r )
*
*           Note that 1 .le. A .le. 1 + abs(M)
*
            ssmin = ha / a
            ssmax = fa*a
            IF( mm.EQ.zero ) THEN
*
*              Note that M is very tiny
*
               IF( l.EQ.zero ) THEN
                  t = sign( two, ft )*sign( one, gt )
               ELSE
                  t = gt / sign( d, ft ) + m / t
               END IF
            ELSE
               t = ( m / ( s+t )+m / ( r+l ) )*( one+a )
            END IF
            l = sqrt( t*t+four )
            crt = two / l
            srt = t / l
            clt = ( crt+srt*m ) / a
            slt = ( ht / ft )*srt / a
         END IF
      END IF
      IF( swap ) THEN
         csl = srt
         snl = crt
         csr = slt
         snr = clt
      ELSE
         csl = clt
         snl = slt
         csr = crt
         snr = srt
      END IF
*
*     Correct signs of SSMAX and SSMIN
*
      IF( pmax.EQ.1 )
     $   tsign = sign( one, csr )*sign( one, csl )*sign( one, f )
      IF( pmax.EQ.2 )
     $   tsign = sign( one, snr )*sign( one, csl )*sign( one, g )
      IF( pmax.EQ.3 )
     $   tsign = sign( one, snr )*sign( one, snl )*sign( one, h )
      ssmax = sign( ssmax, tsign )
      ssmin = sign( ssmin, tsign*sign( one, f )*sign( one, h ) )
      RETURN
*
*     End of DLASV2
*

---

## dlartg

   use la_constants, &
   only: wp=>dp, zero=>dzero, half=>dhalf, one=>done, &
         safmin=>dsafmin, safmax=>dsafmax
!
!  .. Scalar Arguments ..
   real(wp) :: c, f, g, r, s
!  ..
!  .. Local Scalars ..
   real(wp) :: d, f1, fs, g1, gs, u, rtmin, rtmax
!  ..
!  .. Intrinsic Functions ..
   intrinsic :: abs, sign, sqrt
!  ..
!  .. Constants ..
   rtmin = sqrt( safmin )
   rtmax = sqrt( safmax/2 )
!  ..
!  .. Executable Statements ..
!
   f1 = abs( f )
   g1 = abs( g )
   if( g == zero ) then
      c = one
      s = zero
      r = f
   else if( f == zero ) then
      c = zero
      s = sign( one, g )
      r = g1
   else if( f1 > rtmin .and. f1 < rtmax .and. &
            g1 > rtmin .and. g1 < rtmax ) then
      d = sqrt( f*f + g*g )
      c = f1 / d
      r = sign( d, f )
      s = g / r
   else
      u = min( safmax, max( safmin, f1, g1 ) )
      fs = f / u
      gs = g / u
      d = sqrt( fs*fs + gs*gs )
      c = abs( fs ) / d
      r = sign( d, f )
      s = gs / r
      r = r*u
   end if
   return