// This file includes untranslated text (ja).

# LAPACK 参照まとめ

Flair.md に言及されている LAPACK ルーティンとその依存関係
ソース: https://www.netlib.org/lapack/explore-html/

==============
## 1. DGEBRD
==============

URL: https://www.netlib.org/lapack/explore-html/dc/d1c/group__gebrd_ga1314f3a906c316785fe32996698901a8.html#ga1314f3a906c316785fe32996698901a8

### 概要
一般の実 M×N 行列 A を直交変換によって上または下二重対角形式 B に変換する。
  Q**T * A * P = B

- m >= n の場合: B は上二重対角行列
- m <  n の場合: B は下二重対角行列

### パラメータ

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| M         | INTEGER               | 行数 (M >= 0) |
| N         | INTEGER               | 列数 (N >= 0) |
| A         | DOUBLE PRECISION配列  | 入力行列 (LDA×N)。出力時は二重対角形式とリフレクタ情報で上書き |
| LDA       | INTEGER               | 先頭次元 (>= max(1,M)) |
| D         | DOUBLE PRECISION配列  | 二重対角行列の対角要素 |
| E         | DOUBLE PRECISION配列  | 副対角要素 (長さ min(M,N)-1) |
| TAUQ      | DOUBLE PRECISION配列  | Q リフレクタのスカラー因子 |
| TAUP      | DOUBLE PRECISION配列  | P リフレクタのスカラー因子 |
| WORK      | DOUBLE PRECISION配列  | ワークスペース (最適サイズ: (M+N)*NB) |
| LWORK     | INTEGER               | ワークスペース長 (>= max(M,N) if min(M,N) > 0) |
| INFO      | INTEGER               | ステータス (0=成功, <0=引数エラー) |

### アルゴリズム
ブロック化アルゴリズムを採用:
- ブロック化フェーズ: DLABRD で行/列を削減
- 行列更新: DGEMM で残部分行列を更新
- 非ブロック化フェーズ: 残余に対して DGEBD2 にフォールバック

### 依存関係（呼び出し先）
- DGEBD2: 非ブロック二重対角化
- DLABRD: ブロック化削減（最初の nb 行/列）
- DGEMM:  一般行列積（更新用）
- ILAENV: 最適ブロックサイズ決定

==============
## 2. DBDSQR
==============

URL: https://www.netlib.org/lapack/explore-html/d6/d51/group__bdsqr_gade20fbf9c91aa7de0c3d565b39588dc5.html#gade20fbf9c91aa7de0c3d565b39588dc5

### 概要
暗黙ゼロシフト QR アルゴリズムを使用して、実 N×N 二重対角行列 B の
特異値分解 (SVD) を計算する。SVD は以下の形式で与えられる:
  B = Q * S * P**T

- S: 特異値の対角行列
- Q: 左特異ベクトルの直交行列
- P: 右特異ベクトルの直交行列

DGEBRD で計算された A = U*B*VT に対しては:
  A = (U*Q) * S * (P**T*VT)
が A の SVD となる。

参考文献:
- Demmel & Kahan, LAPACK Working Note #3
  (SIAM J. Sci. Statist. Comput. vol.11, no.5, pp.873-912, Sept 1990)
- Parlett & Fernando, Technical Report CPAM-554, UC Berkeley, July 1992

### 関数シグネチャ
```fortran
subroutine dbdsqr(
  character uplo,
  integer n, ncvt, nru, ncc,
  double precision d(*), e(*),
  double precision vt(ldvt, *),
  integer ldvt,
  double precision u(ldu, *),
  integer ldu,
  double precision c(ldc, *),
  integer ldc,
  double precision work(*),
  integer info
)
```

### パラメータ

| パラメータ | 説明 |
|-----------|------|
| UPLO  | B が上 ('U') または下 ('L') 二重対角行列かを指定 |
| N     | 行列の次数 (N >= 0) |
| NCVT  | VT の列数 (NCVT >= 0) |
| NRU   | U の行数 (NRU >= 0) |
| NCC   | C の列数 (NCC >= 0) |
| D     | 対角要素 (入力)→ 降順の特異値 (出力) |
| E     | 副対角要素 (N-1 要素)。成功時は破棄 |
| VT    | N×NCVT 行列。P**T * VT で上書き |
| LDVT  | VT の先頭次元 |
| U     | NRU×N 行列。U * Q で上書き |
| LDU   | U の先頭次元 |
| C     | N×NCC 行列。Q**T * C で上書き |
| LDC   | C の先頭次元 |
| WORK  | ワーク配列 (次元: 4*N または 4*(N-1)) |
| INFO  | 0=成功, <0=無効な引数, >0=収束失敗 |

### 内部パラメータ
- TOLMUL: 収束基準制御 (デフォルト: max(10, min(100, EPS^(-1/8))))
- MAXITR: 内部ループの最大パス数 (デフォルト: 6)。実際の制限は MAXITR*N 反復

### 依存関係（呼び出し先）
- DLASQ1: 非回転ケースの高速 SVD
- DLARTG: 平面回転の生成
- DLASR:  回転列の適用
- DLASV2: 2×2 行列の SVD
- DROT:   ベクトル回転
- DSCAL:  ベクトルスケーリング
- DSWAP:  ベクトル交換

==============
## 3. DGESVD
==============

URL: https://www.netlib.org/lapack/explore-html/d1/d7f/group__gesvd_gac6bd5d4e645049e49bb70691180abf07.html#gac6bd5d4e645049e49bb70691180abf07

### 概要
実 M×N 行列 A の特異値分解 (SVD) を計算し、オプションで左および/または
右特異ベクトルも計算する。SVD は以下の形式:
  A = U * SIGMA * transpose(V)

- SIGMA: M×N 行列 (min(m,n) 個の非ゼロ対角要素のみ)
- U: M×M 直交行列
- V: N×N 直交行列
- 特異値は実数・非負・降順で返される
- 返されるのは V**T (V の転置) であり V ではない

### 関数シグネチャ
```fortran
subroutine dgesvd(character jobu, character jobvt, integer m, integer n,
                   double precision a(lda, *), integer lda,
                   double precision s(*), double precision u(ldu, *),
                   integer ldu, double precision vt(ldvt, *),
                   integer ldvt, double precision work(*),
                   integer lwork, integer info)
```

### パラメータ

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| JOBU   | CHARACTER*1       | 左特異ベクトルの計算方法: 'A'(全M列), 'S'(min(m,n)列), 'O'(Aに上書き), 'N'(計算しない) |
| JOBVT  | CHARACTER*1       | 右特異ベクトルの計算方法: 'A'(全N行), 'S'(min(m,n)行), 'O'(Aに上書き), 'N'(計算しない) |
| M      | INTEGER           | 行列 A の行数 |
| N      | INTEGER           | 行列 A の列数 |
| A      | DOUBLE配列(LDA,N) | 入力行列。終了時に破棄または上書き |
| LDA    | INTEGER           | A の先頭次元 (>= max(1,M)) |
| S      | DOUBLE配列        | 降順の特異値 (min(M,N) 個) |
| U      | DOUBLE配列(LDU,*) | 左特異ベクトル (JOBU に依存) |
| LDU    | INTEGER           | U の先頭次元 (>=1; JOBU='S'or'A' なら >=M) |
| VT     | DOUBLE配列(LDVT,N)| 右特異ベクトルの転置 (JOBVT に依存) |
| LDVT   | INTEGER           | VT の先頭次元 |
| WORK   | DOUBLE配列        | ワークスペース。WORK(1) に最適サイズを返す |
| LWORK  | INTEGER           | WORK 配列の次元 (-1 でサイズ問い合わせ) |
| INFO   | INTEGER           | 0=成功, <0=引数エラー, >0=収束失敗 |

### ワークスペース要件
- Path 1, 1t: MAX(1, 5*MIN(M,N))
- その他:     MAX(1, 3*MIN(M,N) + MAX(M,N), 5*MIN(M,N))
- LWORK=-1 で最適サイズを問い合わせ可能

### 計算パス
- Path 1-9:   M >= N の各種ベクトル組み合わせ (有益な場合 QR を使用)
- Path 1t-9t: N > M の各種ベクトル組み合わせ (有益な場合 LQ を使用)
- Path 10/10t: 適度な次元比 (直接二重対角化)

### 注意事項
- JOBU と JOBVT は同時に 'O' にはできない

### 依存関係（呼び出し先）
- DGEBRD:  二重対角化
- DGEQRF:  QR 分解
- DORGLQ:  直交行列生成 (LQ)
- DORGQR:  直交行列生成 (QR)
- DBDSQR:  二重対角 SVD 反復
- DORGBR:  二重対角化ベクトル生成
- DLACPY:  行列コピー
- DLASET:  行列初期化
- DLANGE:  行列ノルム
- DLASCL:  行列スケーリング

==============
## 4. DGESDD
==============

URL: https://www.netlib.org/lapack/explore-html/df/d22/group__gesdd_ga8941e5ff50de36580dae8940015e9cb0.html#ga8941e5ff50de36580dae8940015e9cb0

### 概要
分割統治アルゴリズムを使用して、実 M×N 行列 A の特異値分解 (SVD) を計算する。
特異ベクトルが必要な場合は分割統治アルゴリズムを使用する。
  A = U * SIGMA * transpose(V)

- 返されるのは VT = V**T (V の転置) であり V ではない

### 関数シグネチャ
```fortran
subroutine dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt,
                  work, lwork, iwork, info)
```

### パラメータ

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| JOBZ   | CHARACTER*1       | 特異ベクトルの計算方法: 'A'(全列/行), 'S'(min(M,N)列/行), 'O'(Aに上書き), 'N'(計算しない) |
| M      | INTEGER           | 行数 (M >= 0) |
| N      | INTEGER           | 列数 (N >= 0) |
| A      | DOUBLE配列        | 入力行列。JOBZ='O' 以外は終了時に破棄 |
| LDA    | INTEGER           | A の先頭次元 (>= max(1,M)) |
| S      | DOUBLE配列        | 降順の特異値 |
| U      | DOUBLE配列        | 左特異ベクトル |
| LDU    | INTEGER           | U の先頭次元 |
| VT     | DOUBLE配列        | 右特異ベクトルの転置 |
| LDVT   | INTEGER           | VT の先頭次元 |
| WORK   | DOUBLE配列        | ワークスペース |
| LWORK  | INTEGER           | ワークスペース次元 (-1 でサイズ問い合わせ) |
| IWORK  | INTEGER配列       | 整数ワークスペース (次元: 8×min(M,N)) |
| INFO   | INTEGER           | 0=成功, <0=無効な引数, >0=収束失敗 |

### ワークスペース要件 (最小)
- JOBZ='N': lwork >= 3×min(M,N) + max(M, 7×min(M,N))
- JOBZ='O': lwork >= 3×min(M,N) + max(M, 5×min(M,N)²+4×min(M,N))
- JOBZ='S': lwork >= 4×min(M,N)² + 7×min(M,N)
- JOBZ='A': lwork >= 4×min(M,N)² + 6×min(M,N) + max(M,N)

### アルゴリズム戦略
M >= N の場合:
- M >> N なら: まず QR 分解を適用してから三角行列に対して SVD
- それ以外:    直接二重対角化

N > M の場合:
- N >> M なら: まず LQ 分解を適用してから三角行列に対して SVD
- それ以外:    直接二重対角化

### スケーリングとエラー処理
入力要素が安全な数値範囲外の場合は自動スケーリングし、特異値を適切に再スケーリング。
NaN エントリを検出し、計算前に全入力パラメータを検証する。

### ソース
ファイル: dgesdd.f (LAPACK 3.12.1), 行: 209

### 依存関係（呼び出し先）
- DGEBRD:  二重対角化
- DBDSDC:  分割統治による二重対角 SVD
- DGEQRF:  QR 分解
- DGELQF:  LQ 分解
- DORGQR:  直交行列生成 (QR)
- DORGLQ:  直交行列生成 (LQ)
- DORGBR:  二重対角化ベクトル生成
- DORMBR:  直交変換適用
- DLACPY:  行列コピー
- DLASET:  行列初期化
- DLASCL:  行列スケーリング
- DLANGE:  行列ノルム
- DLAMCH:  マシン定数

==============
## 依存関係グラフ（概要）
==============

DGESVD ──┬── DGEBRD ──┬── DGEBD2
         │             ├── DLABRD
         │             └── DGEMM
         └── DBDSQR ──┬── DLASQ1
                      ├── DLARTG
                      ├── DLASR
                      ├── DLASV2
                      ├── DROT
                      ├── DSCAL
                      └── DSWAP

DGESDD ──┬── DGEBRD  (上記と同じ)
         ├── DBDSDC  (分割統治版、DBDSQR の代替)
         ├── DGEQRF
         ├── DGELQF
         ├── DORGQR / DORGLQ / DORGBR / DORMBR
         └── DLACPY / DLASET / DLASCL / DLANGE / DLAMCH
