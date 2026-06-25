# flair

flair 時系列予測アルゴリズム

## ライセンスと帰属

本コードは **Apache License, Version 2.0** で配布される。ライセンス全文は同梱の
[`LICENSE`](./LICENSE)、第三者著作物の帰属表示は [`NOTICE`](./NOTICE) を参照。

### 第三者由来コードの一覧

| ファイル | 由来 | 由来ライセンス | 関係 |
|---|---|---|---|
| `src/svd.rs` | [nalgebra v0.35.0](https://github.com/dimforge/nalgebra/tree/v0.35.0)（Copyright 2020 Sébastien Crozet） | Apache-2.0 | 移植・改変（`Vec<f64>` 特殊化） |
| `src/constants.rs` | [FLAIR (flaircast)](https://github.com/Mellon-Inc/FLAIR) `flaircast/_constants.py`（Copyright 2026 Takato Honda） | Apache-2.0 | Rust へ移植 |
| `src/frequency.rs` | 同上 `flaircast/_frequency.py` | Apache-2.0 | Rust へ移植 |

> いずれも由来は Apache-2.0。Apache-2.0 §4(b) に従い「改変ファイルである」旨を各ファイル
> 冒頭の SPDX ヘッダ／出典コメントに明記し、§4(d) に従い原著作権表示を `NOTICE` に集約している。

### 参照のみ（コード非流用）の素材

以下は **数学アルゴリズム（アイデア）のみを参照**しており、ソースコードは一切複製していない。
独立実装である。

- `src/linalg.rs`: Golub-Reinsch SVD (1970) の独自実装。
  **Numerical Recipes 掲載コードは独自ライセンスで複製禁止のため一切流用していない。**
- `src/linalg.rs` `brentq`: Brent (1973) の根探索アルゴリズム。
- `src/svd.rs` の 2×2 SVD: Qiao & Wang の論文アルゴリズム。
- `lapack.txt` / 本書の LAPACK 節: [LAPACK](https://www.netlib.org/lapack/) ルーチンの
  説明（API ドキュメントの参照であり、Fortran ソースの複製ではない）。

### 採用時チェックリスト（flair-rs へ取り込む場合）

- [ ] `LICENSE`（Apache-2.0 全文）をリポジトリ直下に配置（既にあれば統合）。
- [ ] `NOTICE` の第三者帰属（nalgebra / FLAIR）を flair-rs の `NOTICE` にマージ。
- [ ] 各 `.rs` 冒頭の `SPDX-License-Identifier: Apache-2.0` ヘッダを維持。
- [ ] `Cargo.toml` に `license = "Apache-2.0"` を追加（現状未設定）。
- [ ] `linalg.rs` を採用する場合、Numerical Recipes 非流用の注記を残す。

## Lapack

### dgebrd

```
!> DGEBRD reduces a general real M-by-N matrix A to upper or lower
!> bidiagonal form B by an orthogonal transformation: Q**T * A * P = B.
!>
!> If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.
```
- https://www.netlib.org/lapack/explore-html/dc/d1c/group__gebrd_ga1314f3a906c316785fe32996698901a8.html#ga1314f3a906c316785fe32996698901a8

### dbdsqr

```
!> DBDSQR computes the singular values and, optionally, the right and/or
!> left singular vectors from the singular value decomposition (SVD) of
!> a real N-by-N (upper or lower) bidiagonal matrix B using the implicit
!> zero-shift QR algorithm.  The SVD of B has the form
!>
!>    B = Q * S * P**T
!>
!> where S is the diagonal matrix of singular values, Q is an orthogonal
!> matrix of left singular vectors, and P is an orthogonal matrix of
!> right singular vectors.  If left singular vectors are requested, this
!> subroutine actually returns U*Q instead of Q, and, if right singular
!> vectors are requested, this subroutine returns P**T*VT instead of
!> P**T, for given real input matrices U and VT.  When U and VT are the
!> orthogonal matrices that reduce a general matrix A to bidiagonal
!> form:  A = U*B*VT, as computed by DGEBRD, then
!>
!>    A = (U*Q) * S * (P**T*VT)
!>
!> is the SVD of A.  Optionally, the subroutine may also compute Q**T*C
!> for a given real input matrix C.
!>
!> See  by J. Demmel and W. Kahan,
!> LAPACK Working Note #3 (or SIAM J. Sci. Statist. Comput. vol. 11,
!> no. 5, pp. 873-912, Sept 1990) and
!>  by
!> B. Parlett and V. Fernando, Technical Report CPAM-554, Mathematics
!> Department, University of California at Berkeley, July 1992
!> for a detailed description of the algorithm.
```

- https://www.netlib.org/lapack/explore-html/d6/d51/group__bdsqr_gade20fbf9c91aa7de0c3d565b39588dc5.html#gade20fbf9c91aa7de0c3d565b39588dc5

## dgesvd

```
!> DGESVD computes the singular value decomposition (SVD) of a real
!> M-by-N matrix A, optionally computing the left and/or right singular
!> vectors. The SVD is written
!>
!>      A = U * SIGMA * transpose(V)
!>
!> where SIGMA is an M-by-N matrix which is zero except for its
!> min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
!> V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
!> are the singular values of A; they are real and non-negative, and
!> are returned in descending order.  The first min(m,n) columns of
!> U and V are the left and right singular vectors of A.
!>
!> Note that the routine returns V**T, not V.
```

- https://www.netlib.org/lapack/explore-html/d1/d7f/group__gesvd_gac6bd5d4e645049e49bb70691180abf07.html#gac6bd5d4e645049e49bb70691180abf07

## dgesdd

```
!> DGESDD computes the singular value decomposition (SVD) of a real
!> M-by-N matrix A, optionally computing the left and right singular
!> vectors.  If singular vectors are desired, it uses a
!> divide-and-conquer algorithm.
!>
!> The SVD is written
!>
!>      A = U * SIGMA * transpose(V)
!>
!> where SIGMA is an M-by-N matrix which is zero except for its
!> min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
!> V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
!> are the singular values of A; they are real and non-negative, and
!> are returned in descending order.  The first min(m,n) columns of
!> U and V are the left and right singular vectors of A.
!>
!> Note that the routine returns VT = V**T, not V.
```

- https://www.netlib.org/lapack/explore-html/df/d22/group__gesdd_ga8941e5ff50de36580dae8940015e9cb0.html#ga8941e5ff50de36580dae8940015e9cb0

## Logic Flow

1. 学習フェーズ: 過去データ
  ↓ 前処理（ポジティブシフト）
  ↓ P選択（BIC）
  ↓ 行列折り畳み → Level × Shape分解
  ↓ OptShrink → Levelのノイズ補正
  ↓ Box-Cox変換 + NLinearセンタリング
  ↓ Ridge（LSR1差分ターゲット）→ β₀,β₁,β₂,(β₃)
  ↓ ダンプトレンド φ の推定
  ↓ LWCP → h_test[j]

2. 予測フェーズ: 200サンプル生成
  1. Levelノイズ（bootstrap LOO残差 × sqrt(1+h_test)）→ level_hat 200本
  2. フェーズノイズ（残差行列Rから列サンプリング × デフレーション）
  3. 予測値 = level_hat × shape × (1 + フェーズノイズ) - y_shift

## 工程

1. Level × Shape分解（FLAIRの根幹）
2. Period選択とBIC（「P=何？」を自動で決める）
3. OptShrink（ノイズ除去の最適化）
4. Box-Cox変換
5. Ridge回帰とLOOCV（LSR1差分ターゲット含む）
6. LWCP（予測区間の幅をどう決めるか）
7. フェーズノイズとブートストラップ（確率的サンプルパスの生成）
8. フォールバック・カスケード

## 1. levelとshape

以下の二項対立概念で、時系列データを単純化する
- level: 傾向。データの初めから終わりまでを貫く一過性のファンクション。
- shape: 周期。季節とも言う。flairでは、このShapeをグローバルに1つだけ（最近K=2周期の平均）に固定(frozen shapeと呼称)する。

## 2. Period

データ全体
    ├── Level  … 各周期をひとまとめにした「大きさの時系列」
    └── Shape  … 1周期内の「配分パターン」
                   └── Periodに依存する（P=12なら長さ12のベクトル）

  ShapeはPが決まって初めて「何次元のベクトルか」が決まります。P=12なら12個の割合、P=7なら7個の割合。

  ---
  ただ「LevelはPに依存しないか？」というとそうでもなくて：

  P=12 → L = 12ステップごとの合計 → n_c 個の値（n_c = 完全周期の数）
  P=7  → L = 7ステップごとの合計  → n_c 個の値

  PはLevelとShape両方の形を決める大元です。

  **実装上の制限**：`_MAX_COMPLETE = 500`。n_c の上限。長い系列ではメモリ・速度ガードとして直近500周期だけを Level fitting に使う。また `_MIN_COMPLETE = 3` が最小値で、これを下回ると Branch 2 にフォールバック。

  ---
  なので学習の順番は：

  1. Pを決める（MDL/BIC）
  2. Pに基づいてmatrixに折り畳む（n_c は min(完全周期数, 500)）
  3. Levelを取り出す（各列の合計）
  4. Shapeを取り出す（各行の割合）
  5. LevelをRidgeで予測
  6. Level × Shapeで最終予測を組み立てる

問題設定

freq="H" と指定されたとき、候補は [24, 168] です（24時間周期 or
168時間=週周期）。どちらが「このデータに合っているか」を自動で選びたい。

---
BIC（ベイズ情報量基準）とは

「モデルのフィット度」と「モデルの複雑さへの罰則」を足したスコアです：

BIC = データへのフィットの悪さ + パラメータ数 × log(データ数)

小さいほど良いモデル。フィットが良くてもパラメータが多すぎると罰則が大きくなります。

直感的には「シンプルなのに良く当てはまるモデルを選ぶ」基準です。

---
FLAIRでのBICの使い方

各候補Pについて、データをP×n_c行列に折り畳んでSVD（特異値分解）します。

SVDで行列を「rank-1成分（Level×Shape）」と「残差」に分解したとき：

rank-1成分 = σ₁ × u₁ × v₁ᵀ  ← Levelっぽい構造
残差       = Σᵢ₌₂ σᵢ²        ← 説明できなかった部分（= RSS₁）

σ₁, σ₂, ... は特異値（大きいほどその成分が重要）。

---
SVDの直感

行列を「重要な構造の層」に分解する操作です：

P×n_c行列 = σ₁×(パターン1) + σ₂×(パターン2) + σ₃×(パターン3) + ...
              ↑一番重要        ↑二番目          ↑三番目

FLAIRはrank-1（一番重要な層だけ）でLevelとShapeを表現します。残差（σ₂以降）が小さいほど「このPでうまく説明できている」。

---
BICの式

$$P^* = \arg\min_{P_c \in \{1\} \cup \mathcal{C}} \Big[ n \log\!\big(\text{RSS}(P_c) / n\big) + k(P_c) \log n \Big]$$

パラメータ数 $k(P_c)$：
- $P_c \geq 2$ の場合：$k(P_c) = P_c + n_c - 1$（Shape の P 個 ＋ Level 系列の $n_c - 1$ 自由度）
- $P_c = 1$（ヌルモデル）の場合：$k(1) = 1$

**注意**：$k$ の計算に使うのは $n$（総観測数）ではなく $n_c$（完全周期の数）。例えば hourly で $n=1000$、$P=24$ なら $n_c = 41$、$k = 24 + 41 - 1 = 64$。

各候補PのBIC（具体例）：

P=24のBIC  = n × log(RSS₁(P=24) / n) + (24 + n_c - 1) × log(n)
P=168のBIC = n × log(RSS₁(P=168) / n) + (168 + n_c - 1) × log(n)

P=168は残差が小さくなりやすい（週パターンまで捉えられる）が、パラメータ数が多い分罰則も大きい。この天秤でどちらが良いか決まります。

---
P=1 nullモデル

「そもそも周期性がない」という仮説も候補に入れます：

P=1のBIC = n × log(Var(y)) + 1 × log(n)

「ただの平均+ノイズ」モデル（k=1）。これより良いPが見つからなければ周期性なしと判断し、Branch 2（生系列Ridge）に移行します。

---
現在のRustコードとの差分

現在のRustはP=1 nullがない：

```rust
// 現在（0.1.0相当）
// 候補が1つしかないとき即座にそれを選ぶ
if candidates.len() == 1 {
    candidates[0]
} else {
    // BICで比較（でもP=1 nullなし）
}

実装後：

// P=1 null を常に基準として計算
let rss_null = variance(y_sel) * t_max;
let bic_null = t_max * ln(rss_null / t_max) + ln(t_max);
let mut best_bic = bic_null;  // nullが基準
let mut best_p = 1;

for p_cand in candidates {
    let bic = ...;
    if bic < best_bic { best_p = p_cand; best_bic = bic; }
}
```

```rust
fn get_periods(freq: &str) -> Vec<usize> {
      match resolve_freq(freq).as_str() {
          "H" => vec![24, 168],  // ← ハードコード // カレンダー周期で決め打ち
          "D" => vec![7, 365],
          ...
      }
  }
```

# OptShrink（Gavish-Donoho 2014）

問題設定

BICでPが決まり、データをP×n行列に折り畳んでSVDしました。

rank-1成分（σ₁ × u₁ × v₁ᵀ）がLevelとShapeの構造を表しているはずですが、σ₁自体がノイズで膨らんでいる可能性があります。

---
なぜσ₁が膨らむか

データにはノイズが乗っています。ノイズのない「真のσ₁」より、観測データから計算した「実測σ₁」は必ず大きくなります。

実測σ₁ = 真のσ₁ + ノイズによる膨らみ

膨らんだσ₁をそのまま使うと、Levelが過大推定されます。

---
OptShrinkの発想

「ノイズがランダム行列だとしたら、σ₁はどのくらい膨らむか」を数学的に計算して、最適な縮小率cを求める。

補正後のσ₁ = 実測σ₁ × c   （c ∈ (0, 1]）

cは「真の信号がノイズより十分強ければ1に近い」「ノイズに埋もれていれば小さくなる」。

---
ランダム行列のノイズ水準をどう推定するか

ここでMarchenko-Pastur分布が登場します。

「純粋なノイズだけからなるP×n行列のSVD」をやったとき、特異値はどう分布するか？という問いへの答えがMarchenko-Pastur分布です。

ノイズ行列の特異値の中央値 ≈ σ_noise × √(μ_β)

逆に言えば：

σ_noise ≈ 実測の特異値の中央値 / √(μ_β)

実測の特異値の中央値から「ノイズの大きさ」を推定できる。これがOptShrinkの核心です。

---
μ_βとは

β = min(P,n) / max(P,n)（行列のアスペクト比）に対応するMarchenko-Pastur分布の中央値です。

β=1（正方行列）→ μ_β ≈ 0.57
β=0.1（細長い行列）→ μ_β ≈ 0.27

これが _mp_median 関数で数値的に計算している値です。閉じた式がなく数値積分が必要なため、Rustで自前実装が必要な唯一の要素でした。

---
縮小率cの計算（Gavish-Donoho式）

A = σ₁² - (1+β) × σ_noise²
c = √(A + √(A² - 4β × σ_noise⁴)) / (√2 × σ₁)

σ₁が閾値 (1+√β) × σ_noise 以下なら信号がノイズに埋もれているとみなしてc=1（縮小しない）。

---
FLAIRでの「One SVD」原則

BICのためにSVDはすでに1回やっています。OptShrinkはその同じSVD結果を使い回すので追加のSVD計算ゼロ。これが「One SVD」原則です。

```rust
# _select_period がSVD結果を返す
P, secondary, period, cal, svd_s, nc_svd = _select_period(y, n, freq)

# OptShrinkはsvd_sを使い回す（再計算なし）
L = L * _optshrink_factor(svd_s, P, nc_svd)
```

## 3. box cox 変換

- bc_lambda()で、変換後のLevelが最も正規分布に近くなるλ[0,1]を計算し、box_to_cox()を行う。後で戻す。
- 正値観測が `_MIN_POSITIVE_FOR_BC = 10` 未満の場合は λ=1 固定（Box-Cox適用なし）。
- 逆Box-Cox の指数は `_BC_EXP_CLIP = 30` でクリップ（オーバーフロー防止）。

```
L（実数空間）
  ↓ bc(L, λ)              ← Box-Cox変換
L_bc（変換空間）
  ↓ L_innov[i] = L_bc[i] - L_bc[n_c]  ← NLinearセンタリング（最終観測値を引く）
L_innov（センタリング済み変換空間）
  ↓ Ridge予測
L_bc_forecast
  ↓ bc_inv(・, λ)         ← 逆変換
L_forecast（実数空間、必ず正）
```

**NLinearセンタリングについて**

Box-Cox変換後に最終観測値 $L^{(\lambda)}_{n_c}$ を引く：

$$L^\text{innov}_i = L^{(\lambda)}_i - L^{(\lambda)}_{n_c}$$

これにより Ridge の正則化パラメータ α → ∞ の極限でモデルが Seasonal Naïve（「最後の観測値をそのまま繰り返す」）に収束する性質が得られる。つまり Seasonal Naïve が正則化パスの安全な右端に位置し、Ridgeはそこからの「改善」方向にしか動かない。

## 4. 最小二乗法とRidge回帰

level、つまり下記の式で次のlevelを予測する：

$$L^\text{innov}_i = \beta_0 + \beta_1 (i/n_c) + \beta_2 \cdot L^\text{innov}_{i-1} + \beta_3 \cdot L^\text{innov}_{i-\text{sec}} + \varepsilon_i$$

- $\beta_0, \beta_1, \beta_2$ は常に使う3係数（p=3）
- $\beta_3 \cdot L^\text{innov}_{i-\text{sec}}$ は**2次周期ラグ**。hourlyなら「1週間前の同じ時間帯のLevel」（sec=7）。2次周期がない場合はこの項を省略（p=3）、ある場合はp=4となる。

**自由度ガード（DoF Guard）**：`n_train < 2p` のとき（学習点数が特徴量数の2倍未満）、ラグ特徴量を削除してp=1に縮退し、強制的に P=1 フォールバックに移行する。短い系列でリッジが不安定になるのを防ぐ。

β₀, β₁, β₂, (β₃) は定数の係数。これを導出するのが、回帰。

Ridget回帰とは、予測値と実際の値のズレ（残差）の二乗和を最小にする係数を選ぶ(最小二乗法)際に、係数の種数の発散(過学習)を防ぐ回帰手法。

最小二乗法に「係数が大きくなりすぎることへの罰則」を追加したものです：

最小化したいもの = Σ(残差²) + α × Σ(β²)

α（アルファ）が正則化パラメータ。大きいほど係数を0に近づける力が強くなります。

- α = 0 → 最小二乗法と同じ
- α → ∞ → 全係数が0に（予測しない）
- α = 適切な値 → 過学習を防ぎながら予測できる

---
αはどう決める？ → LOOCV

α をデータから自動で決めるのが**LOOCV（Leave-One-Out Cross Validation、一個抜き交差検証）**です。

やり方：

データが n 点あるとき、
  1点目を抜いて残り(n-1)点で学習 → 抜いた1点を予測 → 誤差を記録
  2点目を抜いて残り(n-1)点で学習 → 抜いた2点を予測 → 誤差を記録
  ...
  n点目を抜いて残り(n-1)点で学習 → 抜いた1点を予測 → 誤差を記録

全誤差の平均が最小になるαを選ぶ

「もし過去のこのデータが無かったら、うまく予測できたか？」を全データ点で試す、という発想です。

---
FLAIRの工夫：Soft-Average GCV

普通にLOOCVをやると「25種類のαそれぞれについてn回学習」が必要で重い。

FLAIRはSVDを1回やれば全αのGCV誤差が閉形式で計算できるという数学的トリックを使っています。25個のRidge解を1つのSVDから取り出す（One SVD for Ridge）ことで高速化しています。

さらに「一番良いαを選ぶ」のではなく、複数のαを誤差に応じた重みで混ぜる（Soft-Average）ことで予測を安定させています。ハード選択より relMASE 1.1%・relCRPS 2.0% の改善。

// αごとのGCV誤差をsoftmax重みに変換して平均
let w = softmax(-(gcv - gcv_min) / gcv_min);
beta = Σ w[i] × beta_alpha[i]

**注意**：「One SVD」という言葉はOptShrinkの文脈でも使われる（BICのSVDをOptShrinkが使い回す原則）。混同に注意。OptShrinkについては該当セクションを参照。

## 5. LSR1差分ターゲット（ランダムウォーク事前分布）

まず「ランダムウォーク」とは

毎ステップ「今の値 + ランダムなノイズ」で次の値が決まる系列です：

L[t] = L[t-1] + ε   （εはランダムなノイズ）

株価や気温の日次変化がこれに近い。「今日の値から明日を予測する最良の推測は今日の値そのもの」という性質があります。

これをRidgeの式に当てはめると：

β₀ ≈ 0, β₁ ≈ 0, β₂ ≈ 1

つまりβ₂=1がランダムウォークの自然な姿です。

---
問題：Ridgeはβ₂を0に引っ張る

Ridgeの罰則 α × Σ(β²) は全係数を0に向かって縮小させます。

でもランダムウォーク的なLevelに対して「β₂を0に近づける」のは間違った方向への縮小です。本来β₂≈1であるべきなのに、罰則が邪魔をします。

---
LSR1の解決策：変数を変換する

「β₂を0に縮小する」代わりに、**「β₂からの乖離（δ₂ = 1 − β₂）を0に縮小する」**ように式を書き換えます。

元の式：
L[t] = β₀ + β₁×t + β₂×L[t-1] + ε

両辺から L[t-1] を引く：
L[t] - L[t-1] = β₀ + β₁×t + (β₂-1)×L[t-1] + ε
ΔL[t]         = β₀ + β₁×t + (-δ₂)×L[t-1] + ε

ここで δ₂ = 1 − β₂ と置くと：

ΔL[t] = β₀ + β₁×t + δ₂×(-L[t-1]) + ε

予測ターゲットが L から ΔL（差分）に変わった。Ridgeが縮小するのは δ₂ で、δ₂→0 は β₂→1（ランダムウォーク）に対応します。

---
Pythonコードとの対応

# _DIFF_TARGET = True のとき
y_target = np.diff(L_innov[start-1:])      # ΔL を予測
X_full[:, nb] = -L_innov[start-1:-1]       # -L[t-1] を特徴量に

# Ridgeが返すのは δ₂、β₂に戻すには：
beta[nb] = 1.0 - theta[nb]                 # β₂ = 1 - δ₂

---
現在のRustコードとの差分

現在のRustは _DIFF_TARGET が未実装で、L_innov を直接予測しています：

// 現在（0.1.0相当）
row[nb] = l_innov[ti - 1];   // +L[t-1]
// y_target = l_innov[start..]  ← Lをそのまま予測

変更後はこうなります：

// LSR1実装後
row[nb] = -l_innov[ti - 1];  // -L[t-1]（符号反転）
// y_target = diff(l_innov)   ← ΔLを予測
// beta[nb] = 1.0 - theta[nb] ← δ₂からβ₂に戻す

変更箇所は少ないですが、予測ループ側は変更不要（beta を復元した後は同じ式を使うので）。

---
まとめ

┌─────────────────────┬──────────┬────────────────────────┐
│                     │  現Rust  │ LSR1（Python 0.3.0〜） │
├─────────────────────┼──────────┼────────────────────────┤
│ 予測ターゲット      │ L[t]     │ ΔL[t]                  │
├─────────────────────┼──────────┼────────────────────────┤
│ ラグ特徴量          │ +L[t-1]  │ -L[t-1]                │
├─────────────────────┼──────────┼────────────────────────┤
│ Ridgeが縮小するもの │ β₂ → 0   │ δ₂ → 0（つまりβ₂ → 1） │
├─────────────────────┼──────────┼────────────────────────┤
│ 事前分布の意味      │ 平均回帰 │ ランダムウォーク       │
└─────────────────────┴──────────┴────────────────────────┘

---
ダンプトレンド（damped trend）

Level予測の再構成時に、長期的なトレンドが発散しないよう減衰させます：

$$\phi = \max(\hat{\rho}_1(\Delta L),\ 0)\ \wedge\ (1 - \varepsilon)$$

- $\hat{\rho}_1(\Delta L)$：Level差分 ΔL の1次自己相関係数
- $\max(\cdot, 0)$：負の自己相関（平均回帰的）は無視し、減衰をかけない
- $\wedge (1 - \varepsilon)$：1.0 に張り付かないようにクリップ（非MLE、保守的推定）

この $\phi$ で将来のLevel変化を指数的に減衰させることで、長ホライズンでの分散が無制限に広がるのを防ぐ。

Levelの話はここで一区切りです。次はLWCP（予測区間の幅をどう決めるか）に進みます。ここから「点予測」から「確率的予測」の話になります

## 6. LWCP（Leverage-Weighted Conformal Prediction）

まず「予測区間の幅」の問題から

Levelノイズのεの大きさは「過去のLOOCV残差」から決めると言いました。

具体的には：

過去の予測ミスの大きさ ≈ εの標準偏差

でもこれは「1ステップ先の予測ミス」から推定した値です。

問題：1ステップ先と12ステップ先では、不確かさの大きさが違うはずです：

1ステップ先:  ほぼ確実 → 区間が狭い
6ステップ先:  そこそこ不確か → 区間が中くらい
12ステップ先: かなり不確か → 区間が広い

ホライズンが遠いほど区間が広がるべき。これをどう計算するか？

---
Leverageとは

Ridge回帰の文脈で、**「この訓練データ点は予測にどれだけ影響力があるか」**を表す数値です。記号は h_ii（0〜1の値）。

直感的には：

h_ii が大きい ← この点を抜いたら予測が大きく変わる（影響力大）
h_ii が小さい ← この点を抜いてもほぼ変わらない（影響力小）

LOOCVの「1点抜いて再学習」を毎回やらなくても、h_ii を使えば抜いたときの残差が計算できるというのがRidgeの数学的性質です：

LOO残差[i] = 実際の残差[i] / (1 - h_ii)

---
テスト点のLeverage（h_test）

訓練データではなく予測したい未来の点にもLeverageが計算できます。

未来の点は「訓練データから遠い」ほどLeverageが大きくなります。ホライズンが遠いほど、訓練データから遠い → h_test が大きい。

---
LWCPの本体

予測区間の幅を sqrt(1 + h_test) でスケールします：

εのスケール = σ_loo × sqrt(1 + h_test[j])
                ↑過去の      ↑ホライズンjの
                残差の大きさ   不確かさの拡大率

- h_test[0] （1ステップ先）→ 小さい → sqrt(1+小) ≈ 1 → 区間ほぼそのまま
- h_test[11] （12ステップ先）→ 大きい → sqrt(1+大) >> 1 → 区間が広がる

---
LWCP正規化（訓練側）

訓練側のLOOCV残差も同じスケールで割っておきます：

```
# Pythonコード（_level.py）
loo_raw = residuals / (1 - h_avg)          # 通常のLOO残差
loo = loo_raw / sqrt(1 + h_avg)            # LWCP正規化
```

「訓練データ点のLeverageによるばらつきを除去した、純粋な予測誤差」にする操作です。これで訓練残差とテスト点のスケールが揃います。

---
現在のRustコードとの差分

現在のRustの ridge_sa はLOO残差に / (1-h) だけ適用してLWCP正規化なし：

```
// 現在（0.1.0相当）
let loo: Vec<f64> = residuals.iter().zip(h_avg.iter())
    .map(|(&ri, &hi)| ri / (1.0 - hi).max(EPS))
    .collect();
// h_test の計算自体が存在しない

// LWCP実装後
let loo: Vec<f64> = residuals.iter().zip(h_avg.iter())
    .map(|(&ri, &hi)| ri / (1.0 - hi).max(EPS) / sqrt(1.0 + hi))
    .collect();
// + h_test[j] をホライズンごとに計算する関数が必要
```

## 7. フェーズノイズとブートストラップ

フェーズノイズとは

Levelノイズ（①）はLevel系列の不確かさを表していました。フェーズノイズ（②）は**「同じ周期内の各位相が、典型的なShapeからどのくらいズレるか」**の不確かさです。

- フェーズノイズの計算に使うウィンドウは直近 `_PHASE_NOISE_K = 50` 周期に限定。

例えば月次データで：

典型的なShape: 1月=8.3%, 2月=6.7%, 3月=7.5%, ...
ある年の実際:  1月=8.8%, 2月=6.2%, 3月=7.9%, ...  ← ズレがある
別の年の実際:  1月=7.9%, 2月=7.1%, 3月=7.2%, ...  ← 別のズレ

このズレを「相対残差」として記録したのが残差行列Rです：

R[phase][period] = (実際 - 典型) / 典型
                  = (mat - L×S) / (L×S)

  ---
ブートストラップとは

「過去の残差をそのまま再利用して未来のノイズを模倣する」手法です。

正規分布などの仮定を置かず、実際に観測されたばらつきのパターンをそのまま使います：

過去50周期分の残差行列R（P×50）
  ↓
サンプリング時に「列ごと」ランダムに選ぶ
  ↓
選んだ列をそのままフェーズノイズとして使う

---
「列ごと」が重要

列 = 1周期分の全位相のセットです。

列3を選んだ場合:
  1月のノイズ = R[0][3]
  2月のノイズ = R[1][3]
  3月のノイズ = R[2][3]
  ...

1つの周期の中での相関が保たれる。これが先ほどの「シナリオコヒーレント」の実体です。独立にランダムサンプリングしてしまうと：

1月だけ突出して高い → 2月は逆に低い

という不自然なシナリオが混入します。

---
James-Steinバイアス補正（0.6.0追加）

残差行列Rの各位相の平均（phase_mean）は、本来ゼロに近いはずです。Shapeが正確に推定できていれば残差の平均はゼロになるはずなので。

でも実際にはサンプリングノイズでゼロからズレます。このズレを無条件に信用するのは良くない。

James-Steinの posterior-mean 縮小：

補正後の残差 = R - phase_mean × noise_fraction

noise_fraction = se² / (phase_mean² + se²)

- `se`（標準誤差）= 各位相の残差の標準偏差 / √K（K = `_PHASE_NOISE_K` = 50）
- phase_mean が se より十分大きい → noise_fractionが小さい → ほぼ補正しない（信号として信頼）
- phase_mean が se と同程度 → noise_fractionが大きい → ゼロに引き寄せる（ノイズとみなす）

---
ホライズン適応デフレーション（0.6.0追加）

LWCPでLevelノイズが sqrt(1+h_test) で広がります。フェーズノイズはそのままだと二重に広がってしまう。

全体の分散 = Levelノイズ分散 × (1+h_test) + フェーズノイズ分散
                                ↑すでに広がっている

フェーズノイズ側を 1/sqrt(1+h_test) で縮小して帳尻を合わせます：

phase_deflate = 1.0 / sqrt(1.0 + h_test[step_idx])
phase_noise = phase_noise * phase_deflate

---
Levelノイズのブートストラップ（0.6.0追加）

Levelノイズ（①）も同じ発想です。0.5.0まではStudent-t分布（パラメトリック）を使っていましたが、0.6.0からLOOCV残差を直接リサンプリング
します：

// Student-t（旧）
ε ~ t(ν) × σ_loo × sqrt(1+h_test)

// Bootstrap（新）
ε = LOO残差からランダムに1つ選ぶ × sqrt(1+h_test)

実際の残差の歪み・尖度をそのまま保持できるのがメリット。正規分布に近くないデータ（スパイクが多いなど）で効果が出ます。

## 8. フォールバック・カスケード（3段階）

FLAIRは学習窓の情報だけで分岐を決定する（テストデータ不使用）。

### Branch 1（ランク1閉形式）― 通常パス

発動条件：BICが $\hat{P} \geq 2$ を選択 **かつ** $n_c \geq 3$ **かつ** DoFガード通過（$n_\text{train} \geq 2p$）

通常の FLAIR 完全版が動く。α → ∞ の極限では $\hat{L}(h) = L_{n_c}$（最終Levelで固定）となり、$\hat{y}_h = L_{n_c} \cdot S_{h \bmod P}$ = Seasonal Naïve と一致する。つまり正則化パスの右端が Seasonal Naïve であり、Ridgeはそこからの「改善」方向にしか動かない。

### Branch 2（生系列へのプレーンRidge）

発動条件（いずれか）：
- BICが $\hat{P} = 1$ を選択（周期性なし）
- $\hat{P} \geq 2$ だが $n_c < 3$
- DoFガードが発動（$n_\text{train} < 2p$）

$P = 1$ に設定。行列を $1 \times n$ にreshapeすると、Levelは $y$ そのもの、Shapeはスカラー1に縮退する。同じRidge式（切片・線形ドリフト・AR(1)・2次周期ラグ）を生系列に適用する。「周期性なし」「データ不足」「DoF不足」の3種のトリガーがすべて同一の予測挙動に収束する。

### Branch 3（最終値ガウシアン）

発動条件：Branch 2 の後さらに $n < 3$ の場合（Ridgeが解けないほど短い系列）

$\hat{y}_h = y_n$（最後の観測値をそのまま使用）、分散は直近 $K$ ラグ差分のスケールで決める。GIFTEvalベンチマークでは実際には発火しない。

---

**整数スナップ**は全3ブランチで統一適用（整数値系列の場合 $\tilde{Y} \leftarrow \text{round}(\tilde{Y})$）。

## データフロー

入力
y_raw:    Vec<f64>
freq:     &str
horizon:  usize
n_samples: usize
seed:     u64
  │
  ▼
[前処理(preprocess)]
y:       Vec<f64>
y_shift: f64
  │
  ▼
[周期選択(select_period)]
p:         usize
secondary: Option<usize>
svd_s:     Vec<f64>
nc_svd:    usize
  │
  ├──────────────────────────────┐
  ▼                              │
[行列折り畳み(reshape_matrix)]     │  svd_s, nc_svd は
mat: Vec<Vec<f64>>  // P × n_c   │  select_period から直接渡す
  │                              │
  ├─────────────┐                │
  ▼             ▼                │
[Level抽出(extract_level)]        │
l_raw: Vec<f64>                  │
                                 │
[Shape抽出(extract_shape)]        │
s_forecast: Vec<f64>             │
s_hist:     Vec<Vec<f64>>        │
  │                              │
  ▼                              │
[ノイズ縮小(optshrink)] ◄──────── ┘
l: Vec<f64>
  │
  ▼
[クロス周期除去(cross_period)]
l_work: Vec<f64>
s2:     Vec<f64>
  │
  ▼
[変換(box_cox)]
lam:     f64
l_innov: Vec<f64>   // センタリング済み変換空間
last_l:  f64        // NLinearセンタリングの基準値
  │
  ▼
[回帰(ridge_lsr1)]
beta:      Vec<f64>
loo_resid: Vec<f64>
h_avg:     Vec<f64>
  │
  ├─────────────────┐
  ▼                 ▼
[減衰推定(damped_trend)]  [幅計算(lwcp)]
phi: f64                  h_test: Vec<f64>  // 長さ horizon
  │                         │
  └────────┬─────────────────┘
            ▼
[Levelサンプリング(sample_level_paths)]
l_paths: Vec<Vec<f64>>  // n_samples × horizon
  │
  ▼
[逆変換(invert_box_cox)]
l_hat_all: Vec<Vec<f64>>  // n_samples × horizon
  │
  ├──────────────────────────────────┐
  ▼                                  ▼
[位相ノイズ生成(sample_phase_noise)]  （条件分岐）
phase_noise: Vec<Vec<f64>>  // n_samples × horizon
  │
  └──────────────┬───────────────────┘
                  ▼
[組み立て(assemble)]
raw_samples: Vec<Vec<f64>>  // n_samples × horizon
  │
  ▼
[後処理(clip_and_snap)]
samples: Vec<Vec<f64>>  // n_samples × horizon  ← 最終出力

---

## 付録：記号・略語一覧

### ギリシャ文字

| 記号 | 読み方 | 統計学での一般的な意味 | このドキュメントでの意味 |
|------|--------|----------------------|------------------------|
| α | アルファ | 正則化の強さ | Ridge回帰の正則化パラメータ。大きいほど係数を0に引き寄せる |
| β₀, β₁, β₂, β₃ | ベータ | 回帰係数 | Ridgeが学習する係数。β₂はLevelの自己回帰係数（ランダムウォークなら≈1） |
| δ₂ | デルタ | 差・乖離量 | LSR1変換後の変数。δ₂ = 1 − β₂。「β₂が1からどれだけズレているか」 |
| ε | イプシロン | 誤差項 / 微小な正数 | 回帰モデルの残差ノイズ。または数値的なゼロ除算防止用の微小値 |
| λ | ラムダ | パラメータ全般 | Box-Cox変換の累乗パラメータ。0〜1の値。0に近いほど対数変換に近い |
| ν | ニュー | 自由度 | Student-t分布の自由度（旧バージョンのLevelノイズモデルで使用） |
| ρ̂₁ | ロー（ハット付き） | 自己相関係数の推定値 | ΔLの1次自己相関の標本推定値。ダンプトレンドφの計算に使う |
| σ | シグマ | 標準偏差 / 特異値 | 文脈によって使い分け。**特異値として使う場合**（σ₁, σ₂, ...）は行列の「重要度の大きさ」を表す。**σ_noise** はノイズの標準偏差 |
| φ | ファイ | 任意のパラメータ | ダンプトレンドの減衰率。0〜1の値で、1に近いほどトレンドが長く続く |
| β（行列のアスペクト比） | ベータ | — | OptShrink専用。β = min(P,n) / max(P,n)。行列の縦横比 |
| μ_β | ミュー・ベータ | — | OptShrink専用。アスペクト比βに対応するMarchenko-Pastur分布の中央値 |

---

### 「ハット（^）」「アスタリスク（*）」「チルダ（~）」の意味

| 記法 | 意味 | 例 |
|------|------|-----|
| X̂（ハット） | Xの**推定値・予測値** | L̂ = 予測されたLevel、P* = 選ばれた最適周期 |
| X*（アスタリスク） | **最適な**X、または**事前分布の中心** | P* = BICで選んだ最良のP、β* = Ridgeの事前分布中心 |
| X̃（チルダ） | 中間処理後・仮の値 | S̃ = 正規化前のShape |
| X^(λ) | Box-Cox変換後のX | L^(λ) = Box-Cox変換後のLevel |
| ΔX | Xの1次差分（X[t] − X[t-1]） | ΔL = Levelの差分 |

---

### 変数（データ・行列）

| 変数名 | 正式名称（英語） | 日本語 | 意味 |
|--------|----------------|--------|------|
| y | observed values | 観測値 | 入力の時系列データそのもの |
| y_raw | raw observations | 生観測値 | 前処理前のデータ |
| y_shift | location shift | ポジティブシフト量 | 全値を正にするために足す定数 |
| y_n | last observation | 最終観測値 | Branch 3で予測値として使う |
| mat | reshaped matrix | 折り畳み行列 | y を P × n_c に整形した行列。行=位相、列=周期 |
| L | Level | レベル | 各周期の合計値の時系列。「大局的な水準」 |
| L_raw | raw Level | 生レベル | OptShrink補正前のLevel |
| L_bc | Box-Cox transformed Level | 変換後レベル | Box-Cox変換後のLevel |
| L_innov | Level innovations | レベル・イノベーション | NLinearセンタリング後のLevel。最終観測値を引いた差分 |
| L_hat / L̂ | predicted Level | レベル予測値 | Ridgeが予測した将来のLevel |
| L_paths | Level sample paths | レベル・サンプルパス | n_samples × m の行列。200通りのLevel将来シナリオ |
| S | Shape | シェイプ | 1周期内の配分パターン。長さPのベクトル。全要素の和=1 |
| S_forecast | forecast Shape | 予測用シェイプ | 将来予測に適用するfrozen shape（直近K=2周期の平均） |
| S_hist | historical Shape | 過去シェイプ | フェーズノイズ計算用の過去Shape履歴 |
| R | residual matrix | 残差行列 | R[位相][周期] = (実際 − 典型) / 典型。「シェイプからのズレ」の記録 |
| loo_resid | LOO residuals | LOO残差 | LWCP正規化済みの一個抜き交差検証残差 |
| svd_s | singular values | 特異値ベクトル | BICのSVDで計算した特異値列。OptShrinkで使い回す |

---

### 変数（スカラー・インデックス）

| 変数名 | 正式名称（英語） | 日本語 | 意味 |
|--------|----------------|--------|------|
| P | Period | 周期 | 1周期の長さ（hourlyなら24か168） |
| P* | optimal Period | 最適周期 | BICで選ばれたP |
| n | number of observations | 総観測数 | データ点の総数 |
| n_c | number of complete periods | 完全周期数 | n_c = floor(n/P)。上限500、下限3 |
| p | number of features | 特徴量数 | Ridge回帰の係数の数。通常3、2次周期ありなら4 |
| sec | secondary period | 2次周期 | 2番目の周期長。hourlyの週周期なら sec=7（Levelラグの参照先） |
| h_ii | leverage / hat value | レバレッジ | 「この訓練点を抜いたら予測がどれだけ変わるか」の指標。0〜1 |
| h_test[j] | test leverage | テストレバレッジ | ホライズンj番目の予測点のレバレッジ。遠いほど大きい |
| c | shrinkage factor | 縮小率 | OptShrinkがσ₁を補正する倍率。0〜1 |
| φ | damping factor | ダンプ係数 | トレンドの減衰率。ΔLの自己相関から推定 |
| k(P_c) | parameter count | パラメータ数 | BICの罰則項に使うモデルの自由度 |
| w | weights | 重み | Soft-Average GCVで各α候補に付けるsoftmax重み |

---

### 略語

| 略語 | 正式名称（英語） | 日本語訳 | このドキュメントでの使われ方 |
|------|----------------|---------|---------------------------|
| AR(1) | first-order AutoRegressive model | 1次自己回帰モデル | Ridge特徴量のラグ項（β₂ × L[t-1]）の形式 |
| BIC | Bayesian Information Criterion | ベイズ情報量基準 | 候補周期Pを選ぶスコア。小さいほど良い |
| CRPS | Continuous Ranked Probability Score | 連続ランク確率スコア | 確率予測の精度指標。分布全体の精度を測る |
| DoF | Degrees of Freedom | 自由度 | 統計モデルが「自由に動ける」パラメータの数 |
| FLAIR | Factored Level And Interleaved Ridge | — | このアルゴリズムの名称。Level×Shape因数分解 + Ridge の意 |
| GCV | Generalized Cross Validation | 汎化交差検証 | LOOCVの計算を閉形式で近似する手法。SVD1回で全α分を計算できる |
| LOO | Leave-One-Out | 一個抜き | 1点だけ除外して残りで学習・検証する方式 |
| LOOCV | Leave-One-Out Cross Validation | 一個抜き交差検証 | LOOをデータ全点で繰り返してモデルを評価する手法 |
| LWCP | Leverage-Weighted Conformal Prediction | レバレッジ重み付き共形予測 | ホライズンに応じて予測区間の幅を広げる仕組み |
| LSR1 | Level Shift with Random-walk prior, rank-1 | — | Ridgeの罰則をランダムウォーク事前分布に合わせる変換手法 |
| MASE | Mean Absolute Scaled Error | 平均絶対スケール誤差 | 予測精度の指標。Seasonal Naïveを1.0として相対評価 |
| MDL | Minimum Description Length | 最小記述長 | BICと同じ発想の情報理論的なモデル選択原理 |
| rel（接頭辞） | relative | 相対 | Seasonal Naïveの値を1.0としたときの相対値 |
| RSS | Residual Sum of Squares | 残差平方和 | 予測値と実測値のズレの二乗和。小さいほどよく当たっている |
| SVD | Singular Value Decomposition | 特異値分解 | 行列を「重要度の大きい順」の成分に分解する操作 |

---

### 実装定数

| 定数名 | 値 | 意味 |
|--------|-----|------|
| `_MIN_COMPLETE` | 3 | n_c の最小値。これを下回ると周期分解を諦めてBranch 2へ |
| `_MAX_COMPLETE` | 500 | n_c の上限。メモリ・速度ガード。長い系列は直近500周期のみ使う |
| `_MIN_POSITIVE_FOR_BC` | 10 | Box-Cox変換を適用するのに必要な正値観測数の最小値。未満ならλ=1（変換なし） |
| `_BC_EXP_CLIP` | 30 | 逆Box-Cox変換の指数のクリップ上限。極端な値でのオーバーフロー防止 |
| `_PHASE_NOISE_K` | 50 | フェーズノイズ計算に使う直近周期数のウィンドウ幅 |
| `_SHAPE_K` (K=2) | 2 | frozen shape計算に使う直近周期数 |
| `_N_ALPHAS` | 25 | Soft-Average GCVで試すα候補の数（10⁻⁴〜10⁴の対数等間隔） |

---

## SVD移植コードの検証手順（nalgebra → `src/svd.rs`）

`src/svd.rs` は nalgebra-0.35.0 の SVD（Householder 二重対角化 + implicit-shift QR）を
`Vec<f64>` にハードコード移植したもの。型を primitive 化（端部処理）する過程で
**アルゴリズムが原典から逸脱していないか**を以下の手順で検証する。移植・改修のたびに再実行する。

### 検証で実際に発見された逸脱（参考）

型を primitive 化すると、特に **Givens 回転の符号規約**が崩れやすい。過去の検証で以下4点を検出・修正した:

1. `assemble_u`/`assemble_vt` — 原典は反射ごとに `reflect_with_sign(sign=signum(diag))` で符号を織り込む。「反射後に列を一括符号反転」する独自処理は等価でない。
2. `Givens::rotate`（行ペア左作用）— 原典は `[c·a−s·b; s·a+c·b]`。`s` の符号配置を誤ると転置回転になる（`rotate_rows` とは符号が逆な点に注意）。
3. QRステップのインライン回転 — `subm` への適用式が #2 と同じ符号誤りを持つと特異値が壊れ、かつ収束しない（無限ループ）。
4. `cancel_y`/`cancel_x`/`svd_2x2_uptrig` — `cancel_*` の `c,s,r` 符号規約、および 2×2 SVD が `GivensRotation::new`（norm 値を乗算）でなく `cancel_y` を誤用していた。

**症状の特徴**: これらの逸脱があっても**特異値だけは正しく出る**ことがある（特異値は U・Vt の符号に不変なため）。
`svdvals`／`_period.py` 用途では露見せず、`U·diag(s)·Vt` を使う `_level.py:_ridge_sa` で初めて破綻する。
したがって**特異値の一致だけでは不十分**で、必ず再構成と直交性まで検証すること。

### 手順1: numpy 非依存の自己検証（一次スクリーニング）

numpy が無くても成立すべき数学的不変量を多数のランダム行列で確認する。

検証する不変量（`A` が m×n, m≥n のとき）:
- **再構成**: `A ≈ U·diag(s)·Vt`（最重要。U・Vt の整合性を一発で検出）
- **直交性**: `UᵀU ≈ I`、`Vt·Vtᵀ ≈ I`
- **特異値**: 降順かつ非負
- **API一致**: `svdvals(a)` が `svd(a)` の `s` と一致

```bash
# scratchpad に検証ドライバを置き、svd.rs を #[path] include してコンパイル・実行
rustc --edition 2024 verify_svd.rs -o verify_svd && ./verify_svd
# 期待: 全ケース recon/Uorth/Vorth ≤ 1e-9（スケール相対）、"ALL PASS"
```

対象行列は最低限: 既知の手計算ケース、単位行列、ランク落ち、重複特異値、
各種サイズ（3×2〜50×20）のランダム多数、桁違いスケール（camax スケーリングの検証）。

### 手順2: nalgebra 実物との突合（中間部品の一致確認）

別ディレクトリに `nalgebra = "0.35"` を依存に持つ使い捨てプロジェクトを作り、
同一入力で `Bidiagonal::{u, diagonal, off_diagonal, v_t}` と `SVD::new` の出力を出力させ、
移植版の中間結果（特に二重対角化の `diag`/`off`/U）と値比較する。

```bash
# ngref/ に nalgebra 依存プロジェクトを作って cargo run
# 確認ポイント:
#   - bidiag.diagonal()/off_diagonal() は abs（正）を返す（内部符号付き値とは別）
#   - A = U · B(abs) · Vt が成立する（B は abs の二重対角行列）
#   - 移植版 bidiagonalize の U 第1列が nalgebra の U と一致する
```

### 手順3: numpy/scipy との直接照合（最終確証）

`uv` で numpy/scipy を使い正解値を生成、移植版と突き合わせる。

```bash
# 正解値生成（PEP 723 インラインメタデータで依存解決）
uv run --quiet np_ref.py > np_ref.json   # np.linalg.svd(A, full_matrices=False) と scipy.linalg.svdvals
# 照合（np_ref.json を読んで svd.rs と比較する Rust ドライバ）
rustc --edition 2024 np_compare.rs -o np_compare && ./np_compare
# 期待: 特異値の相対誤差 ≤ 1e-9、再構成 ≤ 1e-9、"ALL MATCH numpy/scipy"
```

照合する量:
- 特異値 `s`（numpy `s` と一致、降順前提）
- `svdvals(a)`（scipy `svdvals` をソートした値と一致）
- 再構成 `U·diag(s)·Vt ≈ A`

> 注意: U・Vt 自体は符号・列順の自由度があり numpy と要素ごとには一致しない。
> 直接比較するのは**特異値**と**再構成 `U·diag(s)·Vt`**（自由度に不変な量）にすること。

テスト行列にはランク落ち・ゼロ列・近接特異値・大型（100×50 程度）を厚めに含める。

### 合格基準

手順1「ALL PASS」かつ手順3「ALL MATCH numpy/scipy」（相対誤差 ≤ 1e-9）。
手順2 は逸脱箇所を切り分けるための診断用（毎回必須ではないが、手順1/3 が落ちたら実施）。
