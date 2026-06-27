# flair

Wasm-compilable implement of time series forecasting algorithm FLAIR.


## Version

| Version | Status    | Date      | Description   |
|---------|-----------|-----------|---------------|
| 0.1.0   | Released  | 2026-04-09 | initial release |
| 0.2.0   | Released  | 2026-06-26 | follow 0.6.1* |
| 0.2.1   | Released  | 2026-06-27 | fix #25 |

This project adheres to [Semantic Versioning](https://semver.org/).

## Reference

- [*Mellon-Inc/FLAIR 0.6.1](https://github.com/Mellon-Inc/FLAIR)
- [FLAIR Algorithm Paperar: Xiv:2605.07222](https://arxiv.org/abs/2605.07222)
- [nalgebra 0.35.0](https://github.com/dimforge/nalgebra)
- [quadrature 0.1.2](https://github.com/Eh2406/quadrature)

## Dependency

- [libm](https://crates.io/crates/libm)
- [test only: nalgebra](https://crates.io/crates/nalgebra)

## License

```
Apache-2.0
Original: "FLAIR: Factored Level And Interleaved Ridge - single-equation time series forecasting"
  https://github.com/Mellon-Inc/FLAIR
  Copyright (c) Takato Honda
Changes: Reimplemented in Rust; linear algebra from scratch; adapted for WASM deployment
Author: Andyou <andyou@animagram.jp>
```

## Test

```sh
# unit tests
cargo test

# integration tests (crash-free + determinism on bundled datasets)
cargo run --example integration_tests --release

# forecast accuracy (80/20 train-test split, bundled datasets)
cargo run --example forecast_validation --release
```

---

[English](#reference) | [日本語](#ja)

## Quick use

```rust
use flair::{forecast_mean, Freq, NoiseMode};

let y: Vec<f64> = vec![/* observed values */];
let (mean_fc, conf) = forecast_mean(&y, &Freq::Monthly, 12, 200, 42, None, NoiseMode::Bootstrap).unwrap();
// mean_fc: Vec<f64> of length 12
// conf.rank1: seasonal signal strength (1.0 = strong seasonality)
// conf.gcv:   Ridge LOO error (lower = more predictable level)
```

With probabilistic intervals:

```rust
use flair::{forecast_quantiles, Freq, NoiseMode};

let (bands, _) = forecast_quantiles(&y, &Freq::Monthly, 12, 200, 42, None, NoiseMode::Bootstrap, &[0.1, 0.5, 0.9]).unwrap();
// bands[0]: 10th percentile (pessimistic)
// bands[1]: median
// bands[2]: 90th percentile (optimistic)
```

## Provided Functions

**Common arguments**

- **`y: &[f64]`** — Observed values as a flat, equally-spaced 1-D array. No timestamps; the interval is given separately via `freq`.
- **`freq: &Freq`** — Observation interval enum. Construct via `Freq::hourly(1)`, `Freq::Daily`, `Freq::Monthly`, etc.
- **`horizon: usize`** — Number of steps ahead to forecast.
- **`n_samples: usize`** — Number of Monte-Carlo sample paths (accuracy vs. speed trade-off).
- **`seed: u64`** — RNG seed for reproducibility. Pass `seed_from_time()` for non-deterministic output.
- **`covariates: Option<(&[f64], &[f64])>`** — Optional exogenous variables `(x_historical, x_future)` in row-major layout. `x_historical` length must be a multiple of `y.len()`; `x_future` length must equal `horizon * k` where `k` is the number of covariate columns.
- **`noise_mode: NoiseMode`** — Level noise sampling model. See [`NoiseMode`](#noisemode) below.

| Fn | Parameter | Output | Description |
|----|-----------|--------|-------------|
| `forecast` | `(y, freq, horizon, n_samples, seed, covariates, noise_mode)` | `Result<(Vec<Vec<f64>>, Confidence)>` `[n_samples][horizon]` | Generates Monte-Carlo sample paths. Each row is one forecast path. Use when the full uncertainty distribution is needed. |
| `forecast_mean` | `(y, freq, horizon, n_samples, seed, covariates, noise_mode)` | `Result<(Vec<f64>, Confidence)>` `[horizon]` | Returns the mean over all sample paths as a single point forecast. |
| `forecast_quantiles` | `(y, freq, horizon, n_samples, seed, covariates, noise_mode, quantiles: &[f64])` | `Result<(Vec<Vec<f64>>, Confidence)>` `[quantile][horizon]` | Aggregates sample paths into quantiles. Pass e.g. `&[0.1, 0.5, 0.9]` to get pessimistic / median / optimistic forecast bands. |
| `seed_from_time` *(std only)* | `()` | `u64` | Returns a non-deterministic seed from the system clock. |

### Freq

```rust
Freq::Secondly(10)
Freq::Minutely(5 | 10 | 15 | 30)
Freq::Hourly(1 | 12)
Freq::Daily
Freq::Weekly
Freq::Monthly
Freq::Quarterly
Freq::Yearly
```

Variants with an interval argument are constructed via fallible constructors (`Freq::hourly(n) -> Result<Freq, Error>`) that reject invalid values.

### NoiseMode

Controls how stochastic Level paths are sampled. Pass `NoiseMode::default()` for the default.

| Variant | Description |
|---------|-------------|
| `Bootstrap` *(default)* | Empirical resample of LOOCV residuals with replacement. Preserves the empirical skew and kurtosis of the predictive distribution. Automatically falls back to `StudentT` when fewer than 4 LOO residuals are available (very short series). |
| `StudentT` | Parametric Student-t with `ν = n_train − p` degrees of freedom. Applies post-hoc shrinkage toward the per-horizon median when `ν < 50` to remove heavy-tail variance inflation. |

```rust
use flair::NoiseMode;

NoiseMode::Bootstrap          // empirical bootstrap (default)
NoiseMode::StudentT           // parametric Student-t
NoiseMode::default()          // same as Bootstrap
```

### Confidence

| field | type | description |
|-------|------|-------------|
| `rank1` | `Option<f64>` | `s[0]²/Σs²` of seasonal matrix. 1.0 = pure rank-1 seasonality. `None` when period=1 (e.g. Yearly) or series too short. |
| `gcv` | `Option<f64>` | Ridge LOO error on Level series. Lower = Level more predictable. Scale depends on Box-Cox transform. |

## Dataset Test

### Result

80/20 train-test split. MASE < 1.0 means better than naive 1-step forecast.
Run: `cargo run --example forecast_validation --release`

| dataset | freq | obs | horizon | rank1 | MAE | RMSE | MAPE | MASE |
|---------|------|-----|---------|-------|-----|------|------|------|
| air_passengers | M | 144 | 12 | 0.999 | 11.50 | 16.48 | 2.66% | 0.55 |
| nottem | M | 240 | 12 | 0.998 | 1.66 | 2.19 | 3.86% | 0.38 |
| noaa_temp_monthly | M | 1,740 | 12 | 0.996 | 0.11 | 0.12 | 36.19% | 1.23 |
| sunspot_year | A | 289 | 10 | n/a | 35.28 | 39.31 | 95.01% | 2.19 |
| noaa_temp_annual | A | 145 | 10 | n/a | 0.18 | 0.21 | 33.42% | 2.17 |
| japan_demand_tokyo | H | 70,128 | 24 | 0.994 | 1856.45 | 2137.50 | 5.58% | 1.47 |
| elec_japan | A | 34 | 5 | n/a | 220.06 | 262.78 | 2.81% | 1.33 |
| elec_usa | A | 34 | 5 | n/a | 298.74 | 331.71 | 2.38% | 1.40 |
| elec_germany | A | 34 | 5 | n/a | 415.22 | 475.72 | 6.35% | 4.21 |
| elec_china | A | 34 | 5 | n/a | 109.65 | 146.13 | 2.10% | 0.77 |

rank1: `n/a` = annual series (period=1, no intra-period structure).

### Reference

| file | variable | freq | range | obs | source |
|------|----------|------|-------|-----|--------|
| `air_passengers.csv` | Monthly airline passengers | M | 1949–1960 | 144 | R built-in `AirPassengers`; originally Box & Jenkins (1976) *Time Series Analysis* |
| `nottem.csv` | Nottingham Castle mean air temperature (°F) | M | 1920–1939 | 240 | R built-in `nottem` |
| `sunspot_year.csv` | Yearly sunspot numbers | A | 1700–1988 | 289 | R built-in `sunspot.year`; source: WDC-SILSO, Royal Observatory of Belgium |
| `noaa_temp_annual.csv` | Global surface temperature anomaly (°C) | A | 1880–2024 | 145 | NOAA Global Surface Temperature (NOAAGlobalTemp) |
| `noaa_temp_monthly.csv` | Global surface temperature anomaly (°C) | M | 1880–2024 | 1,740 | NOAA Global Surface Temperature (NOAAGlobalTemp) |
| `japan_demand_tokyo.csv` ⚠️ | Tokyo electricity demand (MW) | H | 2016–2024 | 70,128 | [japanesepower.org](https://japanesepower.org/) — informational use only, not redistributed |
| `elec_per_capita.csv` | Electric power consumption (kWh per capita) — Japan, USA, Germany, China | A | 1990–2023 | 34 | World Bank WDI — EG.USE.ELEC.KH.PC |

⚠️ `japan_demand_tokyo.csv` is not included in this repository. To use it, download the demand CSV from [japanesepower.org](https://japanesepower.org/), extract the Tokyo column, and place it at `examples/dataset/japan_demand_tokyo.csv`.

## Build size

Measured on release build (`cargo build --release`), WSL2 / Linux x86-64.

| target | library size (rlib) |
|--------|---------------------|
| x86-64 | 783 KB |
| wasm32 | 513 KB |

## Python vs Rust comparison

To compare forecast output between flaircast (Python) and this crate (Rust),
run the same series with the same arguments in both and compare the mean forecast.

Python (requires `uv` and `flaircast` from PyPI):

```sh
uv run --with flaircast python3 - << 'EOF'
import math
from flaircast import forecast
import statistics

y = [100 + 1.5*i + 20*math.sin(2*math.pi*i/12) for i in range(144)]
samples = forecast(y, horizon=12, freq="M", n_samples=500, seed=0)
mean_fc = [statistics.mean(samples[:, h].tolist()) for h in range(12)]
print("Python mean forecast:", [round(v, 1) for v in mean_fc])
EOF
```

Rust (this crate, via `cargo test`):

The unit test `flair::tests::lwcp_vs_python_reference` in `src/flair.rs` runs
the same series (y = 100 + 1.5*t + 20*sin(2*pi*t/12), monthly, 144 points,
seed=0, n_samples=500) and asserts the mean forecast is within +-15 of the
flaircast 0.6.1 reference values recorded in the test.


---

## Ja

### クイックスタート

```rust
use flair::{forecast_mean, Freq, NoiseMode};

let y: Vec<f64> = vec![/* 観測値 */];
let (mean_fc, conf) = forecast_mean(&y, &Freq::Monthly, 12, 200, 42, None, NoiseMode::Bootstrap).unwrap();
// mean_fc: 長さ12のVec<f64>
// conf.rank1: 季節性の強さ（1.0 = 強い季節性）
// conf.gcv:   Ridge LOO誤差（低いほどLevelが予測しやすい）
```

区間予測を使う場合：

```rust
use flair::{forecast_quantiles, Freq, NoiseMode};

let (bands, _) = forecast_quantiles(&y, &Freq::Monthly, 12, 200, 42, None, NoiseMode::Bootstrap, &[0.1, 0.5, 0.9]).unwrap();
// bands[0]: 10パーセンタイル（悲観）
// bands[1]: 中央値
// bands[2]: 90パーセンタイル（楽観）
```

### 提供関数

**共通引数**

- **`y: &[f64]`** — 時系列の観測値のみを等間隔で並べた1次元配列。日時情報は含まない。間隔は `freq` で別途指定する。
- **`freq: &Freq`** — 観測間隔を表すenum。`Freq::hourly(1)`・`Freq::Daily`・`Freq::Monthly` などで構築する。
- **`horizon: usize`** — 何ステップ先まで予測するか。
- **`n_samples: usize`** — モンテカルロサンプルパス本数（精度と速度のトレードオフ）。
- **`seed: u64`** — 乱数シード（再現性のため）。非決定的な出力が必要な場合は `seed_from_time()` を渡す。
- **`covariates: Option<(&[f64], &[f64])>`** — 外生変数 `(x_historical, x_future)` をrow-majorで渡すオプション。`x_historical` の長さは `y.len()` の倍数、`x_future` の長さは `horizon * k`（k=列数）でなければならない。
- **`noise_mode: NoiseMode`** — Level のノイズサンプリングモデル。下記 [`NoiseMode`](#noisemode-1) 参照。

| 関数名 | 引数 | 戻り値 | 説明 |
|----|-----------|--------|-------------|
| `forecast` | `(y, freq, horizon, n_samples, seed, covariates, noise_mode)` | `Result<(Vec<Vec<f64>>, Confidence)>` `[n_samples][horizon]` | モンテカルロサンプルパスを生成する。各行が1本の予測パス。不確実性の全分布が必要な場合に使う。 |
| `forecast_mean` | `(y, freq, horizon, n_samples, seed, covariates, noise_mode)` | `Result<(Vec<f64>, Confidence)>` `[horizon]` | サンプルパスを平均した点予測を返す。最もシンプルな予測用途向け。 |
| `forecast_quantiles` | `(y, freq, horizon, n_samples, seed, covariates, noise_mode, quantiles: &[f64])` | `Result<(Vec<Vec<f64>>, Confidence)>` `[quantile][horizon]` | サンプルパスから指定パーセンタイルを集計する。`&[0.1, 0.5, 0.9]` を渡すと悲観・中央値・楽観の予測帯域を得られる。 |
| `seed_from_time` *（std のみ）* | `()` | `u64` | システム時刻からシードを生成する。再現性が不要な場合に各予測関数へ渡す。 |

### Freq

```rust
Freq::Secondly(10)
Freq::Minutely(5 | 10 | 15 | 30)
Freq::Hourly(1 | 12)
Freq::Daily
Freq::Weekly
Freq::Monthly
Freq::Quarterly
Freq::Yearly
```

間隔引数を持つバリアントはフォールブルコンストラクタ（`Freq::hourly(n) -> Result<Freq, Error>`）で構築する。不正な値はエラーになる。

### NoiseMode

Levelの確率パスをサンプリングする方式を指定する。`NoiseMode::default()` でデフォルト値を使える。

| バリアント | 説明 |
|-----------|------|
| `Bootstrap` *（デフォルト）* | LOOCVの残差プールから復元抽出（bootstrap）する。実データの歪み・尖度を保持する。残差が4件未満（非常に短い系列）の場合は自動的に `StudentT` にフォールバックする。 |
| `StudentT` | 自由度 `ν = n_train − p` のStudent-t分布からサンプリングする。`ν < 50` のとき、ホライズンごとの中央値に向けた事後収縮（post-hoc shrinkage）を適用し、重い裾の分散膨張を除去する。 |

```rust
use flair::NoiseMode;

NoiseMode::Bootstrap          // 経験的bootstrap（デフォルト）
NoiseMode::StudentT           // パラメトリックStudent-t
NoiseMode::default()          // Bootstrap と同じ
```

### 予測信頼性 (Confidence)

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `rank1` | `Option<f64>` | 季節行列の `s[0]²/Σs²`。1.0 = 完全な単一周期性。`None` は period=1（Yearly 等）または系列が短い場合（エラーではない）。 |
| `gcv` | `Option<f64>` | RidgeのLOOCV最小誤差。低いほどLevelが予測しやすい。スケールはBox-Cox変換依存。 |

### データセットテスト

#### 結果

80/20 トレイン・テスト分割。MASE < 1.0 はナイーブ1ステップ予測より優れていることを意味する。
実行: `cargo run --example forecast_validation --release`

| データセット | 頻度 | 観測数 | horizon | rank1 | MAE | RMSE | MAPE | MASE |
|------------|------|--------|---------|-------|-----|------|------|------|
| air_passengers | 月次 | 144 | 12 | 0.999 | 11.50 | 16.48 | 2.66% | 0.55 |
| nottem | 月次 | 240 | 12 | 0.998 | 1.66 | 2.19 | 3.86% | 0.38 |
| noaa_temp_monthly | 月次 | 1,740 | 12 | 0.996 | 0.11 | 0.12 | 36.19% | 1.23 |
| sunspot_year | 年次 | 289 | 10 | n/a | 35.28 | 39.31 | 95.01% | 2.19 |
| noaa_temp_annual | 年次 | 145 | 10 | n/a | 0.18 | 0.21 | 33.42% | 2.17 |
| japan_demand_tokyo | 時次 | 70,128 | 24 | 0.994 | 1856.45 | 2137.50 | 5.58% | 1.47 |
| elec_japan | 年次 | 34 | 5 | n/a | 220.06 | 262.78 | 2.81% | 1.33 |
| elec_usa | 年次 | 34 | 5 | n/a | 298.74 | 331.71 | 2.38% | 1.40 |
| elec_germany | 年次 | 34 | 5 | n/a | 415.22 | 475.72 | 6.35% | 4.21 |
| elec_china | 年次 | 34 | 5 | n/a | 109.65 | 146.13 | 2.10% | 0.77 |

rank1: `n/a` = 年次系列（period=1、周期内構造なし）。

#### データセット出典

| ファイル | 変数 | 頻度 | 期間 | 観測数 | 出典 |
|---------|------|------|------|--------|------|
| `air_passengers.csv` | 月次航空旅客数 | 月次 | 1949–1960 | 144 | R組み込み `AirPassengers`；Box & Jenkins (1976) *Time Series Analysis* |
| `nottem.csv` | ノッティンガム城平均気温（°F） | 月次 | 1920–1939 | 240 | R組み込み `nottem` |
| `sunspot_year.csv` | 年次太陽黒点数 | 年次 | 1700–1988 | 289 | R組み込み `sunspot.year`；WDC-SILSO、ベルギー王立天文台 |
| `noaa_temp_annual.csv` | 全球地表気温偏差（°C） | 年次 | 1880–2024 | 145 | NOAA Global Surface Temperature (NOAAGlobalTemp) |
| `noaa_temp_monthly.csv` | 全球地表気温偏差（°C） | 月次 | 1880–2024 | 1,740 | NOAA Global Surface Temperature (NOAAGlobalTemp) |
| `japan_demand_tokyo.csv` ⚠️ | 東京電力需要（MW） | 時次 | 2016–2024 | 70,128 | [japanesepower.org](https://japanesepower.org/) — 情報目的のみ、再配布不可 |
| `elec_per_capita.csv` | 一人当たり電力消費量（kWh）— 日本・米国・ドイツ・中国 | 年次 | 1990–2023 | 34 | World Bank WDI — EG.USE.ELEC.KH.PC |

⚠️ `japan_demand_tokyo.csv` はリポジトリに含まれていない。使用する場合は [japanesepower.org](https://japanesepower.org/) から需要CSVをダウンロードし、東京列を抽出して `examples/dataset/japan_demand_tokyo.csv` に配置する。

### ビルドサイズ

リリースビルド（`cargo build --release`）での測定値。WSL2 / Linux x86-64。

| ターゲット | ライブラリサイズ（rlib） |
|-----------|----------------------|
| x86-64 | 783 KB |
| wasm32 | 513 KB |
