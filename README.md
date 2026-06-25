# flair

Wasm-compilable implement of time series forecasting algorithm FLAIR.

## Version

| Version | Status    | Date      | Description   |
|---------|-----------|-----------|---------------|
| 0.1.0   | Released  | 2026-4-09 | initial       |
| 0.2.0   | Scheduled | 2026-6-31 | follow 0.6.1* |

This project adheres to [Semantic Versioning](https://semver.org/).

## Reference

- [*FLAIR](https://github.com/Mellon-Inc/FLAIR)
- [Algorithm Paper](https://arxiv.org/abs/2605.07222)
- [LAPACK](https://www.netlib.org/lapack/explore-html)
- [quadrature](https://crates.io/crates/quadrature)

---

[![日本語](https://img.shields.io/badge/言語-日本語-red)](#ja)

## Provided Functions

**Common arguments**

- **`y: &[f64]`** — Observed values as a flat, equally-spaced 1-D array. No timestamps; the interval is given separately via `freq`.
- **`freq: &Freq`** — Observation interval enum. Construct via `Freq::hourly(1)`, `Freq::Daily`, `Freq::Monthly`, etc.
- **`horizon: usize`** — Number of steps ahead to forecast.
- **`n_samples: usize`** — Number of Monte-Carlo sample paths (accuracy vs. speed trade-off).
- **`seed: u64`** — RNG seed for reproducibility. Pass `seed_from_time()` for non-deterministic output.
- **`covariates: Option<(&[f64], &[f64])>`** — Optional exogenous variables `(x_historical, x_future)` in row-major layout. `x_historical` length must be a multiple of `y.len()`; `x_future` length must equal `horizon * k` where `k` is the number of covariate columns.

| Fn | Input | Output | Description |
|----|-------|--------|-------------|
| `forecast` | `y`, `freq`, `horizon`, `n_samples`, `seed`, `covariates` | `Result<(Vec<Vec<f64>>, Confidence)>` `[n_samples][horizon]` | Generates Monte-Carlo sample paths. Each row is one forecast path. Use when the full uncertainty distribution is needed. |
| `forecast_mean` | `y`, `freq`, `horizon`, `n_samples`, `seed`, `covariates` | `Result<(Vec<f64>, Confidence)>` `[horizon]` | Returns the mean over all sample paths as a single point forecast. |
| `forecast_quantiles` | `y`, `freq`, `horizon`, `n_samples`, `seed`, `covariates`, `quantiles: &[f64]` | `Result<(Vec<Vec<f64>>, Confidence)>` `[quantile][horizon]` | Aggregates sample paths into quantiles. Pass e.g. `&[0.1, 0.5, 0.9]` to get pessimistic / median / optimistic forecast bands. |
| `seed_from_time` *(std only)* | — | `u64` | Returns a non-deterministic seed from the system clock. |

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

### Confidence

| field | type | description |
|-------|------|-------------|
| `rank1` | `Option<f64>` | `s[0]²/Σs²` of seasonal matrix. 1.0 = pure rank-1 seasonality. `None` when period=1 (e.g. Yearly) or series too short. |
| `gamma` | `Option<f64>` | Seasonal strength above random-matrix baseline, [0, 1]. 1.0 = strong clean seasonality. |
| `gcv` | `Option<f64>` | Ridge LOO error on Level series. Lower = Level more predictable. Scale depends on Box-Cox transform. |

## Dataset Test

### Result

80/20 train-test split. MASE < 1.0 means better than naive 1-step forecast.
Run: `cargo run --example forecast_validation --release`

| dataset | freq | obs | horizon | rank1 | gamma | MAE | RMSE | MAPE | MASE |
|---------|------|-----|---------|-------|-------|-----|------|------|------|
| air_passengers | M | 144 | 12 | — | — | 16.83 | 20.31 | 4.21% | 0.80 |
| nottem | M | 240 | 12 | — | — | 1.49 | 1.93 | 3.49% | 0.34 |
| noaa_temp_monthly | M | 1,740 | 12 | — | — | 0.07 | 0.08 | 21.35% | 0.76 |
| sunspot_year | A | 289 | 10 | — | — | 36.49 | 41.00 | 96.32% | 2.27 |
| noaa_temp_annual | A | 145 | 10 | — | — | 0.24 | 0.26 | 43.20% | 2.79 |
| japan_demand_tokyo | H | 70,128 | 24 | 0.996 | 0.996 | 1736.82 | 2083.10 | 4.99% | 1.38 |
| elec_japan | A | 34 | 5 | n/a | n/a | 238.75 | 283.17 | 3.05% | 1.44 |
| elec_usa | A | 34 | 5 | n/a | n/a | 331.93 | 374.81 | 2.65% | 1.56 |
| elec_germany | A | 34 | 5 | n/a | n/a | 437.62 | 501.66 | 6.69% | 4.44 |
| elec_china | A | 34 | 5 | n/a | n/a | 466.77 | 549.63 | 8.75% | 3.26 |

rank1/gamma: `n/a` = annual series (period=1, no intra-period structure); `—` = not computed for this run.

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

## Dependency

- [libm](https://crates.io/crates/libm)
- [dev only: nalgebra](https://crates.io/crates/nalgebra)

## License

```
Apache-2.0
Original: "FLAIR: Factored Level And Interleaved Ridge - single-equation time series forecasting"
  https://github.com/Mellon-Inc/FLAIR
  Copyright (c) Takato Honda
Changes: Reimplemented in Rust; linear algebra from scratch; adapted for WASM deployment
Author: Andyou <andyou@animagram.jp>
```

## Ja

### 共通引数

- **`y: &[f64]`** — 時系列の観測値のみを等間隔で並べた1次元配列。日時情報は含まない。間隔は `freq` で別途指定する。
- **`freq: &Freq`** — 観測間隔を表すenum。`Freq::hourly(1)`・`Freq::Daily`・`Freq::Monthly` などで構築する。
- **`horizon: usize`** — 何ステップ先まで予測するか。
- **`n_samples: usize`** — モンテカルロサンプルパス本数（精度と速度のトレードオフ）。
- **`seed: u64`** — 乱数シード（再現性のため）。非決定的な出力が必要な場合は `seed_from_time()` を渡す。
- **`covariates: Option<(&[f64], &[f64])>`** — 外生変数 `(x_historical, x_future)` をrow-majorで渡すオプション。`x_historical` の長さは `y.len()` の倍数、`x_future` の長さは `horizon * k`（k=列数）でなければならない。

### 提供ポート

| Fn | Input | Output | Description |
|----|-------|--------|-------------|
| `forecast` | `y`, `freq`, `horizon`, `n_samples`, `seed`, `covariates` | `Result<(Vec<Vec<f64>>, Confidence)>` `[n_samples][horizon]` | モンテカルロサンプルパスを生成する。各行が1本の予測パス。|
| `forecast_mean` | `y`, `freq`, `horizon`, `n_samples`, `seed`, `covariates` | `Result<(Vec<f64>, Confidence)>` `[horizon]` | サンプルパスを平均した点予測を返す。最もシンプルな予測用途向け。 |
| `forecast_quantiles` | `y`, `freq`, `horizon`, `n_samples`, `seed`, `covariates`, `quantiles: &[f64]` | `Result<(Vec<Vec<f64>>, Confidence)>` `[quantile][horizon]` | サンプルパスから指定パーセンタイルを集計する。`&[0.1, 0.5, 0.9]` を渡すと悲観・中央値・楽観の予測帯域を得られる。 |
| `seed_from_time` *（std のみ）* | — | `u64` | システム時刻からシードを生成する。再現性が不要な場合に各予測関数へ渡す。 |

### 予測信頼性 (Confidence)

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `rank1` | `Option<f64>` | 季節行列の `s[0]²/Σs²`。1.0 = 完全な単一周期性。`None` は period=1（Yearly 等）または系列が短い場合（エラーではない）。 |
| `gamma` | `Option<f64>` | ランダム行列ベースライン除去後の季節強度。[0, 1]。1.0 = 強い季節性。 |
| `gcv` | `Option<f64>` | RidgeのLOOCV最小誤差。低いほどLevelが予測しやすい。スケールはBox-Cox変換依存。 |
