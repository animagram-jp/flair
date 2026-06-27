# flair

Wasm-compilable implement of time series forecasting algorithm FLAIR.

## Version

| Version | Status    | Date      | Description   |
|---------|-----------|-----------|---------------|
| 0.1.0   | Released  | 2026-04-09 | initial      |
| 0.2.0   | Released  | 2026-06-26 | follow 0.6.1* |

This project adheres to [Semantic Versioning](https://semver.org/).

## Reference

- [*Mellon-Inc/FLAIR 0.6.1](https://github.com/Mellon-Inc/FLAIR)
- [FLAIR Algorithm Paperar: Xiv:2605.07222](https://arxiv.org/abs/2605.07222)
- [nalgebra 0.35.0](https://github.com/dimforge/nalgebra)
- [quadrature 0.1.2](https://github.com/Eh2406/quadrature)

---

[日本語版はこちら](#ja)

## Quick use

```rust
use flair::{forecast_mean, Freq};

let y: Vec<f64> = vec![/* observed values */];
let (mean_fc, conf) = forecast_mean(&y, &Freq::Monthly, 12, 200, 42, None).unwrap();
// mean_fc: Vec<f64> of length 12
// conf.rank1: seasonal signal strength (1.0 = strong seasonality)
// conf.gcv:   Ridge LOO error (lower = more predictable level)
```

## Provided Functions

**Common arguments**

- **`y: &[f64]`** — Observed values as a flat, equally-spaced 1-D array. No timestamps; the interval is given separately via `freq`.
- **`freq: &Freq`** — Observation interval enum. Construct via `Freq::hourly(1)`, `Freq::Daily`, `Freq::Monthly`, etc.
- **`horizon: usize`** — Number of steps ahead to forecast.
- **`n_samples: usize`** — Number of Monte-Carlo sample paths (accuracy vs. speed trade-off).
- **`seed: u64`** — RNG seed for reproducibility. Pass `seed_from_time()` for non-deterministic output.
- **`covariates: Option<(&[f64], &[f64])>`** — Optional exogenous variables `(x_historical, x_future)` in row-major layout. `x_historical` length must be a multiple of `y.len()`; `x_future` length must equal `horizon * k` where `k` is the number of covariate columns.

| Fn | Parameter | Output | Description |
|----|-----------|--------|-------------|
| `forecast` | `(y, freq, horizon, n_samples, seed, covariates)` | `Result<(Vec<Vec<f64>>, Confidence)>` `[n_samples][horizon]` | Generates Monte-Carlo sample paths. Each row is one forecast path. Use when the full uncertainty distribution is needed. |
| `forecast_mean` | `(y, freq, horizon, n_samples, seed, covariates)` | `Result<(Vec<f64>, Confidence)>` `[horizon]` | Returns the mean over all sample paths as a single point forecast. |
| `forecast_quantiles` | `(y, freq, horizon, n_samples, seed, covariates, quantiles: &[f64])` | `Result<(Vec<Vec<f64>>, Confidence)>` `[quantile][horizon]` | Aggregates sample paths into quantiles. Pass e.g. `&[0.1, 0.5, 0.9]` to get pessimistic / median / optimistic forecast bands. |
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
| nottem | M | 240 | 12 | 0.998 | 1.66 | 2.19 | 3.87% | 0.38 |
| noaa_temp_monthly | M | 1,740 | 12 | 0.996 | 0.11 | 0.12 | 36.26% | 1.23 |
| sunspot_year | A | 289 | 10 | n/a | 35.28 | 39.31 | 95.01% | 2.19 |
| noaa_temp_annual | A | 145 | 10 | n/a | 0.18 | 0.21 | 33.42% | 2.17 |
| japan_demand_tokyo | H | 70,128 | 24 | 0.995 | 1856.47 | 2137.52 | 5.58% | 1.47 |
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

| 関数名 | 引数 | 戻り値 | 説明 |
|----|-----------|--------|-------------|
| `forecast` | `(y, freq, horizon, n_samples, seed, covariates)` | `Result<(Vec<Vec<f64>>, Confidence)>` `[n_samples][horizon]` | モンテカルロサンプルパスを生成する。各行が1本の予測パス。|
| `forecast_mean` | `(y, freq, horizon, n_samples, seed, covariates)` | `Result<(Vec<f64>, Confidence)>` `[horizon]` | サンプルパスを平均した点予測を返す。最もシンプルな予測用途向け。 |
| `forecast_quantiles` | `(y, freq, horizon, n_samples, seed, covariates, quantiles: &[f64])` | `Result<(Vec<Vec<f64>>, Confidence)>` `[quantile][horizon]` | サンプルパスから指定パーセンタイルを集計する。`&[0.1, 0.5, 0.9]` を渡すと悲観・中央値・楽観の予測帯域を得られる。 |
| `seed_from_time` *（std のみ）* | `()` | `u64` | システム時刻からシードを生成する。再現性が不要な場合に各予測関数へ渡す。 |

### 予測信頼性 (Confidence)

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `rank1` | `Option<f64>` | 季節行列の `s[0]²/Σs²`。1.0 = 完全な単一周期性。`None` は period=1（Yearly 等）または系列が短い場合（エラーではない）。 |
| `gcv` | `Option<f64>` | RidgeのLOOCV最小誤差。低いほどLevelが予測しやすい。スケールはBox-Cox変換依存。 |
