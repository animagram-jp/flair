# flair

This is a Rust implement of time series forecasting method FLAIR by Takato Honda.

## Version

| Version | Status    | Date      | Description |
|---------|-----------|-----------|-------------|
| 0.1.0   | Scheduled | 2026-4-12 | initial     |

## Provided Functions

**Common arguments**

- **`y: &[f64]`** — Observed values as a flat, equally-spaced 1-D array. No timestamps; the interval is given separately via `freq`. NaN is treated as 0.0.
- **`freq: &str`** — Observation interval: `"H"` (hourly), `"D"` (daily), `"W"` (weekly), `"M"` (monthly), `"Q"` (quarterly), `"A"` / `"Y"` (annual), etc.

| Mod     | Fn | Input | Output | Description |
|---------|----|-------|--------|-------------|
| `flair` | `confidence` | `y: &[f64]`, `freq: &str` | `Confidence` | Evaluates how well FLAIR's assumptions fit the input without running a forecast. Returns four fields: `rank1` (seasonal rank-1 strength), `gamma` (seasonal structure above random baseline), `gcv` (Ridge LOO error on the Level series), `impl_ok` (numerical sanity check). Use before forecasting to assess data suitability. |
|         | `forecast` | `y: &[f64]`, `horizon: usize`, `freq: &str`, `n_samples: usize`, `seed: Option<u64>` | `Result<Vec<Vec<f64>>>` `[n_samples][horizon]` | Generates Monte-Carlo sample paths. Each row is one forecast path. Use when the full uncertainty distribution is needed. |
|         | `forecast_mean` | `y: &[f64]`, `horizon: usize`, `freq: &str`, `n_samples: usize`, `seed: Option<u64>` | `Result<Vec<f64>>` `[horizon]` | Returns the mean over all sample paths as a single point forecast. The simplest option for a single-line prediction. |
|         | `forecast_quantiles` | `y: &[f64]`, `horizon: usize`, `freq: &str`, `n_samples: usize`, `seed: Option<u64>`, `quantiles: &[f64]` | `Result<Vec<Vec<f64>>>` `[quantile][horizon]` | Aggregates sample paths into quantiles. Pass e.g. `&[0.1, 0.5, 0.9]` to get pessimistic / median / optimistic forecast bands. |

## Performance

Measured on release build (`cargo build --release`), WSL2 / Linux x86-64.

| dataset | obs | binary size | peak RSS | wall time |
|---------|-----|-------------|----------|-----------|
| japan_demand (hourly) | 70,128 | 687 KB | 9.7 MB |   0.02 s |
| world_bank (annual)   | 34     | -      | 2.9 MB | < 0.01 s |

### Test

Run all checks: `cargo run --example integration_tests`

### confidence

`confidence(y, freq)` — self-evaluation from only input

| field | description |
|---|---|
| `rank1` | `s[0]²/Σs²` of seasonal matrix. 1.0 = pure rank-1 seasonality. `n/a` when period=1 (e.g. annual) or series too short — not an error |
| `gamma` | seasonal strength above random-matrix baseline, [0, 1]. 1.0 = strong clean seasonality |
| `gcv` | Ridge LOO error on Level series. lower = Level more predictable. scale depends on Box-Cox transform |
| `impl_ok` | Box-Cox round-trip and Ridge in-sample sanity check on synthetic data. `false` indicates a build or platform numerical issue |

**japan_demand** (hourly, strong seasonality):
```
  rank1   : 0.996   ← near-perfect rank-1 fit
  gamma   : 0.996   ← strong seasonal structure
  gcv     : 0.0056  ← Level highly predictable
  impl_ok : true
```

**world_bank** (annual, no intra-period structure):
```
  rank1   : n/a     ← period=1 by design; FLAIR runs Level-only AR
  gamma   : n/a
  gcv     : 41186   ← Level predictability in kWh/capita units
  impl_ok : true
```

### forecast

**japan_demand** — Tokyo hourly electricity demand, 70,128 obs (2016-04-01 – 2024-03-31)  
Source: [japanesepower.org](https://japanesepower.org/) — `examples/dataset/japan_demand.csv`  
Call: `forecast_mean(&y, 24, "H", 200, None)`

```
  +01h: 20332    +07h: 24814    +13h: 30050    +19h: 32240
  +02h: 19719    +08h: 27341    +14h: 30521    +20h: 31671
  +03h: 19795    +09h: 29881    +15h: 30029    +21h: 30852
  +04h: 20101    +10h: 31234    +16h: 29950    +22h: 29438
  +05h: 20628    +11h: 31240    +17h: 30677    +23h: 27694
  +06h: 22014    +12h: 31080    +18h: 31624    +24h: 25882
```

**world_bank** — Japan electric power consumption (kWh per capita), 34 annual obs (1990–2022)  
Source: [World Bank](https://data.worldbank.org/indicator/EG.USE.ELEC.KH.PC) — `examples/dataset/world_bank.csv`  
Call: `forecast_mean(&y, 3, "A", 200, None)`

```
  +1y: 7669 kWh/capita
  +2y: 7705 kWh/capita
  +3y: 7734 kWh/capita
```

### determinism

Same seed → bit-identical output. Different seeds → different output.  
`seed=None` → non-deterministic (seeded from system clock).

```
  [OK] determinism (same seed identical; seed=None non-deterministic)
```

## Reference

- https://github.com/TakatoHonda/FLAIR
- https://zenn.dev/t_honda/articles/flair-time-series-forecasting

## Lisence

```
Apache-2.0
Original: https://github.com/TakatoHonda/FLAIR
Changes: Ported from Python to Rust
Author: Andyou <andyou@animagram.jp>
```

## Original Texts (ja)

### 共通引数

- **`y: &[f64]`** — 時系列の観測値のみを等間隔で並べた1次元配列。日時情報は含まない。間隔は `freq` で別途指定する。NaN は 0.0 として扱われる。
- **`freq: &str`** — 観測間隔を表す文字列。`"H"`（時次）・`"D"`（日次）・`"W"`（週次）・`"M"`（月次）・`"Q"`（四半期）・`"A"` / `"Y"`（年次）など。

### 提供ポート

| Mod     | Fn | Input | Output | Description |
|---------|----|-------|--------|-------------|
| `flair` | `confidence` | `y: &[f64]`, `freq: &str` | `Confidence` | 予測を実行せずに入力データの適合度を評価する。`rank1`（季節性の強さ）・`gamma`（季節構造の純粋さ）・`gcv`（レベル系列の予測しやすさ）・`impl_ok`（数値実装の健全性）の4フィールドを返す。予測前のデータ確認用。 |
|         | `forecast` | `y: &[f64]`, `horizon: usize`, `freq: &str`, `n_samples: usize`, `seed: Option<u64>` | `Result<Vec<Vec<f64>>>` `[n_samples][horizon]` | モンテカルロサンプルパスを生成する。各行が1本の予測パス。不確実性の全分布が必要な場合に使う。 |
|         | `forecast_mean` | `y: &[f64]`, `horizon: usize`, `freq: &str`, `n_samples: usize`, `seed: Option<u64>` | `Result<Vec<f64>>` `[horizon]` | サンプルパスを平均した点予測を返す。最もシンプルな予測用途向け。 |
|         | `forecast_quantiles` | `y: &[f64]`, `horizon: usize`, `freq: &str`, `n_samples: usize`, `seed: Option<u64>`, `quantiles: &[f64]` | `Result<Vec<Vec<f64>>>` `[quantile][horizon]` | サンプルパスから指定パーセンタイルを集計する。`&[0.1, 0.5, 0.9]` を渡すと悲観・中央値・楽観の予測帯域を得られる。 |
