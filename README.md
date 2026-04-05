# flair

This is a Rust implement OSS of time series forecasting method FLAIR by Takato Honda.

## lisence

Apache-2.0
Original: https://github.com/TakatoHonda/FLAIR
Changes: Ported from Python to Rust
Author: Andyou <andyou@animagram.jp>

## todo

- 既存に開発者本人がapache-2.0 licenseで公開中の時系列解析FLAIRを、Rustにてメモリ合理性とポータビリティを考慮して再実装する
  - Rust組み込み関数使用可。SVDも一時的に使用可
  - 入力のみを利用した、成果物の自己検証pub fnも必要。信頼性評価を出力する
  - 実体を網羅実装後、docTestとunit testを整備
- 動作検証と静動パフォーマンスメモリ実測を行う

## performance

Measured on release build (`cargo build --release`), WSL2 / Linux x86-64.

| dataset | obs | binary size | peak RSS | wall time |
|---|---|---|---|---|
| japan_demand (hourly) | 70,128 | 687 KB | 9.7 MB | 0.02 s |
| world_bank (annual) | 34 | 688 KB | 2.9 MB | < 0.01 s |

## test

Run all checks: `cargo run --example integration_tests`

### confidence

`confidence(y, freq)` — self-evaluation from input only, no forecast needed.

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
Source: [japanesepower.org](https://japanesepower.org/) — `examples/test_data/japan_electricity_demand.csv`  
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
Source: [World Bank](https://data.worldbank.org/indicator/EG.USE.ELEC.KH.PC) — `examples/test_data/world_bank_electricity/`  
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

## reference

- https://github.com/TakatoHonda/FLAIR
- https://zenn.dev/t_honda/articles/flair-time-series-forecasting
