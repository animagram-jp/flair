# flair

This is a Rust implement OSS of FLAIR by Takato Honda.

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

| example | binary size | peak RSS | wall time |
|---|---|---|---|
| japan_demand_forecast (70,128 obs → 24 h) | 687 KB | 9.7 MB | 0.02 s |
| world_bank_forecast (34 obs → 3 y) | 688 KB | 2.9 MB | < 0.01 s |

## test

### japan_demand_forecast

Input: Tokyo hourly electricity demand, 70,128 observations (2016-04-01 – 2024-03-31)  
Source: [japanesepower.org](https://japanesepower.org/) — `examples/test_data/japan_electricity_demand.csv`  
Call: `forecast_mean(&y, 24, "H", 200, Some(42))`

```
+01h: 20422    +07h: 24987    +13h: 29896    +19h: 32194
+02h: 19813    +08h: 27493    +14h: 30342    +20h: 31635
+03h: 19889    +09h: 29993    +15h: 29858    +21h: 30835
+04h: 20186    +10h: 31305    +16h: 29757    +22h: 29422
+05h: 20724    +11h: 31245    +17h: 30523    +23h: 27676
+06h: 22147    +12h: 31019    +18h: 31549    +24h: 25853
```

### world_bank_forecast

Input: Japan electric power consumption (kWh per capita), 34 annual observations (1990–2022)  
Source: [World Bank](https://data.worldbank.org/indicator/EG.USE.ELEC.KH.PC) — `examples/test_data/world_bank_electricity/`  
Call: `forecast_mean(&y, 3, "A", 200, Some(42))`

```
+1y: 7684 kWh/capita
+2y: 7709 kWh/capita
+3y: 7763 kWh/capita
```

## reference

- https://github.com/TakatoHonda/FLAIR
- https://zenn.dev/t_honda/articles/flair-time-series-forecasting
