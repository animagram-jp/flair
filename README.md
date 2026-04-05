# flair

This is a Rust implement OSS of FLAIR by Takato Honda.
Original: https://github.com/TakatoHonda/FLAIR
License: Apache 2.0
Changes: Ported from Python to Rust
Author: Andyou <andyou@animagram.jp>

## todo

- 既存に開発者本人がapache-2.0 licenseで公開中の時系列解析FLAIRを、Rustにてメモリ合理性とポータビリティを考慮して再実装する
  - Rust組み込み関数使用可。SVDも一時的に使用可
  - 入力のみを利用した、成果物の自己検証pub fnも必要。信頼性評価を出力する
  - 実体を網羅実装後、docTestとunit testを整備
- 動作検証と静動パフォーマンスメモリ実測を行う

## reference

- https://github.com/TakatoHonda/FLAIR
- https://zenn.dev/t_honda/articles/flair-time-series-forecasting