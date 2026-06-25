// SPDX-License-Identifier: Apache-2.0
//
// このファイルは FLAIR (flaircast) (Copyright 2026 Takato Honda, Apache-2.0)
// の flaircast/_constants.py を Rust へ移植したものです。詳細は同梱の NOTICE を参照。

/// FLAIR数値定数。
///
/// 出典: flaircast/_constants.py (Apache-2.0)
/// https://github.com/Mellon-Inc/FLAIR/blob/main/flaircast/_constants.py

/// 汎用ゼロ除算ガード
pub const EPS: f64 = 1e-10;
/// Box-Cox入力のフロア (log(非正) を防ぐ)
pub const EPS_BOXCOX: f64 = 1e-8;
/// BIC計算の log() フロア
pub const EPS_LOG: f64 = 1e-30;
/// LOOCV soft-average でスキップするsoftmax重みの閾値
pub const EPS_WEIGHT: f64 = 1e-15;
/// Shape比率のフロア (乗法分解のゼロ除算防止)
pub const EPS_SHAPE: f64 = 1e-6;

/// 逆Box-Cox (lam=0) の exp() クリップ上限
pub const BC_EXP_CLIP: f64 = 30.0;
/// Box-Cox lambda推定に必要な正値観測の最小数
pub const MIN_POSITIVE_FOR_BC: usize = 10;

/// 周期分解に必要な完全周期の最小数 (未満でP=1フォールバック)
pub const MIN_COMPLETE: usize = 3;
/// 完全周期の上限 (メモリ・速度ガード)
pub const MAX_COMPLETE: usize = 500;

/// true のとき Level Ridge は ΔL_innov を予測 (ランダムウォーク事前分布)
pub const DIFF_TARGET: bool = true;

/// Shape推定に使う直近周期数
pub const SHAPE_K: usize = 2;

/// フェーズノイズ残差行列に使う直近周期数
pub const PHASE_NOISE_K: usize = 50;

/// Ridge LOOCV soft-average のα候補数
pub const N_ALPHAS: usize = 25;
/// αグリッドの log10 下限
pub const ALPHA_LOG_MIN: f64 = -4.0;
/// αグリッドの log10 上限
pub const ALPHA_LOG_MAX: f64 = 4.0;
