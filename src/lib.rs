#![no_std]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

use alloc::vec::Vec;
use core::{
    fmt::{self, Display},
    result::Result,
};

#[cfg(feature = "std")]
use std::time::{SystemTime, UNIX_EPOCH};

pub mod flair;
pub mod svd;

pub use flair::{forecast, forecast_mean, forecast_quantiles};

/// Returns a non-deterministic seed derived from the system clock.
/// use like: `let forcast = flair::forecast(&y, 12, "M", 100, flair::seed_from_time());`
#[cfg(feature = "std")]
pub fn seed_from_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64 ^ d.as_secs().wrapping_mul(0x9e3779b97f4a7c15))
        .unwrap_or(0xdeadbeefcafe1234)
}

// ============================================================
// Trait
// ============================================================

pub trait Flair {
    fn forecast(
            y: &[f64], // 時系列データ
            frequency: &Freq, // データの周波数
            horizon: usize, // 何ステップ先まで予測するか
            n_samples: usize, //  サンプルパス本数
            seed: u64, // 乱数シード。`flair::seed_from_time()`を代入可能
            covariates: Option<(&[f64], &[f64])>, // (x_historical, x_future) // 
        ) -> Result<(Vec<Vec<f64>>, Confidence), Error>;

    fn forecast_mean(
        y: &[f64],
        frequency: &Freq,
        horizon: usize,
        n_samples: usize,
        seed: u64,
        covariates: Option<(&[f64], &[f64])>,
    ) -> Result<(Vec<f64>, Confidence), Error>;

    fn forecast_quantiles(
        y: &[f64],
        frequency: &Freq,
        horizon: usize,
        n_samples: usize,
        seed: u64,
        covariates: Option<(&[f64], &[f64])>,
        quantiles: &[f64], // 各パーセンタイル（0.0〜1.0)
    ) -> Result<(Vec<Vec<f64>>, Confidence), Error>;
}

pub enum Freq {
    Secondly(usize),  // 10
    Minutely(usize),  // 5, 10, 15, 30
    Hourly(usize),    // 1, 12
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Yearly,
}

pub struct Confidence {
    pub rank1: Option<f64>,  // σ₁²/Σσᵢ² — rank-1構造の強さ（1.0=完全な単一周期性、~1/P=フラット）
    pub gcv:   Option<f64>,  // RidgeのLOOCV最小誤差 — Levelの予測可能性。低いほど良い
    // rank1がNoneになるケース:
    //   - Yearly（P=1、季節分解なし）
    //   - 系列が短くてMIN_COMPLETE周期に満たない
}

// ============================================================
// Error
// ============================================================

#[derive(Debug, Clone)]
pub enum Error {
    InvalidFreq(usize),         // Freq::new() に無効な間隔が渡された
    InvalidInput(&'static str), // y が空など
    Svd(SvdError),
}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::InvalidFreq(n) => write!(f, "invalid frequency: {}", n),
            Error::InvalidInput(msg) => write!(f, "invalid input: {}", msg),
            Error::Svd(e) => write!(f, "svd error: {}", e),
        }
    }
}

#[derive(Debug, Clone)]
pub enum SvdError {
    DimensionMismatch,
    ConvergenceFailed,
}

impl Display for SvdError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SvdError::DimensionMismatch => write!(f, "dimension mismatch"),
            SvdError::ConvergenceFailed => write!(f, "convergence failed"),
        }
    }
}