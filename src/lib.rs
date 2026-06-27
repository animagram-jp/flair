#![no_std]
extern crate core;
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

use core::{
    fmt::{self, Display},
    result::Result,
};
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::time::{SystemTime, UNIX_EPOCH};

pub mod constants;
pub mod double_exponential;
pub mod flair;
pub mod optshrink;
pub mod svd;

pub use flair::{
    forecast, 
    forecast_mean, 
    forecast_quantiles
};

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
// Freq (required as arguments)
// ============================================================

pub enum Freq {
    Secondly(usize),  // 10
    Minutely(usize),  // 5, 10, 15, 30
    Hourly(usize),    // 1, 2, 12
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Yearly,
}

impl Freq {
    pub fn secondly(n: usize) -> Result<Self, Error> {
        match n {
            10 => Ok(Freq::Secondly(n)),
            _ => Err(Error::InvalidFreq(n)),
        }
    }
    pub fn minutely(n: usize) -> Result<Self, Error> {
        match n {
            5 | 10 | 15 | 30 => Ok(Freq::Minutely(n)),
            _ => Err(Error::InvalidFreq(n)),
        }
    }
    pub fn hourly(n: usize) -> Result<Self, Error> {
        match n {
            1 | 2 | 12 => Ok(Freq::Hourly(n)),
            _ => Err(Error::InvalidFreq(n)),
        }
    }
}

// ============================================================
// Confidence (returned alongside forecasts)
// ============================================================

pub struct Confidence {
    /// sigma_1^2 / sum(sigma_i^2): rank-1 signal strength (1.0 = single period, ~1/P = flat).
    /// None for Yearly (P=1) or series too short for the Level x Shape decomposition.
    pub rank1: Option<f64>,
    /// Minimum Ridge LOOCV error; lower is more predictable.
    pub gcv:   Option<f64>,
}

// ============================================================
// Error (this crate can provide)
// ============================================================

#[derive(Debug, Clone)]
pub enum Error {
    InvalidFreq(usize),
    InvalidInput(&'static str),
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