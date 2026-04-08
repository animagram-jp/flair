#![no_std]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

use core::{
    fmt,
    result
};

pub mod flair;
pub mod svd;

pub use flair::{
    confidence, 
    forecast, 
    forecast_mean, 
    forecast_quantiles, 
    Confidence
};

/// Error type for SVD operations
#[derive(Debug, Clone)]
pub enum SvdError {
    DimensionMismatch,
    ConvergenceFailed,
    InvalidInput(&'static str),
}

impl fmt::Display for SvdError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SvdError::DimensionMismatch => write!(f, "dimension mismatch"),
            SvdError::ConvergenceFailed => write!(f, "convergence failed"),
            SvdError::InvalidInput(msg) => write!(f, "invalid input: {}", msg),
        }
    }
}

/// Error type for flair operations
#[derive(Debug, Clone)]
pub enum Error {
    Svd(SvdError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::Svd(e) => write!(f, "svd error: {}", e),
        }
    }
}

pub type Result<T> = result::Result<T, Error>;

/// Returns a non-deterministic seed derived from the system clock.
///
/// Only available with the `std` feature (enabled by default).
/// Pass the result directly to `forecast`, `forecast_mean`, or `forecast_quantiles`.
///
/// ```rust,no_run
/// let y = vec![1.0f64; 24];
/// let fc = flair::forecast_mean(&y, 12, "M", 100, flair::seed_from_time());
/// ```
#[cfg(feature = "std")]
pub fn seed_from_time() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64 ^ d.as_secs().wrapping_mul(0x9e3779b97f4a7c15))
        .unwrap_or(0xdeadbeefcafe1234)
}
