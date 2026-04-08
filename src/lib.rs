// #![no_std]
extern crate alloc;

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
