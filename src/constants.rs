// SPDX-License-Identifier: Apache-2.0
//
// Ported from FLAIR (flaircast) (Copyright 2026 Takato Honda, Apache-2.0).
// Source: flaircast/_constants.py
// See the bundled NOTICE for details.

/// General-purpose division guard.
pub const EPS: f64 = 1e-10;
/// Floor for Box-Cox input; prevents log of a non-positive number.
pub const EPS_BOXCOX: f64 = 1e-8;
/// Floor inside log() to avoid -inf in BIC calculations.
pub const EPS_LOG: f64 = 1e-30;
/// Softmax weight threshold below which a candidate is skipped in LOOCV soft-average.
pub const EPS_WEIGHT: f64 = 1e-15;
/// Floor for Shape proportions; keeps the multiplicative decomposition away from divide-by-zero.
pub const EPS_SHAPE: f64 = 1e-6;

/// Clip range for exp() in the inverse Box-Cox path (lam = 0).
pub const BC_EXP_CLIP: f64 = 30.0;
/// Minimum positive observations required to estimate Box-Cox lambda.
pub const MIN_POSITIVE_FOR_BC: usize = 10;

/// Minimum complete periods required for the Level x Shape decomposition; below this, P = 1.
pub const MIN_COMPLETE: usize = 3;
/// Cap on complete periods (memory and speed guard).
pub const MAX_COMPLETE: usize = 500;

/// When true, the Level Ridge fits delta(L_innov) — random-walk prior on beta_2.
pub const DIFF_TARGET: bool = true;

/// Number of recent periods used for Shape estimation.
pub const SHAPE_K: usize = 2;

/// Number of recent periods used for the phase-noise residual matrix.
pub const PHASE_NOISE_K: usize = 50;

/// Number of alpha candidates in the Ridge LOOCV soft-average grid.
pub const N_ALPHAS: usize = 25;
/// log10 of the minimum Ridge alpha.
pub const ALPHA_LOG_MIN: f64 = -4.0;
/// log10 of the maximum Ridge alpha.
pub const ALPHA_LOG_MAX: f64 = 4.0;
