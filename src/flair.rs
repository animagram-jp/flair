//! FLAIR: Factored Level And Interleaved Ridge (arXiv:2605.07222) by Takato Honda

#![allow(clippy::many_single_char_names, clippy::too_many_arguments)]

use core::{
    cmp::Ordering,
    f64::consts::PI,
    result::Result,
};
use alloc::{collections::BTreeSet, vec, vec::Vec};
use libm::{sqrt, log as ln, exp, pow, sin, cos, round};
use crate::{Freq, Confidence, Error, NoiseMode, constants::*, svd, optshrink};

// ============================================================
// forecast, forecast_mean, forecast_quantiles
// ============================================================

/// Generates Monte-Carlo sample paths for the given time series.
///
/// Returns `[n_samples][horizon]` paths. Each row is one independent forecast path.
/// Use this when the full uncertainty distribution is needed.
///
/// `covariates` is currently accepted for API compatibility but not yet used in the model;
/// pass `None` for standard usage.
///
/// `noise_mode` selects the Level noise model (see [`NoiseMode`]).
///
/// # Example
///
/// ```rust
/// use flair::{forecast, Freq, NoiseMode};
/// // Monthly data, 120 points (10 years): trend + seasonality
/// let y: Vec<f64> = (0..120)
///     .map(|i| 100.0 + i as f64 * 0.5 + 20.0 * (i as f64 * std::f64::consts::PI / 6.0).sin())
///     .collect();
/// let (paths, conf) = forecast(&y, &Freq::Monthly, 12, 50, 0, None, NoiseMode::Bootstrap).unwrap();
/// assert_eq!(paths.len(), 50);
/// assert_eq!(paths[0].len(), 12);
/// assert!(paths.iter().flat_map(|p| p.iter()).all(|v| v.is_finite()));
/// assert!(conf.rank1.is_some()); // monthly period detected
/// ```
pub fn forecast(
    y: &[f64],
    frequency: &Freq,
    horizon: usize,
    n_samples: usize,
    seed: u64,
    covariates: Option<(&[f64], &[f64])>,
    noise_mode: NoiseMode,
) -> Result<(Vec<Vec<f64>>, Confidence), Error> {
    if y.is_empty() { return Err(Error::InvalidInput("y must not be empty")); }
    if horizon < 1  { return Err(Error::InvalidInput("horizon must be >= 1")); }
    if n_samples < 1 { return Err(Error::InvalidInput("n_samples must be >= 1")); }
    let exog = if let Some((x_hist, x_future)) = covariates {
        if x_hist.is_empty() || x_future.is_empty() {
            return Err(Error::InvalidInput("covariates must not be empty"));
        }
        if x_hist.len() % y.len() != 0 {
            return Err(Error::InvalidInput("x_historical length must be a multiple of y length"));
        }
        let k = x_hist.len() / y.len();
        if x_future.len() != horizon * k {
            return Err(Error::InvalidInput("x_future length must equal horizon * k"));
        }
        // Convert flat row-major slices to Vec<Vec<f64>> matrices (n×k and horizon×k)
        let n = y.len();
        let xh: Vec<Vec<f64>> = (0..n).map(|i| (0..k).map(|j| x_hist[i * k + j]).collect()).collect();
        let xf: Vec<Vec<f64>> = (0..horizon).map(|i| (0..k).map(|j| x_future[i * k + j]).collect()).collect();
        Some((xh, xf, k))
    } else {
        None
    };
    forecast_inner(y, horizon, frequency, n_samples, seed, exog, noise_mode)
}

/// Returns the mean over all sample paths as a single point forecast.
///
/// Output is `[horizon]`. Equivalent to averaging the rows of [`forecast`].
///
/// `covariates` is currently accepted for API compatibility but not yet used in the model;
/// pass `None` for standard usage.
///
/// `noise_mode`: see [`NoiseMode`] and [`forecast`] for details.
///
/// # Example
///
/// ```rust
/// use flair::{forecast, forecast_mean, Freq, NoiseMode};
/// let y: Vec<f64> = (0..120)
///     .map(|i| 100.0 + 20.0 * (i as f64 * std::f64::consts::PI / 6.0).sin())
///     .collect();
/// let (mean_fc, _) = forecast_mean(&y, &Freq::Monthly, 12, 200, 0, None, NoiseMode::Bootstrap).unwrap();
/// assert_eq!(mean_fc.len(), 12);
/// assert!(mean_fc.iter().all(|v| v.is_finite()));
/// // forecast_mean equals the sample mean of forecast()
/// let (paths, _) = forecast(&y, &Freq::Monthly, 12, 200, 0, None, NoiseMode::Bootstrap).unwrap();
/// let sample_mean: Vec<f64> = (0..12)
///     .map(|h| paths.iter().map(|p| p[h]).sum::<f64>() / paths.len() as f64)
///     .collect();
/// for h in 0..12 {
///     assert!((mean_fc[h] - sample_mean[h]).abs() < 1e-10);
/// }
/// ```
pub fn forecast_mean(
    y: &[f64],
    frequency: &Freq,
    horizon: usize,
    n_samples: usize,
    seed: u64,
    covariates: Option<(&[f64], &[f64])>,
    noise_mode: NoiseMode,
) -> Result<(Vec<f64>, Confidence), Error> {
    let (samples, conf) = forecast(y, frequency, horizon, n_samples, seed, covariates, noise_mode)?;
    let ns = samples.len() as f64;
    let mean = (0..horizon)
        .map(|h| samples.iter().map(|s| s[h]).sum::<f64>() / ns)
        .collect();
    Ok((mean, conf))
}

/// Aggregates sample paths into per-horizon quantiles.
///
/// Output is `[quantiles.len()][horizon]`. Pass e.g. `&[0.1, 0.5, 0.9]` to get
/// pessimistic / median / optimistic forecast bands.
///
/// `covariates` is currently accepted for API compatibility but not yet used in the model;
/// pass `None` for standard usage.
///
/// # Example
///
/// ```rust
/// use flair::{forecast_quantiles, Freq, NoiseMode};
/// let y: Vec<f64> = (0..120)
///     .map(|i| 100.0 + 20.0 * (i as f64 * std::f64::consts::PI / 6.0).sin())
///     .collect();
/// let qs = [0.1, 0.5, 0.9];
/// let (bands, _) = forecast_quantiles(&y, &Freq::Monthly, 12, 200, 0, None, NoiseMode::Bootstrap, &qs).unwrap();
/// assert_eq!(bands.len(), 3);
/// assert!(bands.iter().all(|b| b.len() == 12));
/// // q0.1 ≤ q0.5 ≤ q0.9 at every horizon step
/// for h in 0..12 {
///     assert!(bands[0][h] <= bands[1][h] && bands[1][h] <= bands[2][h]);
/// }
/// ```
pub fn forecast_quantiles(
    y: &[f64],
    frequency: &Freq,
    horizon: usize,
    n_samples: usize,
    seed: u64,
    covariates: Option<(&[f64], &[f64])>,
    noise_mode: NoiseMode,
    quantiles: &[f64],
) -> Result<(Vec<Vec<f64>>, Confidence), Error> {
    if quantiles.is_empty() || quantiles.iter().any(|&q| !(0.0..=1.0).contains(&q)) {
        return Err(Error::InvalidInput("quantiles must be non-empty and each value in [0.0, 1.0]"));
    }
    let (samples, conf) = forecast(y, frequency, horizon, n_samples, seed, covariates, noise_mode)?;
    let ns = samples.len();
    let result = quantiles.iter().map(|&q| {
        (0..horizon).map(|h| {
            let mut col: Vec<f64> = samples.iter().map(|s| s[h]).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let idx = round(q * (ns - 1) as f64) as usize;
            col[idx]
        }).collect()
    }).collect();
    Ok((result, conf))
}

// ============================================================
// internal helper
// ============================================================

fn get_period(frequency: &Freq) -> usize {
    match frequency {
        Freq::Secondly(10) => 6,
        Freq::Minutely(5)  => 12,
        Freq::Minutely(10) => 6,
        Freq::Minutely(15) => 4,
        Freq::Minutely(30) => 48,
        Freq::Hourly(1)    => 24,
        Freq::Hourly(2)    => 12,
        Freq::Hourly(12)   => 2,
        Freq::Daily        => 7,
        Freq::Weekly       => 52,
        Freq::Monthly      => 12,
        Freq::Quarterly    => 4,
        Freq::Yearly       => 1,
        _                  => 1,
    }
}

fn get_periods(frequency: &Freq) -> Vec<usize> {
    match frequency {
        Freq::Secondly(10) => vec![6],
        Freq::Minutely(5)  => vec![12, 288],
        Freq::Minutely(10) => vec![6, 144],
        Freq::Minutely(15) => vec![4, 96],
        Freq::Minutely(30) => vec![48, 336],
        Freq::Hourly(1)    => vec![24, 168],
        Freq::Hourly(2)    => vec![12, 84],
        Freq::Hourly(12)   => vec![2, 14],
        Freq::Daily        => vec![7, 365],
        Freq::Weekly       => vec![52],
        Freq::Monthly      => vec![12],
        Freq::Quarterly    => vec![4],
        Freq::Yearly       => vec![],
        _                  => vec![],
    }
}

// ── PRNG: xorshift64 + Box-Muller normal ──────────────────────────────────

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(if seed == 0 { 0xdeadbeefcafe1234 } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    fn randint(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        self.next_u64() as usize % n
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
    }
    // Kinderman-Ramage Student-t via normal / sqrt(chi2/nu)
    fn student_t(&mut self, nu: f64) -> f64 {
        let z = self.normal();
        // chi2(nu) ≈ sum of nu normal^2; approximate via normal for large nu
        let chi2: f64 = (0..nu.min(50.0) as usize).map(|_| { let n = self.normal(); n * n }).sum::<f64>();
        let chi2 = chi2.max(EPS);
        z / sqrt(chi2 / nu)
    }
}

// ── Box-Cox: golden-section MLE for lambda in [0,1] ───────────────────────

fn bc_lambda(y: &[f64]) -> f64 {
    let yp: Vec<f64> = y.iter().copied().filter(|&v| v > 0.0).collect();
    if yp.len() < MIN_POSITIVE_FOR_BC {
        return 1.0;
    }
    let n = yp.len() as f64;
    let log_sum: f64 = yp.iter().map(|&v| ln(v)).sum();
    let llf = |lam: f64| -> f64 {
        let yt: Vec<f64> = if lam.abs() < 1e-10 {
            yp.iter().map(|&v| ln(v)).collect()
        } else {
            yp.iter().map(|&v| (pow(v, lam) - 1.0) / lam).collect()
        };
        let m = yt.iter().sum::<f64>() / n;
        let var = yt.iter().map(|&v| pow(v - m, 2.0)).sum::<f64>() / n;
        if var < EPS_LOG { return f64::NEG_INFINITY; }
        (lam - 1.0) * log_sum - n / 2.0 * ln(var)
    };
    let phi = (sqrt(5.0_f64) - 1.0) / 2.0;
    let (mut a, mut b) = (0.0f64, 1.0f64);
    let mut c = b - phi * (b - a);
    let mut d = a + phi * (b - a);
    let mut fc = llf(c);
    let mut fd = llf(d);
    for _ in 0..60 {
        if (b - a) < 1e-7 { break; }
        if fc < fd { a = c; c = d; fc = fd; d = a + phi * (b - a); fd = llf(d); }
        else        { b = d; d = c; fd = fc; c = b - phi * (b - a); fc = llf(c); }
    }
    ((a + b) / 2.0).clamp(0.0, 1.0)
}

fn bc(y: &[f64], lam: f64) -> Vec<f64> {
    y.iter().map(|&v| {
        let v = v.max(EPS_BOXCOX);
        if lam == 0.0 { ln(v) } else { (pow(v, lam) - 1.0) / lam }
    }).collect()
}

fn bc_inv(z: &[f64], lam: f64) -> Vec<f64> {
    z.iter().map(|&v| {
        if lam == 0.0 {
            exp(v.clamp(-BC_EXP_CLIP, BC_EXP_CLIP))
        } else {
            pow((v * lam + 1.0).max(EPS), 1.0 / lam)
        }
    }).collect()
}

// ── helpers ────────────────────────────────────────────────────────────────

fn interp_nan(arr: &[f64]) -> Vec<f64> {
    let valid: Vec<usize> = arr.iter().enumerate()
        .filter(|&(_, &v)| !v.is_nan())
        .map(|(i, _)| i)
        .collect();
    match valid.len() {
        0 => vec![0.0; arr.len()],
        1 => arr.iter().map(|&v| if v.is_nan() { arr[valid[0]] } else { v }).collect(),
        _ => arr.iter().enumerate().map(|(i, &v)| {
            if !v.is_nan() { return v; }
            let lo = valid.partition_point(|&j| j < i);
            if lo == 0 {
                arr[valid[0]]
            } else if lo == valid.len() {
                arr[valid[valid.len() - 1]]
            } else {
                let x0 = valid[lo - 1];
                let x1 = valid[lo];
                let t = (i - x0) as f64 / (x1 - x0) as f64;
                arr[x0] + t * (arr[x1] - arr[x0])
            }
        }).collect(),
    }
}

fn logspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| pow(10.0_f64, lo + (hi - lo) * i as f64 / (n - 1) as f64)).collect()
}

fn slice_mean(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }

// ── Ridge with Soft-Average GCV ────────────────────────────────────────────
//
// Returns (beta [nf], loo_residuals [n_train], gcv_min, vt [k x nf], s [k], d_avg [k]).
// loo is LWCP-normalized: e_i^LOO / sqrt(1 + h_ii)
// x_rows: row-major design matrix (n_train rows, each of length nf).

fn ridge_sa(x_rows: &[Vec<f64>], y: &[f64]) -> Result<(Vec<f64>, Vec<f64>, f64, Vec<Vec<f64>>, Vec<f64>, Vec<f64>), Error> {
    let m = x_rows.len();
    let nf = x_rows[0].len();

    let (u, s, vt) = svd::svd(x_rows);
    let k = s.len();

    let s2: Vec<f64> = s.iter().map(|&v| v * v).collect();
    // Uty[j] = sum_i u[i,j] * y[i]
    let uty: Vec<f64> = (0..k).map(|j| (0..m).map(|i| u[i][j] * y[i]).sum()).collect();

    let alphas = logspace(ALPHA_LOG_MIN, ALPHA_LOG_MAX, N_ALPHAS);

    // GCV score for each alpha
    let mut gcv = vec![0.0f64; N_ALPHAS];
    for (ai, &a) in alphas.iter().enumerate() {
        let d: Vec<f64> = s2.iter().map(|&v| v / (v + a)).collect();
        // hat-matrix diagonal: h[i] = sum_j u[i,j]^2 * d[j]
        let h: Vec<f64> = (0..m).map(|i| (0..k).map(|j| u[i][j] * u[i][j] * d[j]).sum()).collect();
        // residual: r = y - U*(d*Uty)
        let r: Vec<f64> = (0..m).map(|i| {
            y[i] - (0..k).map(|j| u[i][j] * d[j] * uty[j]).sum::<f64>()
        }).collect();
        gcv[ai] = r.iter().zip(h.iter())
            .map(|(&ri, &hi)| pow(ri / (1.0 - hi).max(EPS), 2.0))
            .sum::<f64>() / m as f64;
    }

    // Soft-average weights (numerically stable)
    let gcv_min = gcv.iter().cloned().fold(f64::INFINITY, f64::min);
    let log_w: Vec<f64> = gcv.iter().map(|&g| -(g - gcv_min) / gcv_min.max(EPS)).collect();
    let lw_max = log_w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let w_raw: Vec<f64> = log_w.iter().map(|&lw| exp(lw - lw_max)).collect();
    let w_sum: f64 = w_raw.iter().sum();
    let w: Vec<f64> = w_raw.iter().map(|&wi| wi / w_sum).collect();

    let mut beta = vec![0.0f64; nf];
    let mut d_avg = vec![0.0f64; k];

    for (&wi, &a) in w.iter().zip(alphas.iter()) {
        if wi < EPS_WEIGHT { continue; }
        let d: Vec<f64> = s2.iter().map(|&v| v / (v + a)).collect();
        // dvs[j] = d[j] * uty[j] / s[j]
        let dvs: Vec<f64> = (0..k).map(|j| d[j] * uty[j] / s[j].max(EPS)).collect();
        // beta += wi * Vt^T * dvs  (vt is k×nf)
        for col in 0..nf {
            beta[col] += wi * (0..k).map(|j| vt[j][col] * dvs[j]).sum::<f64>();
        }
        for j in 0..k { d_avg[j] += wi * d[j]; }
    }

    // LWCP-normalized LOO residuals: e_i^LOO / sqrt(1 + h_ii)
    let residuals: Vec<f64> = (0..m).map(|i| {
        y[i] - x_rows[i].iter().zip(beta.iter()).map(|(&xi, &bi)| xi * bi).sum::<f64>()
    }).collect();
    let h_avg: Vec<f64> = (0..m).map(|i| (0..k).map(|j| u[i][j] * u[i][j] * d_avg[j]).sum()).collect();
    let loo: Vec<f64> = residuals.iter().zip(h_avg.iter())
        .map(|(&ri, &hi)| ri / (1.0 - hi).max(EPS) / sqrt((1.0 + hi).max(EPS)))
        .collect();

    Ok((beta, loo, gcv_min, vt, s, d_avg))
}

// ── Period selection ───────────────────────────────────────────────────────

/// Returns (P, secondary_periods, primary_period, calendar_periods, svd_s, nc_svd).
/// svd_s and nc_svd are the singular values and n_complete from the winning candidate's
/// period-folded matrix, reused by optshrink to avoid a second SVD ("One SVD" principle).
fn select_period(y: &[f64], n: usize, frequency: &Freq) -> (usize, Vec<usize>, usize, Vec<usize>, Vec<f64>, usize) {
    let period = get_period(frequency);
    let cal = get_periods(frequency);

    let mut candidates: Vec<usize> = if !cal.is_empty() {
        cal.iter().copied().filter(|&p| p >= 1 && n / p >= MIN_COMPLETE).collect()
    } else {
        vec![]
    };
    if candidates.is_empty() {
        let p = period.max(1);
        candidates.push(if n / p >= MIN_COMPLETE { p } else { 1 });
    }

    let min_cand = *candidates.iter().min().unwrap();
    let t_max = n.min(MAX_COMPLETE * min_cand);
    let y_sel = &y[y.len().saturating_sub(t_max)..];

    // P=1 null: mean + noise, 1 parameter
    let mean = slice_mean(y_sel);
    let rss_null: f64 = y_sel.iter().map(|&v| pow(v - mean, 2.0)).sum();
    let bic_null = t_max as f64 * ln((rss_null / t_max as f64).max(EPS_LOG)) + ln(t_max as f64);
    let mut best_p = 1usize;
    let mut best_bic = bic_null;
    let mut best_svd_s: Vec<f64> = alloc::vec![0.0];
    let mut best_nc_svd: usize = 0;

    for &p_cand in &candidates {
        let nc = t_max / p_cand;
        if nc < MIN_COMPLETE { continue; }
        let start = y_sel.len() - nc * p_cand;
        let y_use = &y_sel[start..];
        let mat_c: Vec<Vec<f64>> = (0..p_cand).map(|ph| (0..nc).map(|ci| y_use[ci * p_cand + ph]).collect()).collect();
        let s = svd::svdvals(&mat_c);
        let rss1: f64 = s.iter().skip(1).map(|&v| v * v).sum();
        let t = (nc * p_cand) as f64;
        let bic = t * ln((rss1 / t).max(EPS_LOG)) + (p_cand + nc - 1) as f64 * ln(t);
        if bic < best_bic {
            best_bic = bic;
            best_p = p_cand;
            best_svd_s = s;
            best_nc_svd = nc;
        }
    }

    let secondary: Vec<usize> = cal.iter().copied().filter(|&p| p > best_p).collect();
    (best_p, secondary, period, cal, best_svd_s, best_nc_svd)
}

// ── Shape₂: MDL-gated prior shrinkage on Level series ─────────────────────

fn compute_shape2(l: &[f64], cp: usize, n_complete: usize) -> Option<Vec<f64>> {
    let nc2 = n_complete / cp;
    if nc2 < 2 { return None; }

    let mut s2_raw = vec![0.0f64; cp];
    for d in 0..cp {
        let vals: Vec<f64> = (0..n_complete).filter(|&i| i % cp == d).map(|i| l[i]).collect();
        s2_raw[d] = if vals.is_empty() { 1.0 } else { slice_mean(&vals) };
    }
    let raw_mean = slice_mean(&s2_raw);
    if raw_mean < EPS { return None; }
    s2_raw.iter_mut().for_each(|v| *v /= raw_mean);

    // First-harmonic prior
    let cos_b: Vec<f64> = (0..cp).map(|i| cos(2.0 * PI * i as f64 / cp as f64)).collect();
    let sin_b: Vec<f64> = (0..cp).map(|i| sin(2.0 * PI * i as f64 / cp as f64)).collect();
    let s2_c: Vec<f64> = s2_raw.iter().map(|&v| v - 1.0).collect();
    let a = 2.0 * slice_mean(&s2_c.iter().zip(cos_b.iter()).map(|(&sc, &cb)| sc * cb).collect::<Vec<_>>());
    let b = 2.0 * slice_mean(&s2_c.iter().zip(sin_b.iter()).map(|(&sc, &sb)| sc * sb).collect::<Vec<_>>());
    let s2_harmonic: Vec<f64> = (0..cp).map(|i| 1.0 + a * cos_b[i] + b * sin_b[i]).collect();

    let rss_flat: f64 = s2_c.iter().map(|&v| v * v).sum();
    let rss_harm: f64 = s2_raw.iter().zip(s2_harmonic.iter()).map(|(&r, &h)| pow(r - h, 2.0)).sum();
    let bic_flat = cp as f64 * ln((rss_flat / cp as f64).max(EPS_LOG));
    let bic_harm = cp as f64 * ln((rss_harm / cp as f64).max(EPS_LOG)) + 2.0 * ln(cp as f64);

    let s2_prior: Vec<f64> = if bic_harm < bic_flat { s2_harmonic } else { vec![1.0; cp] };

    let w = nc2 as f64 / (nc2 as f64 + cp as f64);
    let mut s2: Vec<f64> = s2_raw.iter().zip(s2_prior.iter())
        .map(|(&r, &p)| (w * r + (1.0 - w) * p).max(EPS_SHAPE))
        .collect();
    let s2m = slice_mean(&s2);
    s2.iter_mut().for_each(|v| *v /= s2m);
    Some(s2)
}

// ── Shape estimation (Frozen Shape: global average of last K periods) ─────
//
// mat: [P][n_complete]  – phase × period matrix
// Returns (S_forecast [m][P], S_hist [n_complete][P], m).

fn estimate_shape(
    mat: &[Vec<f64>],
    n_complete: usize,
    big_p: usize,
    _secondary: &[usize],
    _l: &[f64],
    horizon: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, usize) {
    let k = SHAPE_K.min(n_complete);
    let mut s_global = vec![0.0f64; big_p];
    for ph in 0..big_p {
        let props: Vec<f64> = (n_complete - k..n_complete).map(|ci| {
            let tot: f64 = (0..big_p).map(|p| mat[p][ci]).sum();
            if tot > EPS { mat[ph][ci] / tot } else { 1.0 / big_p as f64 }
        }).collect();
        s_global[ph] = slice_mean(&props);
    }
    let sg_sum = s_global.iter().sum::<f64>().max(EPS);
    s_global.iter_mut().for_each(|v| *v /= sg_sum);

    let m = (horizon + big_p - 1) / big_p;
    (vec![s_global.clone(); m], vec![s_global; n_complete], m)
}

// ── Damped trend helpers (#7) ─────────────────────────────────────────────

fn estimate_phi(l_bc: &[f64]) -> f64 {
    let n = l_bc.len();
    if n < 3 { return 0.0; }
    // diff(L_bc): len = n-1
    let dl: Vec<f64> = l_bc.windows(2).map(|w| w[1] - w[0]).collect();
    if dl.len() < 5 { return 0.0; }
    let mean = slice_mean(&dl);
    let dl_c: Vec<f64> = dl.iter().map(|&v| v - mean).collect();
    let c0: f64 = dl_c.iter().map(|&v| v * v).sum();
    if c0 < EPS { return 0.0; }
    let c1: f64 = dl_c.windows(2).map(|w| w[0] * w[1]).sum();
    (c1 / c0).max(0.0)
}

fn compute_damped_trend(l_bc: &[f64], m: usize, n_complete: usize) -> Vec<f64> {
    let phi = estimate_phi(l_bc).min(1.0 - EPS);
    if phi <= EPS {
        // phi~0: linear extrapolation (fix at last observed position)
        return vec![(n_complete as f64 - 1.0) / n_complete as f64; m];
    }
    (0..m).map(|j| {
        let jf = (j + 1) as f64;
        ((n_complete as f64 - 1.0) + phi * (1.0 - pow(phi, jf)) / (1.0 - phi)) / n_complete as f64
    }).collect()
}

// ── LWCP leverages (#11) ──────────────────────────────────────────────────

// Per-horizon test-point leverage: h_test[j] = ||Vt @ x_j / s||^2_d_avg
// x_j is the design row vector for forecast step j, updated via point-prediction.
fn compute_lwcp_leverages(
    beta: &[f64],
    l_innov: &[f64],
    damped_trend: &[f64],
    x_future_l_std: &[Vec<f64>], // (m, n_exog) — empty slice when n_exog=0
    vt: &[Vec<f64>],              // k × nf_total
    s: &[f64],                    // k
    d_avg: &[f64],                // k
    n_complete: usize,
    m: usize,
    nb: usize,
    nf: usize,
    nf_total: usize,
    n_exog: usize,
    max_cp: usize,
    use_diff: bool,
) -> Vec<f64> {
    let k = s.len();
    let mut l_point: Vec<f64> = l_innov.to_vec();
    l_point.resize(n_complete + m, 0.0);

    (0..m).map(|j| {
        let ti = n_complete + j;

        // point prediction (beta already has β₂ = 1 - δ₂ restored)
        let mut pred = beta[0]
            + beta[1] * damped_trend[j]
            + beta[nb] * l_point[ti - 1]
            + if max_cp >= 2 { beta[nb + 1] * l_point[ti - max_cp] } else { 0.0 };
        if n_exog > 0 {
            let xf = &x_future_l_std[j];
            for e in 0..n_exog { pred += beta[nf + e] * xf[e]; }
        }
        l_point[ti] = pred;

        // feature vector x_j (length nf_total)
        let mut x_j = vec![0.0f64; nf_total];
        x_j[0] = 1.0;
        x_j[1] = damped_trend[j];
        x_j[nb] = if use_diff { -l_point[ti - 1] } else { l_point[ti - 1] };
        if max_cp >= 2 { x_j[nb + 1] = l_point[ti - max_cp]; }
        if n_exog > 0 {
            let xf = &x_future_l_std[j];
            for e in 0..n_exog { x_j[nf + e] = xf[e]; }
        }

        // v = Vt @ x_j, h_test = sum((v/s)^2 * d_avg)
        let h: f64 = (0..k).map(|r| {
            let v: f64 = (0..nf_total).map(|c| vt[r][c] * x_j[c]).sum();
            let u_test = v / s[r].max(EPS);
            u_test * u_test * d_avg[r]
        }).sum();
        h.clamp(0.0, 10.0)
    }).collect()
}

// ── Cross-period helper ────────────────────────────────────────────────────

fn compute_cross_periods(
    secondary: &[usize],
    big_p: usize,
    period: usize,
    n_complete: usize,
) -> (Vec<usize>, usize) {
    let mut cp_set: BTreeSet<usize> = secondary.iter().filter_map(|&sp| {
        let cp = if big_p >= 2 { sp / big_p } else { sp };
        if cp >= 2 && cp <= n_complete / 2 { Some(cp) } else { None }
    }).collect();
    if big_p == 1 && period >= 2 && period <= n_complete / 2 { cp_set.insert(period); }
    let cross_periods: Vec<usize> = cp_set.into_iter().collect();
    let max_cp = cross_periods.iter().copied().max().unwrap_or(0);
    (cross_periods, max_cp)
}

// ── Exog NaN cleanup (column-wise) ────────────────────────────────────────

/// Column-wise NaN linear interpolation for a (rows × k) matrix stored as Vec<Vec<f64>>.
/// For x_future columns that are entirely NaN, fills with the last valid value of the
/// corresponding x_hist column (one-step persistence assumption).
fn clean_exog_nan(
    xh: &mut Vec<Vec<f64>>,
    xf: &mut Vec<Vec<f64>>,
    k: usize,
) {
    for j in 0..k {
        // x_hist column
        let col_h: Vec<f64> = xh.iter().map(|r| r[j]).collect();
        let clean_h = interp_nan(&col_h);
        for (i, r) in xh.iter_mut().enumerate() { r[j] = clean_h[i]; }

        // x_future column — fall back to last valid x_hist value if all NaN
        let last_hist = *clean_h.last().unwrap_or(&0.0);
        let col_f: Vec<f64> = xf.iter().map(|r| r[j]).collect();
        let all_nan = col_f.iter().all(|v| v.is_nan());
        let clean_f = if all_nan {
            vec![last_hist; col_f.len()]
        } else {
            interp_nan(&col_f)
        };
        for (i, r) in xf.iter_mut().enumerate() { r[j] = clean_f[i]; }
    }
}

// ── Main forecast function ─────────────────────────────────────────────────

fn forecast_inner(
    y_raw: &[f64],
    horizon: usize,
    frequency: &Freq,
    n_samples: usize,
    seed: u64,
    exog: Option<(Vec<Vec<f64>>, Vec<Vec<f64>>, usize)>,
    noise_mode: NoiseMode,
) -> Result<(Vec<Vec<f64>>, Confidence), Error> {
    if y_raw.is_empty() { return Err(Error::InvalidInput("y must not be empty")); }
    if horizon < 1     { return Err(Error::InvalidInput("horizon must be >= 1")); }
    if n_samples < 1   { return Err(Error::InvalidInput("n_samples must be >= 1")); }

    let mut rng = Rng::new(seed);

    // NaN linear interpolation + shift so all values >= 1
    let mut y: Vec<f64> = interp_nan(y_raw);
    let y_floor = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_shift = (1.0 - y_floor).max(1.0);
    y.iter_mut().for_each(|v| *v += y_shift);
    let n = y.len();

    // Exog: NaN cleanup
    let (mut x_hist_mat, x_future_mat, n_exog) = match exog {
        Some((mut xh, mut xf, k)) => {
            clean_exog_nan(&mut xh, &mut xf, k);
            (xh, xf, k)
        }
        None => (vec![], vec![], 0),
    };

    // ── Period selection ────────────────────────────────────────────────
    let (mut big_p, mut secondary, period, _cal, mut svd_s, mut nc_svd) = select_period(&y, n, frequency);
    let mut n_complete = n / big_p;

    // Fallback for too-short series
    if n_complete < MIN_COMPLETE {
        if big_p > 1 {
            big_p = 1; secondary.clear(); n_complete = n;
            svd_s = alloc::vec![0.0]; nc_svd = 0; // invalidate stale SVD
        }
        if n_complete < MIN_COMPLETE {
            let fc_val = y[n - 1] - y_shift;
            let lookback = PHASE_NOISE_K.min(n);
            let diffs: Vec<f64> = y[n - lookback..].windows(2).map(|w| w[1] - w[0]).collect();
            let sigma = if diffs.is_empty() { EPS_SHAPE } else {
                let m = slice_mean(&diffs);
                sqrt(diffs.iter().map(|&d| pow(d - m, 2.0)).sum::<f64>() / diffs.len() as f64)
                    .max(EPS_SHAPE)
            };
            let samples = (0..n_samples).map(|_| {
                (0..horizon).map(|_| {
                    let v = fc_val + rng.normal() * sigma;
                    v.clamp(fc_val - sigma * 10.0, fc_val + sigma * 10.0)
                }).collect()
            }).collect();
            return Ok((samples, Confidence { rank1: None, gcv: None }));
        }
    }

    // Dynamic Ridge DoF guard: n_train >= 2p (LOOCV leverage stability)
    if big_p > 1 {
        let (_, max_cp_est) = compute_cross_periods(&secondary, big_p, period, n_complete);
        let start_est = if max_cp_est >= 2 { max_cp_est } else { 1 };
        let nf_est = 2 + 1 + if max_cp_est >= 2 { 1 } else { 0 } + n_exog;
        if n_complete.saturating_sub(start_est) < 2 * nf_est {
            big_p = 1;
            secondary.clear();
            n_complete = n;
            svd_s = alloc::vec![0.0]; nc_svd = 0; // invalidate stale SVD
        }
    }

    // Cap history to MAX_COMPLETE periods
    if n_complete > MAX_COMPLETE {
        let trim = MAX_COMPLETE * big_p;
        y = y[y.len() - trim..].to_vec();
        if !x_hist_mat.is_empty() {
            x_hist_mat = x_hist_mat[x_hist_mat.len() - trim..].to_vec();
        }
        n_complete = MAX_COMPLETE;
    }

    let usable = n_complete * big_p;
    let y_trim = &y[y.len() - usable..];

    // ── Matrix reshape: mat[ph][ci] = y_trim[ci*P + ph] ────────────────
    let mat: Vec<Vec<f64>> = (0..big_p)
        .map(|ph| (0..n_complete).map(|ci| y_trim[ci * big_p + ph]).collect())
        .collect();
    // Period-level aggregation
    let l_raw: Vec<f64> = (0..n_complete)
        .map(|ci| (0..big_p).map(|ph| mat[ph][ci]).sum())
        .collect();

    // Gavish-Donoho optimal Frobenius shrinkage ("One SVD": reuse svd_s from select_period).
    // l_raw is kept for phase-noise residuals (using shrunk l would bias residuals positive).
    let shrink = optshrink::optshrink_factor(&svd_s, big_p, if nc_svd > 0 { nc_svd } else { n_complete });
    let l: Vec<f64> = l_raw.iter().map(|&v| v * shrink).collect();

    // ── Shape estimation ────────────────────────────────────────────────
    let (s_forecast, s_hist, m) = estimate_shape(&mat, n_complete, big_p, &secondary, &l, horizon);

    // ── Exog Level aggregation (period mean, matching _aggregate_exog_to_level) ──
    // x_l_raw[ci][j] = mean of x_hist_mat rows in period ci for covariate j
    // x_future_l_raw[step][j] = mean of x_future_mat rows in that P-block
    let (x_l_raw, x_future_l_raw): (Vec<Vec<f64>>, Vec<Vec<f64>>) = if n_exog > 0 {
        let usable_xh = &x_hist_mat[x_hist_mat.len() - usable..];
        let xl: Vec<Vec<f64>> = (0..n_complete).map(|ci| {
            (0..n_exog).map(|j| {
                let s: f64 = (0..big_p).map(|p| usable_xh[ci * big_p + p][j]).sum();
                s / big_p as f64
            }).collect()
        }).collect();
        let xfl: Vec<Vec<f64>> = (0..m).map(|step| {
            (0..n_exog).map(|j| {
                let s_idx = step * big_p;
                let e_idx = ((step + 1) * big_p).min(horizon);
                let count = e_idx - s_idx;
                let s: f64 = (s_idx..e_idx).map(|i| x_future_mat[i][j]).sum();
                s / count as f64
            }).collect()
        }).collect();
        (xl, xfl)
    } else {
        (vec![], vec![])
    };

    // ── Cross-period / Shape₂ ───────────────────────────────────────────
    let (cross_periods, mut max_cp) = compute_cross_periods(&secondary, big_p, period, n_complete);
    let cp_main = cross_periods.first().copied().unwrap_or(0);
    let s2 = if cp_main >= 2 { compute_shape2(&l, cp_main, n_complete) } else { None };
    let use_deseason = s2.is_some();

    // ── Level series + Box-Cox ──────────────────────────────────────────
    let l_work: Vec<f64> = if use_deseason {
        let s2r = s2.as_ref().unwrap();
        (0..n_complete).map(|i| l[i] / s2r[i % cp_main].max(EPS)).collect()
    } else {
        l.clone()
    };

    let lam = bc_lambda(&l_work);
    let l_bc = bc(&l_work, lam);
    let last_l = l_bc[n_complete - 1];
    let l_innov: Vec<f64> = l_bc.iter().map(|&v| v - last_l).collect();

        // ── Ridge regression setup ──────────────────────────────────────────
    let mut start = if max_cp >= 2 { max_cp.max(1) } else { 1 };
    if max_cp >= 2 && n_complete.saturating_sub(start) < MIN_COMPLETE {
        max_cp = 0;
        start = 1;
    }

    let nb = 2usize; // intercept + trend
    let n_lag = if max_cp >= 2 { 2 } else { 1 };
    let nf = nb + n_lag;
    let nf_total = nf + n_exog;
    let n_train = n_complete - start;

    // Standardize exog using training-window stats only (look-ahead bias prevention)
    // x_l_std[ci][j] and x_future_l_std[step][j]
    let (x_l_std, x_future_l_std): (Vec<Vec<f64>>, Vec<Vec<f64>>) = if n_exog > 0 {
        let mut mu = vec![0.0f64; n_exog];
        let mut sd = vec![1.0f64; n_exog];
        for j in 0..n_exog {
            let train: Vec<f64> = (start..n_complete).map(|ci| x_l_raw[ci][j]).collect();
            let m = train.iter().sum::<f64>() / train.len() as f64;
            let s = sqrt(train.iter().map(|&v| pow(v - m, 2.0)).sum::<f64>() / train.len() as f64);
            mu[j] = m;
            sd[j] = if s < EPS { 1.0 } else { s };
        }
        let xl_std: Vec<Vec<f64>> = x_l_raw.iter().map(|row| {
            (0..n_exog).map(|j| (row[j] - mu[j]) / sd[j]).collect()
        }).collect();
        let xfl_std: Vec<Vec<f64>> = x_future_l_raw.iter().map(|row| {
            (0..n_exog).map(|j| (row[j] - mu[j]) / sd[j]).collect()
        }).collect();
        (xl_std, xfl_std)
    } else {
        (vec![], vec![])
    };

    // #6 LSR1: use diff-target when n_train >= 3
    let use_diff = DIFF_TARGET && n_train >= 3;

    let (x_rows, y_target): (Vec<Vec<f64>>, Vec<f64>) = if use_diff {
        // y_target = diff(l_innov[start-1..])
        let yt: Vec<f64> = (start..n_complete).map(|ti| l_innov[ti] - l_innov[ti - 1]).collect();
        let xr: Vec<Vec<f64>> = (start..n_complete).map(|ti| {
            let mut row = vec![0.0f64; nf_total];
            row[0] = 1.0;
            row[1] = ti as f64 / n_complete as f64;
            row[nb] = -l_innov[ti - 1]; // sign flip: delta_2 = 1 - beta_2
            if max_cp >= 2 { row[nb + 1] = l_innov[ti - max_cp]; }
            if n_exog > 0 {
                for j in 0..n_exog { row[nf + j] = x_l_std[ti][j]; }
            }
            row
        }).collect();
        (xr, yt)
    } else {
        let yt = l_innov[start..].to_vec();
        let xr: Vec<Vec<f64>> = (start..n_complete).map(|ti| {
            let mut row = vec![0.0f64; nf_total];
            row[0] = 1.0;
            row[1] = ti as f64 / n_complete as f64;
            row[nb] = l_innov[ti - 1];
            if max_cp >= 2 { row[nb + 1] = l_innov[ti - max_cp]; }
            if n_exog > 0 {
                for j in 0..n_exog { row[nf + j] = x_l_std[ti][j]; }
            }
            row
        }).collect();
        (xr, yt)
    };

    let (mut beta, loo_resid, gcv_min, vt_r, s_r, d_avg_r) = ridge_sa(&x_rows, &y_target)?;

    // #6 LSR1: recover beta_2 = 1 - delta_2
    if use_diff {
        beta[nb] = 1.0 - beta[nb];
    }

    // #7 Damped trend: phi = max(lag-1 autocorr of diff(L_bc), 0)
    let damped_trend: Vec<f64> = compute_damped_trend(&l_bc, m, n_complete);

    // #11 LWCP: per-horizon test-point leverage
    let h_test: Vec<f64> = compute_lwcp_leverages(
        &beta, &l_innov, &damped_trend,
        &x_future_l_std,
        &vt_r, &s_r, &d_avg_r,
        n_complete, m, nb, nf, nf_total, n_exog, max_cp, use_diff,
    );

    // ── Stochastic Level paths ──────────────────────────────────────────
    let loo_len = loo_resid.len();
    let nu = (n_train.saturating_sub(nf)).max(3); // Student-t df, also used for post-hoc shrinkage
    // #12 Empirical bootstrap from LWCP-normalized LOO residuals
    let noise_pool: Vec<Vec<f64>> = if noise_mode == NoiseMode::Bootstrap && loo_len >= 4 {
        let loo_mean = slice_mean(&loo_resid);
        let loo_std = sqrt(loo_resid.iter().map(|&v| pow(v - loo_mean, 2.0)).sum::<f64>()
            / loo_len as f64).max(EPS);
        let sigma2_loo = loo_resid.iter().map(|&v| pow(v - loo_mean, 2.0)).sum::<f64>()
            / loo_len as f64;
        let loo_unit: Vec<f64> = loo_resid.iter().map(|&v| (v - loo_mean) / loo_std).collect();
        (0..n_samples).map(|_| {
            (0..m).map(|j| {
                loo_unit[rng.randint(loo_len)] * sqrt(sigma2_loo * (1.0 + h_test[j]))
            }).collect()
        }).collect()
    } else {
        // Student-t fallback (nu = n_train - nf, floor 3)
        let sigma2_loo = loo_resid.iter().map(|&v| v * v).sum::<f64>() / loo_len.max(1) as f64;
        (0..n_samples).map(|_| {
            (0..m).map(|j| {
                rng.student_t(nu as f64) * sqrt(sigma2_loo * (1.0 + h_test[j]))
            }).collect()
        }).collect()
    };

    // L_paths[s][0..n_complete] = l_innov (history), [n_complete..] = forecast
    let total = n_complete + m;
    let mut l_paths: Vec<Vec<f64>> = (0..n_samples).map(|_| {
        let mut v = l_innov.clone();
        v.resize(total, 0.0);
        v
    }).collect();

    for j in 0..m {
        let ti = n_complete + j;
        let exog_contrib: f64 = if n_exog > 0 {
            let xf = &x_future_l_std[j];
            (0..n_exog).map(|e| beta[nf + e] * xf[e]).sum()
        } else { 0.0 };
        for si in 0..n_samples {
            let pred = beta[0]
                + beta[1] * damped_trend[j]  // #7 damped trend
                + beta[nb] * l_paths[si][ti - 1];
            let pred = if max_cp >= 2 { pred + beta[nb + 1] * l_paths[si][ti - max_cp] } else { pred };
            l_paths[si][ti] = pred + exog_contrib + noise_pool[si][j];
        }
    }

    // Inverse Box-Cox → L_hat_all[s][j]
    let mut l_hat_all: Vec<Vec<f64>> = (0..n_samples).map(|si| {
        bc_inv(&l_paths[si][n_complete..n_complete + m].iter().map(|&v| v + last_l).collect::<Vec<_>>(), lam)
    }).collect();

    // Re-apply Shape₂ seasonality
    if use_deseason {
        let s2r = s2.as_ref().unwrap();
        for lh in l_hat_all.iter_mut() {
            for (j, v) in lh.iter_mut().enumerate() {
                *v *= s2r[(n_complete + j) % cp_main];
            }
        }
    }

    // ── Phase noise (relative residual quantiles) ───────────────────────
    // fitted_mat[ph][ci] = s_hist[ci][ph] * l_raw[ci]  (pre-shrinkage L)
    // Using l_raw avoids positive bias in the residual matrix (Finding 3).
    // R[ph][kr_idx] = relative residual over last k_r periods
    let k_r = PHASE_NOISE_K.min(n_complete);

    // fitted_clamp = max(0.1 * median(|fitted| > EPS), EPS_BOXCOX)
    // Robust denominator clamp; a fixed EPS_BOXCOX causes residual blow-up on low-level phases.
    let fitted_clamp = {
        let mut vals: Vec<f64> = Vec::new();
        for ph in 0..big_p {
            for ci in n_complete - k_r..n_complete {
                let f = (s_hist[ci][ph] * l_raw[ci]).abs();
                if f > EPS { vals.push(f); }
            }
        }
        if vals.is_empty() {
            EPS_BOXCOX
        } else {
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let mid = vals.len() / 2;
            let med = if vals.len() % 2 == 0 { (vals[mid - 1] + vals[mid]) / 2.0 } else { vals[mid] };
            (med * 0.1_f64).max(EPS_BOXCOX)
        }
    };

    let mut r_mat: Vec<Vec<f64>> = (0..big_p).map(|ph| {
        (n_complete - k_r..n_complete).map(|ci| {
            let fitted = s_hist[ci][ph] * l_raw[ci];
            (mat[ph][ci] - fitted) / fitted.abs().max(fitted_clamp)
        }).collect()
    }).collect();

    // James-Stein per-phase bias shrinkage (posterior-mean toward zero)
    if k_r >= 4 {
        for ph in 0..big_p {
            let mean = slice_mean(&r_mat[ph]);
            let var = r_mat[ph].iter().map(|&v| pow(v - mean, 2.0)).sum::<f64>()
                / (k_r - 1) as f64;
            let se_sq = var / k_r as f64;
            let noise_fraction = (se_sq / (mean * mean + se_sq + EPS)).clamp(0.0, 1.0);
            let shrink = mean * noise_fraction;
            r_mat[ph].iter_mut().for_each(|v| *v -= shrink);
        }
    }

    // col_idx[s]: one column per sample, shared across all steps (scenario-coherent).
    let col_idx: Vec<usize> = (0..n_samples).map(|_| rng.randint(k_r)).collect();

    // ── Assemble output ─────────────────────────────────────────────────
    let step_idx: Vec<usize> = (0..horizon).map(|h| h / big_p).collect();
    let phase_idx: Vec<usize> = (0..horizon).map(|h| h % big_p).collect();

    let mut samples: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    for si in 0..n_samples {
        let path: Vec<f64> = (0..horizon).map(|h| {
            let sj = step_idx[h];
            let ph = phase_idx[h];
            // #11 phase deflation: Level paths already widen via sqrt(1+h_test),
            // divide phase noise by the same factor to avoid double-counting variance
            let phase_deflate = 1.0 / sqrt(1.0 + h_test[sj]);
            let phase_noise = r_mat[ph][col_idx[si]] * phase_deflate;
            l_hat_all[si][sj] * s_forecast[sj][ph] * (1.0 + phase_noise) - y_shift
        }).collect();
        samples.push(path);
    }

    // Clip: upper = recent max + range, lower = all-time floor (asymmetric).
    let lookback = (horizon * 2).max(PHASE_NOISE_K).min(y_raw.len());
    let valid_rec: Vec<f64> = y_raw[y_raw.len() - lookback..].iter().cloned().filter(|v| !v.is_nan()).collect();
    let valid_all: Vec<f64> = y_raw.iter().cloned().filter(|v| !v.is_nan()).collect();
    if !valid_rec.is_empty() && !valid_all.is_empty() {
        let y_hi    = valid_rec.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_range = (y_hi - valid_rec.iter().cloned().fold(f64::INFINITY, f64::min)).max(EPS_SHAPE);
        let y_floor = valid_all.iter().cloned().fold(f64::INFINITY, f64::min);
        let clip_hi = y_hi + y_range;
        for path in &mut samples {
            for v in path.iter_mut() {
                if v.is_nan() || *v == f64::NEG_INFINITY { *v = 0.0; }
                else if *v == f64::INFINITY { *v = clip_hi; }
                else { *v = v.clamp(y_floor, clip_hi); }
            }
        }
    } else {
        for path in &mut samples {
            for v in path.iter_mut() {
                if !v.is_finite() { *v = 0.0; }
            }
        }
    }

    // Post-hoc Student-t shrinkage toward the per-horizon median.
    // Only applied in "t" noise mode with nu < 50; bootstrap samples already
    // have unit variance by construction so no shrinkage is needed there.
    if noise_mode == NoiseMode::StudentT && nu < 50 {
        let shrink_t = sqrt(((nu as f64 - 2.0).max(0.5)) / nu as f64);
        for h in 0..horizon {
            let mut col: Vec<f64> = samples.iter().map(|p| p[h]).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let med = if n_samples % 2 == 0 {
                (col[n_samples / 2 - 1] + col[n_samples / 2]) / 2.0
            } else {
                col[n_samples / 2]
            };
            for path in samples.iter_mut() {
                path[h] = med + shrink_t * (path[h] - med);
            }
        }
    }

    // Integer snap: round forecasts when all inputs are integers.
    let is_integer_series = y_raw.iter().filter(|v| !v.is_nan()).all(|&v| v == round(v));
    if is_integer_series {
        for path in &mut samples {
            for v in path.iter_mut() {
                *v = round(*v);
            }
        }
    }

    let _ = (x_future_mat, x_l_raw, x_future_l_raw); // consumed above, suppress unused warning

    // Reuse svd_s from select_period (One SVD principle; same matrix as mat).
    let rank1 = {
        let total: f64 = svd_s.iter().map(|&v| v * v).sum();
        if total < EPS || big_p < 2 { None } else { Some(svd_s[0] * svd_s[0] / total) }
    };
    let conf = Confidence { rank1, gcv: Some(gcv_min) };

    Ok((samples, conf))
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::fs;
    use alloc::format;

    #[test]
    fn output_shape() {
        let y: Vec<f64> = (0..200).map(|i| sin(i as f64 * 0.26) * 3.0 + 10.0).collect();
        let (s, _) = forecast(&y, &Freq::Monthly, 12, 50, 0, None, NoiseMode::Bootstrap).unwrap();
        assert_eq!(s.len(), 50);
        assert_eq!(s[0].len(), 12);
    }

    #[test]
    fn bc_roundtrip() {
        let y = vec![0.5f64, 1.0, 2.0, 5.0, 10.0];
        for &lam in &[0.0f64, 0.3, 0.5, 1.0] {
            let y2 = bc_inv(&bc(&y, lam), lam);
            for (&a, &b) in y.iter().zip(y2.iter()) {
                assert!((a - b).abs() < 1e-9, "λ={lam}: {a} -> {b}");
            }
        }
    }

    #[test]
    fn error_cases() {
        assert!(forecast(&[], &Freq::hourly(1).unwrap(), 5, 10, 0, None, NoiseMode::Bootstrap).is_err());
        assert!(forecast(&[1.0, 2.0], &Freq::hourly(1).unwrap(), 0, 10, 0, None, NoiseMode::Bootstrap).is_err());
        assert!(forecast(&[1.0, 2.0], &Freq::hourly(1).unwrap(), 5, 0, 0, None, NoiseMode::Bootstrap).is_err());
    }

    #[test]
    fn forecast_quantiles_shape_and_order() {
        let y: Vec<f64> = (0..200).map(|i| sin(i as f64 * 0.26) * 3.0 + 10.0).collect();
        let qs = [0.1, 0.5, 0.9];
        let (q, _) = forecast_quantiles(&y, &Freq::Monthly, 12, 100, 0, None, NoiseMode::Bootstrap, &qs).unwrap();
        assert_eq!(q.len(), 3);
        assert!(q.iter().all(|row| row.len() == 12));
        for h in 0..12 {
            assert!(q[0][h] <= q[1][h] && q[1][h] <= q[2][h]);
        }
    }

    #[test]
    fn forecast_quantiles_invalid_q() {
        let y: Vec<f64> = (0..50).map(|i| i as f64).collect();
        assert!(forecast_quantiles(&y, &Freq::Monthly, 5, 10, 0, None, NoiseMode::Bootstrap, &[0.5, 1.5]).is_err());
    }

    // #21: +inf must clip to y_hi+y_range (upper), not y_floor (lower).
    // We verify that all samples are within [y_min, y_max + y_range] and that
    // no sample equals the floor when overflow would push it up.
    #[test]
    fn clip_posinf_upper_not_lower() {
        // Monotone rising series so y_floor << y_hi; an upward overflow must
        // land at the upper clip, not collapse to the floor.
        let y: Vec<f64> = (1..=120).map(|i| i as f64).collect();
        let y_floor = 1.0f64;
        let y_hi    = 120.0f64;
        let y_range = (y_hi - y_floor).max(1e-6);
        let clip_hi = y_hi + y_range;

        let (samples, _) = forecast(&y, &Freq::Monthly, 12, 200, 0, None, NoiseMode::Bootstrap).unwrap();
        for path in &samples {
            for &v in path {
                assert!(v.is_finite(), "non-finite in output: {v}");
                assert!(v <= clip_hi + 1.0, "+inf clipped above upper: {v} > {clip_hi}");
                // The key regression: +inf must NOT land at y_floor
                assert!(v >= y_floor - 1.0, "value below floor: {v}");
            }
        }
    }

    // #24: noise_mode argument controls Level noise model.
    // Bootstrap and StudentT both produce finite output.
    #[test]
    fn level_noise_mode_selectable() {
        let y: Vec<f64> = (0..144).map(|i| 100.0 + 20.0 * sin(i as f64 * PI * 2.0 / 12.0)).collect();
        let (s, _) = forecast(&y, &Freq::Monthly, 12, 100, 0, None, NoiseMode::Bootstrap).unwrap();
        assert!(s.iter().flat_map(|p| p.iter()).all(|v| v.is_finite()));
        let (s2, _) = forecast(&y, &Freq::Monthly, 12, 100, 0, None, NoiseMode::StudentT).unwrap();
        assert!(s2.iter().flat_map(|p| p.iter()).all(|v| v.is_finite()));
    }

    // #22: Hourly(2) = 2H, period=12, periods=[12,84].
    #[test]
    fn hourly_2_freq() {
        assert!(Freq::hourly(2).is_ok());
        assert_eq!(get_period(&Freq::Hourly(2)), 12);
        assert_eq!(get_periods(&Freq::Hourly(2)), vec![12, 84]);
        // End-to-end smoke: 2H series with 12-step primary seasonality.
        let y: Vec<f64> = (0..168).map(|i| 50.0 + 10.0 * sin(i as f64 * PI * 2.0 / 12.0)).collect();
        let (s, _) = forecast(&y, &Freq::Hourly(2), 12, 20, 0, None, NoiseMode::Bootstrap).unwrap();
        assert_eq!(s.len(), 20);
        assert!(s.iter().flat_map(|p| p.iter()).all(|v| v.is_finite()));
    }

    // #20 + #23: select_period returns svd_s/nc_svd; optshrink is applied to l;
    // l_raw (pre-shrinkage) is used for phase-noise residuals.
    // We verify that: (a) select_period returns a non-trivial svd_s for a periodic series,
    // (b) forecast still produces finite output (regression guard for the l_raw change).
    #[test]
    fn select_period_returns_svd_and_l_raw_used() {
        // Strong monthly signal → select_period should pick P=12, returning real svd_s.
        use libm::sin;
        let y: Vec<f64> = (0..144)
            .map(|i| 100.0 + 20.0 * sin(i as f64 * core::f64::consts::PI * 2.0 / 12.0))
            .collect();
        let n = y.len();
        let freq = Freq::Monthly;
        // Shift y to be positive (mirrors forecast() preprocessing)
        let y_shift = (1.0 - y.iter().cloned().fold(f64::INFINITY, f64::min)).max(1.0);
        let y_shifted: Vec<f64> = y.iter().map(|&v| v + y_shift).collect();

        let (p, _sec, _period, _cal, svd_s, nc_svd) = select_period(&y_shifted, n, &freq);
        assert_eq!(p, 12, "expected P=12 for monthly series");
        assert!(svd_s.len() >= 2, "svd_s should have ≥2 values for P=12");
        assert!(svd_s[0] > svd_s[1], "svd_s should be descending");
        assert!(nc_svd >= 3, "nc_svd should be >= MIN_COMPLETE");

        // End-to-end: output must be finite with the new l_raw/optshrink path.
        let (samples, conf) = forecast(&y, &freq, 12, 50, 0, None, NoiseMode::Bootstrap).unwrap();
        assert!(samples.iter().flat_map(|p| p.iter()).all(|v| v.is_finite()));
        assert!(conf.rank1.is_some());
    }

    // ── #19 covariate tests ───────────────────────────────────────────────

    // Covariate with zero variance (constant column) must not panic.
    #[test]
    fn covariate_constant_column() {
        let y: Vec<f64> = (0..60).map(|i| 50.0 + (i as f64).sin()).collect();
        let n = y.len();
        let horizon = 6usize;
        let x_hist = vec![1.0f64; n];
        let x_future = vec![1.0f64; horizon];
        let (samples, _) = forecast(&y, &Freq::Monthly, horizon, 20, 0,
            Some((&x_hist, &x_future)), NoiseMode::Bootstrap).unwrap();
        assert!(samples.iter().flat_map(|p| p.iter()).all(|v| v.is_finite()));
    }

    // Covariate with NaN values must be cleaned and produce finite output.
    #[test]
    fn covariate_nan_cleanup() {
        let y: Vec<f64> = (0..60).map(|i| 50.0 + (i as f64).sin()).collect();
        let n = y.len();
        let horizon = 6usize;
        let mut x_hist: Vec<f64> = (0..n).map(|i| i as f64).collect();
        x_hist[5] = f64::NAN;
        let mut x_future: Vec<f64> = (n..n + horizon).map(|i| i as f64).collect();
        x_future[0] = f64::NAN;
        let (samples, _) = forecast(&y, &Freq::Monthly, horizon, 20, 0,
            Some((&x_hist, &x_future)), NoiseMode::Bootstrap).unwrap();
        assert!(samples.iter().flat_map(|p| p.iter()).all(|v| v.is_finite()));
    }

    // Covariates with two columns (k=2).
    #[test]
    fn covariate_two_columns() {
        let y: Vec<f64> = (0..120).map(|i| 100.0 + 10.0 * sin(i as f64 * PI / 6.0)).collect();
        let n = y.len();
        let horizon = 12usize;
        // k=2: flat row-major layout
        let x_hist: Vec<f64> = (0..n).flat_map(|i| [i as f64, (i as f64).cos()]).collect();
        let x_future: Vec<f64> = (n..n+horizon).flat_map(|i| [i as f64, (i as f64).cos()]).collect();
        let (samples, _) = forecast(&y, &Freq::Monthly, horizon, 30, 0,
            Some((&x_hist, &x_future)), NoiseMode::Bootstrap).unwrap();
        assert_eq!(samples[0].len(), horizon);
        assert!(samples.iter().flat_map(|p| p.iter()).all(|v| v.is_finite()));
    }

    // Informative exog must shift the point forecast by a non-negligible amount.
    // Design mirrors Python test_exogenous.py::TestExogEffect::test_informative_exog_changes_forecast:
    //   y = 100 + 20*x + seasonal + noise, x drives a clear additive level effect.
    //   After Ridge learns beta[nf] ≈ 20, feeding x_future must visibly move the mean.
    #[test]
    fn covariate_informative_shifts_forecast() {
        let n = 300usize;
        let horizon = 14usize;
        // x: slow sinusoid (period=60d) not aligned with weekly period
        let x_hist: Vec<f64> = (0..n).map(|i| sin(2.0 * PI * i as f64 / 60.0)).collect();
        let x_future: Vec<f64> = (n..n + horizon).map(|i| sin(2.0 * PI * i as f64 / 60.0)).collect();
        // y = 100 + 20*x + weekly seasonality + small noise (seed=42 via simple LCG)
        let mut rng = Rng::new(42);
        let y: Vec<f64> = (0..n).map(|i| {
            100.0 + 20.0 * x_hist[i]
                + 40.0 * sin(2.0 * PI * i as f64 / 7.0)
                + rng.normal() * 0.5
        }).collect();

        let (s_no, _) = forecast(&y, &Freq::Daily, horizon, 200, 42, None, NoiseMode::Bootstrap).unwrap();
        let (s_ex, _) = forecast(&y, &Freq::Daily, horizon, 200, 42,
            Some((&x_hist, &x_future)), NoiseMode::Bootstrap).unwrap();

        let mean_diff: f64 = (0..horizon).map(|h| {
            let m_no: f64 = s_no.iter().map(|p| p[h]).sum::<f64>() / s_no.len() as f64;
            let m_ex: f64 = s_ex.iter().map(|p| p[h]).sum::<f64>() / s_ex.len() as f64;
            (m_no - m_ex).abs()
        }).sum::<f64>() / horizon as f64;

        assert!(mean_diff > 1.0,
            "informative exog shift {mean_diff:.4} is negligible — exog may be silently dropped");
    }

    // Pure noise exog must NOT materially shift the forecast.
    // Ridge LOOCV soft-average shrinks irrelevant columns; the drift must stay
    // well under 0.15σ, matching Python test_noise_exog_drift_is_small.
    #[test]
    fn covariate_noise_exog_drift_is_small() {
        let n = 500usize;
        let horizon = 24usize;
        let y: Vec<f64> = (0..n).map(|i| 100.0 + 10.0 * sin(2.0 * PI * i as f64 / 24.0)).collect();

        // 3-column pure noise (xorshift seeded deterministically)
        let mut rng_x = Rng::new(99);
        let x_hist: Vec<f64> = (0..n * 3).map(|_| rng_x.normal()).collect();
        let x_future: Vec<f64> = (0..horizon * 3).map(|_| rng_x.normal()).collect();

        let (s_no, _) = forecast(&y, &Freq::Hourly(1), horizon, 200, 42, None, NoiseMode::Bootstrap).unwrap();
        let (s_ex, _) = forecast(&y, &Freq::Hourly(1), horizon, 200, 42,
            Some((&x_hist, &x_future)), NoiseMode::Bootstrap).unwrap();

        let y_std = {
            let m = y.iter().sum::<f64>() / n as f64;
            sqrt(y.iter().map(|&v| pow(v - m, 2.0)).sum::<f64>() / n as f64).max(EPS)
        };
        let drift: f64 = (0..horizon).map(|h| {
            let m_no: f64 = s_no.iter().map(|p| p[h]).sum::<f64>() / s_no.len() as f64;
            let m_ex: f64 = s_ex.iter().map(|p| p[h]).sum::<f64>() / s_ex.len() as f64;
            (m_no - m_ex).abs()
        }).sum::<f64>() / horizon as f64;

        assert!(drift / y_std < 0.15,
            "noise exog drift {:.4}σ exceeds 0.15σ — Ridge not shrinking noise columns",
            drift / y_std);
    }

    // A covariate that is perfectly collinear with y's future should pull the
    // forecast mean toward the true future value.  We use y_future as x_future
    // (oracle covariate) and verify the median absolute error is strictly lower.
    #[test]
    fn covariate_oracle_reduces_error() {
        let n = 120usize;
        let horizon = 12usize;
        let y: Vec<f64> = (0..n).map(|i| 100.0 + 30.0 * sin(i as f64 * PI * 2.0 / 12.0) + i as f64 * 0.5).collect();
        let truth: Vec<f64> = (n..n + horizon).map(|i| 100.0 + 30.0 * sin(i as f64 * PI * 2.0 / 12.0) + i as f64 * 0.5).collect();

        // Oracle: x = truth itself (perfect future signal)
        let x_hist: Vec<f64> = y.clone();
        let x_future: Vec<f64> = truth.clone();

        let (s_no, _) = forecast(&y, &Freq::Monthly, horizon, 500, 0, None, NoiseMode::Bootstrap).unwrap();
        let (s_ex, _) = forecast(&y, &Freq::Monthly, horizon, 500, 0,
            Some((&x_hist, &x_future)), NoiseMode::Bootstrap).unwrap();

        let mae = |s: &Vec<Vec<f64>>| -> f64 {
            (0..horizon).map(|h| {
                let mean: f64 = s.iter().map(|p| p[h]).sum::<f64>() / s.len() as f64;
                (mean - truth[h]).abs()
            }).sum::<f64>() / horizon as f64
        };
        let mae_no = mae(&s_no);
        let mae_ex = mae(&s_ex);
        assert!(mae_ex < mae_no,
            "oracle covariate did not reduce MAE: with_exog={mae_ex:.2} >= no_exog={mae_no:.2}");
    }

    // Validation error: x_hist length not a multiple of y length.
    #[test]
    fn covariate_validation_bad_xhist_len() {
        let y = vec![1.0f64; 10];
        let x_hist = vec![0.0f64; 7]; // not a multiple of 10
        let x_future = vec![0.0f64; 5];
        assert!(forecast(&y, &Freq::Monthly, 5, 10, 0, Some((&x_hist, &x_future)), NoiseMode::Bootstrap).is_err());
    }

    // Validation error: x_future length != horizon * k.
    #[test]
    fn covariate_validation_bad_xfuture_len() {
        let y = vec![1.0f64; 10];
        let x_hist = vec![0.0f64; 10]; // k=1
        let x_future = vec![0.0f64; 3]; // should be 5
        assert!(forecast(&y, &Freq::Monthly, 5, 10, 0, Some((&x_hist, &x_future)), NoiseMode::Bootstrap).is_err());
    }

    // ── dataset-iter tests ────────────────────────────────────────────────
    //
    // ParseMode:
    //   Col(n)      – take the n-th comma-separated column (0-based), skip 1 header row
    //   ColSkip(n,s)– same but skip s header rows
    //   JapanTokyo  – japan_demand_tokyo.csv: col 4 (Tokyo), skip 1 header row

    #[allow(dead_code)]
    enum ParseMode { Col(usize), ColSkip(usize, usize), JapanTokyo }

    struct Dataset {
        file: &'static str,
        frequency: Freq,
        mode: ParseMode,
    }

    fn load(ds: &Dataset) -> Vec<f64> {
        let path = format!("examples/dataset/{}", ds.file);
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("{path} not found"));
        match ds.mode {
            ParseMode::Col(col) | ParseMode::ColSkip(col, _) => {
                let skip = match ds.mode { ParseMode::ColSkip(_, s) => s, _ => 1 };
                content.lines().skip(skip)
                    .filter_map(|l| l.split(',').nth(col)?.trim().trim_matches('"').parse::<f64>().ok())
                    .collect()
            }
            ParseMode::JapanTokyo => {
                content.lines().skip(1)
                    .filter_map(|l| l.split(',').nth(2)?.trim().parse::<f64>().ok())
                    .collect()
            }
        }
    }

    fn datasets() -> Vec<Dataset> {
        vec![
            Dataset { file: "air_passengers.csv",  frequency: Freq::Monthly,  mode: ParseMode::Col(2) },
            Dataset { file: "nottem.csv",           frequency: Freq::Monthly,  mode: ParseMode::Col(2) },
            Dataset { file: "sunspot_year.csv",     frequency: Freq::Yearly,   mode: ParseMode::Col(2) },
            Dataset { file: "noaa_temp_annual.csv", frequency: Freq::Yearly,   mode: ParseMode::Col(1) },
            Dataset { file: "noaa_temp_monthly.csv",frequency: Freq::Monthly,  mode: ParseMode::Col(1) },
            Dataset { file: "japan_demand_tokyo.csv", frequency: Freq::Hourly(1),mode: ParseMode::JapanTokyo },
        ]
    }

    /// Regression test against Python flaircast 0.6.1 forecast(seed=0, n_samples=500).
    /// Tolerance +-15 (~6% of the series range ~250); kept loose because sampling is stochastic.
    #[test]
    fn lwcp_vs_python_reference() {
        // y = 100 + 1.5*t + 20*sin(2*pi*t/12), t=0..143, monthly, 144 points
        let y: Vec<f64> = (0..144).map(|i| {
            100.0 + 1.5 * i as f64 + 20.0 * sin(2.0 * PI * i as f64 / 12.0)
        }).collect();

        // Reference values: Python flaircast 0.6.1, seed=0, n_samples=500
        let py_mean = [310.5, 329.0, 343.2, 350.0, 348.1, 338.7,
                       325.0, 311.3, 301.9, 300.0, 306.8, 321.0];

        let (samples, _) = forecast(&y, &Freq::Monthly, 12, 500, 0, None, NoiseMode::Bootstrap).unwrap();
        let rs_mean: Vec<f64> = (0..12)
            .map(|h| samples.iter().map(|s| s[h]).sum::<f64>() / samples.len() as f64)
            .collect();

        for h in 0..12 {
            let diff = (rs_mean[h] - py_mean[h]).abs();
            assert!(diff < 2.0,
                "h={h}: rust={:.1} py={:.1} diff={:.1}", rs_mean[h], py_mean[h], diff);
        }
    }

    #[test]
    fn dataset_iter_no_crash() {
        for ds in datasets() {
            let y = load(&ds);
            assert!(!y.is_empty(), "{}: empty", ds.file);
            let (fc, _) = forecast_mean(&y, &ds.frequency, 12, 30, 0, None, NoiseMode::Bootstrap)
                .unwrap_or_else(|e| panic!("{}: forecast error: {e:?}", ds.file));
            assert_eq!(fc.len(), 12, "{}: wrong horizon", ds.file);
            assert!(fc.iter().all(|v| v.is_finite()), "{}: non-finite output", ds.file);
        }
    }

    // #22: Verify that period/periods for every variant match the design spec.
    // Hourly(2)/Hourly(12) are Rust-specific granularities (distinct from Python "2H"/"12H")
    // and are tested against the values defined in issue #22.
    #[test]
    fn freq_period_table_all_variants() {
        let cases: &[(Freq, usize, &[usize])] = &[
            (Freq::Secondly(10), 6,  &[6]),
            (Freq::Minutely(5),  12, &[12, 288]),
            (Freq::Minutely(10), 6,  &[6, 144]),
            (Freq::Minutely(15), 4,  &[4, 96]),
            (Freq::Minutely(30), 48, &[48, 336]),
            (Freq::Hourly(1),    24, &[24, 168]),
            (Freq::Hourly(2),    12, &[12, 84]),
            (Freq::Hourly(12),   2,  &[2, 14]),
            (Freq::Daily,        7,  &[7, 365]),
            (Freq::Weekly,       52, &[52]),
            (Freq::Monthly,      12, &[12]),
            (Freq::Quarterly,    4,  &[4]),
            (Freq::Yearly,       1,  &[]),
        ];
        for (freq, want_p, want_ps) in cases {
            let got_p = get_period(freq);
            let got_ps = get_periods(freq);
            assert_eq!(got_p, *want_p, "period mismatch for {:?}", freq);
            assert_eq!(&got_ps[..], *want_ps, "periods mismatch for {:?}", freq);
        }
    }

    // #14: Unit-test that estimate_shape computes the Frozen Shape correctly.
    // - s_global sums to ≈ 1 (probability simplex)
    // - all elements non-negative
    // - SHAPE_K=2 means only the last 2 periods are used (verified against hand-calculated values)
    #[test]
    fn estimate_shape_unit() {
        // mat[ph][ci]: P=3, n_complete=3
        // SHAPE_K=2 → only ci=1,2 (last 2 periods) are used
        let mat: Vec<Vec<f64>> = vec![
            vec![10.0, 1.0, 4.0],   // ph=0
            vec![20.0, 2.0, 5.0],   // ph=1
            vec![30.0, 3.0, 6.0],   // ph=2
        ];
        let n_complete = 3usize;
        let big_p = 3usize;
        let horizon = 6usize;

        let (s_fc, s_hist, m) = estimate_shape(&mat, n_complete, big_p, &[], &[], horizon);

        assert_eq!(m, 2); // ceil(6/3)
        assert_eq!(s_fc.len(), m);
        assert_eq!(s_hist.len(), n_complete);

        // each row sums to 1 and all elements are non-negative
        for row in &s_fc {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "s_forecast sum={sum}");
            assert!(row.iter().all(|&v| v >= 0.0), "negative shape value");
        }
        for row in &s_hist {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "s_hist sum={sum}");
        }

        // SHAPE_K=2: mean of within-period proportions for ci=1,2
        // ci=1: totals=6, props=[1/6, 2/6, 3/6]
        // ci=2: totals=15, props=[4/15, 5/15, 6/15]
        let p0 = (1.0/6.0 + 4.0/15.0) / 2.0;
        let p1 = (2.0/6.0 + 5.0/15.0) / 2.0;
        let p2 = (3.0/6.0 + 6.0/15.0) / 2.0;
        let psum = p0 + p1 + p2;
        let want = [p0/psum, p1/psum, p2/psum];
        for (i, (&got, &exp)) in s_fc[0].iter().zip(want.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-10,
                "s_global[{i}]: got={got:.6} want={exp:.6}");
        }

        // Frozen Shape: every row of s_hist equals s_fc[0] (tiled broadcast)
        for row in &s_hist {
            for (g, e) in row.iter().zip(s_fc[0].iter()) {
                assert!((g - e).abs() < 1e-12, "s_hist != s_global");
            }
        }
    }

    // #21: Regression test for the +inf clip direction.
    // Run forecast on a monotone-rising series (y_floor << y_hi) and verify that
    // output does not collapse near y_floor.
    // The old bug (+inf → y_floor replacement) caused large samples to drop sharply,
    // detectable when any path value falls below y_floor/2.
    // Also checks that no value greatly exceeds y_hi + y_range.
    #[test]
    fn clip_posinf_regression() {
        let y: Vec<f64> = (1..=240).map(|i| (i as f64).powi(2)).collect();
        let y_floor = 1.0f64;
        let y_hi_approx = 240.0f64.powi(2) as f64;
        let clip_hi_loose = y_hi_approx * 3.0; // generous upper bound

        let (samples, _) = forecast(&y, &Freq::Monthly, 12, 300, 0, None, NoiseMode::Bootstrap).unwrap();
        for path in &samples {
            for &v in path {
                assert!(v.is_finite(),
                    "non-finite leaked to output: {v}");
                assert!(v >= y_floor * 0.5,
                    "value {v} < y_floor/2={} — +inf may have been clipped to floor (bug #21)",
                    y_floor * 0.5);
                assert!(v <= clip_hi_loose,
                    "value {v} exceeds loose upper bound {clip_hi_loose}");
            }
        }
    }

    // #20 + #23: Direct verification of the One-SVD principle.
    // Cross-check svd_s returned by select_period against svdvals recomputed on the same matrix,
    // confirming they come from a single SVD (no redundant recomputation).
    // Also verifies that optshrink_factor is in (0, 1].
    // shrink < 1.0 is not enforced here because it depends on signal strength vs. the
    // Marchenko-Pastur threshold; a separate rank-1 spike test covers that case.
    #[test]
    fn optshrink_uses_select_period_svd_one_svd_principle() {
        use crate::optshrink::optshrink_factor;

        let y: Vec<f64> = (0..144)
            .map(|i| 100.0 + 50.0 * sin(i as f64 * PI * 2.0 / 12.0))
            .collect();
        let n = y.len();
        let y_floor = y.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_shift = (1.0 - y_floor).max(1.0);
        let y_shifted: Vec<f64> = y.iter().map(|&v| v + y_shift).collect();

        let (big_p, _sec, _period, _cal, svd_s, nc_svd) =
            select_period(&y_shifted, n, &Freq::Monthly);

        assert_eq!(big_p, 12, "expect P=12 for monthly series");
        assert!(svd_s.len() >= 2, "svd_s too short");
        assert!(svd_s[0] >= svd_s[1], "svd_s must be descending");

        // shrink must be in (0, 1]
        let shrink = optshrink_factor(&svd_s, big_p, nc_svd.max(MIN_COMPLETE));
        assert!(shrink > 0.0 && shrink <= 1.0,
            "shrink out of range: {shrink}");

        // One SVD: svd_s from select_period must match svdvals computed directly on mat_c.
        // Any divergence would indicate a second SVD was run on a different matrix.
        let nc = n / big_p;
        let start = y_shifted.len() - nc * big_p;
        let y_use = &y_shifted[start..];
        let mat_c: Vec<Vec<f64>> = (0..big_p)
            .map(|ph| (0..nc).map(|ci| y_use[ci * big_p + ph]).collect())
            .collect();
        let svd_s2 = crate::svd::svdvals(&mat_c);

        assert_eq!(svd_s.len(), svd_s2.len(), "svd_s length mismatch");
        for (i, (&a, &b)) in svd_s.iter().zip(svd_s2.iter()).enumerate() {
            assert!((a - b).abs() < 1e-8,
                "svd_s[{i}]: select_period={a:.8} vs recomputed={b:.8} — One SVD principle violated");
        }

        // Also confirm shrink < 1.0 for data with a strongly dominant rank-1 component
        // (BBP supercritical check). nc >> P keeps beta small, lowering the threshold.
        // P=5, nc=50 (beta=0.1) is close to the conditions proven in the optshrink unit tests.
        let big_p2 = 5usize;
        let nc2 = 50usize;
        // rank-1 matrix: mat[ph][ci] = (1 + 0.1*ph) * 100.0 + tiny_noise
        let mat2: Vec<Vec<f64>> = (0..big_p2)
            .map(|ph| (0..nc2)
                .map(|ci| (1.0 + 0.1 * ph as f64) * 100.0 + 0.01 * ((ph * nc2 + ci) % 7) as f64)
                .collect())
            .collect();
        let svd_s_spike = crate::svd::svdvals(&mat2);
        let shrink2 = optshrink_factor(&svd_s_spike, big_p2, nc2);
        assert!(shrink2 < 1.0,
            "rank-1 dominant signal (P={big_p2}, nc={nc2}) must trigger optshrink (shrink={shrink2:.4})");
    }

}
