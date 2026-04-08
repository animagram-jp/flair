//! FLAIR: Factored Level And Interleaved Ridge
//! Rust port of <https://github.com/TakatoHonda/FLAIR>
//!
#![allow(clippy::many_single_char_names, clippy::too_many_arguments)]

use core::{
    cmp::Ordering,
    f64::consts::PI,
};
use alloc::{
    collections::BTreeSet,
    format,
    string::String,
    vec,
    vec::Vec,
};
use crate::svd;
use libm::{sqrt, log as ln, exp, pow, sin, cos, round};

// ── constants ──────────────────────────────────────────────────────────────

const EPS: f64 = 1e-10;
const EPS_BOXCOX: f64 = 1e-8;
const EPS_LOG: f64 = 1e-30;
const EPS_WEIGHT: f64 = 1e-15;
const EPS_SHAPE: f64 = 1e-6;
const BC_EXP_CLIP: f64 = 30.0;
const MIN_POSITIVE_FOR_BC: usize = 10;
const MIN_COMPLETE: usize = 3;
const MAX_COMPLETE: usize = 500;
const SHAPE_K: usize = 2;
const PHASE_NOISE_K: usize = 50;
const N_ALPHAS: usize = 25;
const ALPHA_LOG_MIN: f64 = -4.0;
const ALPHA_LOG_MAX: f64 = 4.0;

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

fn logspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| pow(10.0_f64, lo + (hi - lo) * i as f64 / (n - 1) as f64)).collect()
}

fn slice_mean(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }

fn median_f64(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = v.len();
    if n % 2 == 0 { (v[n / 2 - 1] + v[n / 2]) / 2.0 } else { v[n / 2] }
}

// ── Ridge with Soft-Average GCV ────────────────────────────────────────────
//
// Returns (beta [nf], loo_residuals [n_train], gcv_min).
// x_rows: row-major design matrix (n_train rows, each of length nf).

fn ridge_sa(x_rows: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
    let m = x_rows.len();
    let nf = x_rows[0].len();

    let (u, s, vt) = svd::full(x_rows).unwrap_or_default();
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

    // LOO residuals
    let residuals: Vec<f64> = (0..m).map(|i| {
        y[i] - x_rows[i].iter().zip(beta.iter()).map(|(&xi, &bi)| xi * bi).sum::<f64>()
    }).collect();
    let h_avg: Vec<f64> = (0..m).map(|i| (0..k).map(|j| u[i][j] * u[i][j] * d_avg[j]).sum()).collect();
    let loo: Vec<f64> = residuals.iter().zip(h_avg.iter())
        .map(|(&ri, &hi)| ri / (1.0 - hi).max(EPS))
        .collect();

    (beta, loo, gcv_min)
}

// ── Frequency helpers ──────────────────────────────────────────────────────

/// Normalise a pandas-style frequency string.
/// "MIN" → "T", anchored offsets "W-SUN" / "Q-DEC" / "A-DEC" → "W" / "Q" / "A".
fn resolve_freq(freq: &str) -> String {
    let f = freq.trim().to_uppercase().replace("MIN", "T");
    for base in ["W", "Q", "A", "Y"] {
        if f.starts_with(&format!("{base}-")) {
            return String::from(base);
        }
    }
    f
}

fn get_period(freq: &str) -> usize {
    match resolve_freq(freq).as_str() {
        "S"         => 60,
        "T" | "MIN" => 60,
        "5T"        => 12,
        "10T"       => 6,
        "15T"       => 4,
        "10S"       => 6,
        "H"         => 24,
        "D"         => 7,
        "W"         => 52,
        "M"         => 12,
        "Q"         => 4,
        "A" | "Y"   => 1,
        _           => 1,
    }
}

fn get_periods(freq: &str) -> Vec<usize> {
    match resolve_freq(freq).as_str() {
        "10S" => vec![6, 360],
        "S"   => vec![60],
        "5T"  => vec![12, 288],
        "10T" => vec![6, 144],
        "15T" => vec![4, 96],
        "H"   => vec![24, 168],
        "D"   => vec![7, 365],
        "W"   => vec![52],
        "M"   => vec![12],
        "Q"   => vec![4],
        _     => vec![],
    }
}

// ── Period selection ───────────────────────────────────────────────────────

/// Returns (P, secondary_periods, primary_period, calendar_periods).
fn select_period(y: &[f64], n: usize, freq: &str) -> (usize, Vec<usize>, usize, Vec<usize>) {
    let period = get_period(freq);
    let cal = get_periods(freq);

    let mut candidates: Vec<usize> = if !cal.is_empty() {
        cal.iter().copied().filter(|&p| p >= 1 && n / p >= MIN_COMPLETE).collect()
    } else {
        vec![]
    };
    if candidates.is_empty() {
        let p = period.max(1);
        candidates.push(if n / p >= MIN_COMPLETE { p } else { 1 });
    }

    let big_p = if candidates.len() == 1 {
        candidates[0]
    } else {
        let min_cand = *candidates.iter().min().unwrap();
        let t_max = n.min(MAX_COMPLETE * min_cand);
        let y_sel = &y[y.len().saturating_sub(t_max)..];
        let mut best_p = candidates[0];
        let mut best_bic = f64::INFINITY;

        for &p_cand in &candidates {
            let nc = t_max / p_cand;
            if nc < MIN_COMPLETE { continue; }
            let start = y_sel.len() - nc * p_cand;
            let y_use = &y_sel[start..];
            // mat_c[ph, ci] = y_use[ci * p_cand + ph]  (shape: p_cand × nc)
            let mat_c: Vec<Vec<f64>> = (0..p_cand).map(|ph| (0..nc).map(|ci| y_use[ci * p_cand + ph]).collect()).collect();
            let s = svd::singvals(&mat_c);
            let rss1: f64 = s.iter().skip(1).map(|&v| v * v).sum();
            let t = (nc * p_cand) as f64;
            let bic = t * ln((rss1 / t).max(EPS_LOG)) + (p_cand + nc - 1) as f64 * ln(t);
            if bic < best_bic { best_bic = bic; best_p = p_cand; }
        }
        best_p
    };

    let secondary: Vec<usize> = cal.iter().copied().filter(|&p| p > big_p).collect();
    (big_p, secondary, period, cal)
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

// ── Shape estimation (Dirichlet-Multinomial empirical Bayes) ───────────────
//
// mat: [P][n_complete]  – phase × period matrix
// l:   [n_complete]     – period-level aggregation
// Returns (S_forecast [m][P], S_hist [n_complete][P], m).

fn estimate_shape(
    mat: &[Vec<f64>],
    n_complete: usize,
    big_p: usize,
    secondary: &[usize],
    l: &[f64],
    horizon: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, usize) {
    let k = SHAPE_K.min(n_complete);
    // S_global[ph] = mean of (mat[ph][ci] / sum_ph mat[ph][ci]) over last k periods
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

    // Context window C (secondary / P)
    let c = secondary.first().copied().and_then(|sp| {
        if sp % big_p == 0 && n_complete >= sp / big_p { Some(sp / big_p) } else { None }
    }).unwrap_or(1);

    let m = (horizon + big_p - 1) / big_p; // ceil(horizon / P)

    if c <= 1 {
        return (vec![s_global.clone(); m], vec![s_global.clone(); n_complete], m);
    }

    // Context-dependent Dirichlet-Multinomial
    let k_ds = (k * c).min(n_complete);
    let ds_start = n_complete - k_ds;
    let ds_mat: Vec<Vec<f64>> = (0..big_p).map(|ph| mat[ph][ds_start..].to_vec()).collect();
    let ds_l = &l[ds_start..];
    let ds_ctx: Vec<usize> = (ds_start..n_complete).map(|i| i % c).collect();

    let ds_totals: Vec<f64> = (0..k_ds).map(|ci| (0..big_p).map(|ph| ds_mat[ph][ci]).sum::<f64>()).collect();
    let ds_props: Vec<Vec<f64>> = (0..big_p).map(|ph| {
        (0..k_ds).map(|ci| {
            if ds_totals[ci] > EPS { ds_mat[ph][ci] / ds_totals[ci] } else { 1.0 / big_p as f64 }
        }).collect()
    }).collect();

    let mp: Vec<f64> = (0..big_p).map(|ph| slice_mean(&ds_props[ph])).collect();
    let vp: Vec<f64> = (0..big_p).map(|ph| {
        let m = mp[ph];
        ds_props[ph].iter().map(|&v| pow(v - m, 2.0)).sum::<f64>() / (k_ds.max(2) - 1) as f64
    }).collect();

    let valid_kappas: Vec<f64> = (0..big_p)
        .filter(|&ph| mp[ph] > EPS_SHAPE && vp[ph] > EPS)
        .map(|ph| mp[ph] * (1.0 - mp[ph]) / vp[ph] - 1.0)
        .collect();
    let kappa = if valid_kappas.len() >= 2 {
        median_f64(valid_kappas).max(0.0)
    } else {
        1e6
    };

    let mut s_ctx: Vec<Vec<f64>> = vec![s_global.clone(); c];
    for c_val in 0..c {
        let mask: Vec<usize> = ds_ctx.iter().enumerate()
            .filter(|&(_, &cv)| cv == c_val).map(|(i, _)| i).collect();
        if mask.is_empty() { continue; }
        let l_sum: f64 = mask.iter().map(|&i| ds_l[i]).sum();
        let denom = (kappa + l_sum).max(EPS);
        let mut s_c: Vec<f64> = (0..big_p).map(|ph| {
            (kappa * s_global[ph] + mask.iter().map(|&i| ds_mat[ph][i]).sum::<f64>()) / denom
        }).collect();
        let sc_sum = s_c.iter().sum::<f64>().max(EPS);
        s_c.iter_mut().for_each(|v| *v /= sc_sum);
        s_ctx[c_val] = s_c;
    }

    let s_forecast: Vec<Vec<f64>> = (0..m).map(|j| s_ctx[(n_complete + j) % c].clone()).collect();
    let s_hist: Vec<Vec<f64>> = (0..n_complete).map(|i| s_ctx[i % c].clone()).collect();
    (s_forecast, s_hist, m)
}

// ── Seasonal strength: γ-dampening ────────────────────────────────────────

/// γ = (r₁ − r_rand) / (1 − r_rand), where r₁ = σ₁²/Σσᵢ².
/// Returns γ ∈ [0, 1].  P < 2 or too-short series → 1.0 (no dampening).
fn estimate_gamma(mat: &[Vec<f64>], big_p: usize, n_complete: usize) -> f64 {
    if big_p < 2 || n_complete < MIN_COMPLETE {
        return 1.0;
    }
    let x: Vec<Vec<f64>> = (0..big_p).map(|r| (0..n_complete).map(|c| mat[r][c]).collect()).collect();
    let s = svd::singvals(&x);
    let total: f64 = s.iter().map(|&v| v * v).sum();
    if total < EPS_LOG {
        return 1.0;
    }
    let r1 = s[0] * s[0] / total;
    let r_rand = 1.0 / big_p.min(n_complete) as f64;
    ((r1 - r_rand) / (1.0 - r_rand).max(EPS)).clamp(0.0, 1.0)
}

/// S^γ with re-normalisation.  S is (n_rows × P); γ=1 returns S unchanged.
fn dampen_shape(s: &mut Vec<Vec<f64>>, gamma: f64) {
    if gamma >= 1.0 - EPS {
        return;
    }
    for row in s.iter_mut() {
        for v in row.iter_mut() {
            *v = pow(v.max(EPS_LOG), gamma);
        }
        let sum: f64 = row.iter().sum();
        let sum = sum.max(EPS);
        row.iter_mut().for_each(|v| *v /= sum);
    }
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

// ── Main forecast function ─────────────────────────────────────────────────

/// Generate probabilistic forecast sample paths.
///
/// # Arguments
/// * `y_raw`     – historical time series (1-D slice)
/// * `horizon`   – steps ahead to forecast (>= 1)
/// * `freq`      – frequency string: `"H"`, `"D"`, `"W"`, `"M"`, `"Q"`,
///                 `"S"`, `"T"`, `"5T"`, `"10T"`, `"15T"`, `"A"` / `"Y"`
/// * `n_samples` – number of Monte-Carlo paths (>= 1)
/// * `seed`      – RNG seed (use any `u64`; same seed → identical output)
///
/// # Returns
/// `Ok(Vec<Vec<f64>>)` of shape `[n_samples][horizon]`, or an error string.
pub fn forecast(
    y_raw: &[f64],
    horizon: usize,
    freq: &str,
    n_samples: usize,
    seed: u64,
) -> Result<Vec<Vec<f64>>, String> {
    if horizon < 1   { return Err(format!("horizon must be >= 1, got {horizon}")); }
    if n_samples < 1 { return Err(format!("n_samples must be >= 1, got {n_samples}")); }
    if y_raw.is_empty() { return Err("y must not be empty".into()); }

    let mut rng = Rng::new(seed);

    // NaN-to-zero + shift so all values >= 1
    let mut y: Vec<f64> = y_raw.iter().map(|&v| if v.is_nan() { 0.0 } else { v }).collect();
    let y_floor = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_shift = (1.0 - y_floor).max(1.0);
    y.iter_mut().for_each(|v| *v += y_shift);
    let n = y.len();

    // ── Period selection ────────────────────────────────────────────────
    let (mut big_p, mut secondary, period, _cal) = select_period(&y, n, freq);
    let mut n_complete = n / big_p;

    // Fallback for too-short series
    if n_complete < MIN_COMPLETE {
        if big_p > 1 { big_p = 1; secondary.clear(); n_complete = n; }
        if n_complete < MIN_COMPLETE {
            let fc_val = y[n - 1] - y_shift;
            let lookback = PHASE_NOISE_K.min(n);
            let diffs: Vec<f64> = y[n - lookback..].windows(2).map(|w| w[1] - w[0]).collect();
            let sigma = if diffs.is_empty() { EPS_SHAPE } else {
                let m = slice_mean(&diffs);
                sqrt(diffs.iter().map(|&d| pow(d - m, 2.0)).sum::<f64>() / diffs.len() as f64)
                    .max(EPS_SHAPE)
            };
            return Ok((0..n_samples).map(|_| {
                (0..horizon).map(|_| {
                    let v = fc_val + rng.normal() * sigma;
                    v.clamp(fc_val - sigma * 10.0, fc_val + sigma * 10.0)
                }).collect()
            }).collect());
        }
    }

    // Cap history to MAX_COMPLETE periods
    if n_complete > MAX_COMPLETE {
        y = y[y.len() - MAX_COMPLETE * big_p..].to_vec();
        n_complete = MAX_COMPLETE;
    }

    let usable = n_complete * big_p;
    let y_trim = &y[y.len() - usable..];

    // ── Matrix reshape: mat[ph][ci] = y_trim[ci*P + ph] ────────────────
    let mat: Vec<Vec<f64>> = (0..big_p)
        .map(|ph| (0..n_complete).map(|ci| y_trim[ci * big_p + ph]).collect())
        .collect();
    // Period-level aggregation
    let l: Vec<f64> = (0..n_complete)
        .map(|ci| (0..big_p).map(|ph| mat[ph][ci]).sum())
        .collect();

    // ── Shape estimation ────────────────────────────────────────────────
    let (mut s_forecast, mut s_hist, m) = estimate_shape(&mat, n_complete, big_p, &secondary, &l, horizon);

    // ── Seasonal strength: dampen Shape when rank-1 structure is weak ───
    let gamma = estimate_gamma(&mat, big_p, n_complete);
    dampen_shape(&mut s_forecast, gamma);
    dampen_shape(&mut s_hist, gamma);

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

    let x_rows: Vec<Vec<f64>> = (start..n_complete).map(|ti| {
        let mut row = vec![0.0f64; nf];
        row[0] = 1.0;
        row[1] = ti as f64 / n_complete as f64;
        row[nb] = l_innov[ti - 1];
        if max_cp >= 2 { row[nb + 1] = l_innov[ti - max_cp]; }
        row
    }).collect();

    let (beta, loo_resid, _) = ridge_sa(&x_rows, &l_innov[start..]);

    // ── Stochastic Level paths ──────────────────────────────────────────
    let loo_len = loo_resid.len();
    // noise_pool[s][j] = random LOO residual for sample s at period step j
    let noise_pool: Vec<Vec<f64>> = (0..n_samples)
        .map(|_| (0..m).map(|_| loo_resid[rng.randint(loo_len)]).collect())
        .collect();

    // L_paths[s][0..n_complete] = l_innov (history), [n_complete..] = forecast
    let total = n_complete + m;
    let mut l_paths: Vec<Vec<f64>> = (0..n_samples).map(|_| {
        let mut v = l_innov.clone();
        v.resize(total, 0.0);
        v
    }).collect();

    for j in 0..m {
        let ti = n_complete + j;
        for si in 0..n_samples {
            let pred = beta[0]
                + beta[1] * (ti as f64 / n_complete as f64)
                + beta[nb] * l_paths[si][ti - 1];
            let pred = if max_cp >= 2 { pred + beta[nb + 1] * l_paths[si][ti - max_cp] } else { pred };
            l_paths[si][ti] = pred + noise_pool[si][j];
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
    // fitted_mat[ph][ci] = s_hist[ci][ph] * l[ci]
    // R[ph][kr_idx] = relative residual over last k_r periods
    let k_r = PHASE_NOISE_K.min(n_complete);
    let r_mat: Vec<Vec<f64>> = (0..big_p).map(|ph| {
        (n_complete - k_r..n_complete).map(|ci| {
            let fitted = s_hist[ci][ph] * l[ci];
            (mat[ph][ci] - fitted) / fitted.abs().max(EPS_BOXCOX)
        }).collect()
    }).collect();

    // col_idx[s][j] = random column in r_mat for (sample s, period step j)
    let col_idx: Vec<Vec<usize>> = (0..n_samples)
        .map(|_| (0..m).map(|_| rng.randint(k_r)).collect())
        .collect();

    // ── Assemble output ─────────────────────────────────────────────────
    let step_idx: Vec<usize> = (0..horizon).map(|h| h / big_p).collect();
    let phase_idx: Vec<usize> = (0..horizon).map(|h| h % big_p).collect();

    let mut samples: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    for si in 0..n_samples {
        let path: Vec<f64> = (0..horizon).map(|h| {
            let sj = step_idx[h];
            let ph = phase_idx[h];
            let phase_noise = r_mat[ph][col_idx[si][sj]];
            l_hat_all[si][sj] * s_forecast[sj][ph] * (1.0 + phase_noise) - y_shift
        }).collect();
        samples.push(path);
    }

    // Clip outliers to ±1 range around recent history
    let lookback = (horizon * 2).max(PHASE_NOISE_K).min(y_raw.len());
    let y_lo = y_raw[y_raw.len() - lookback..].iter().cloned().fold(f64::INFINITY, f64::min);
    let y_hi = y_raw[y_raw.len() - lookback..].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_range = (y_hi - y_lo).max(EPS_SHAPE);
    for path in &mut samples {
        for v in path.iter_mut() {
            if !v.is_finite() { *v = 0.0; }
            else { *v = v.clamp(y_lo - y_range, y_hi + y_range); }
        }
    }

    Ok(samples)
}

/// Quantile forecast: returns `[quantile][horizon]` for each q in `quantiles`.
///
/// `quantiles` must all be in [0, 1].  Returns an error if any are out of range.
pub fn forecast_quantiles(
    y: &[f64],
    horizon: usize,
    freq: &str,
    n_samples: usize,
    seed: u64,
    quantiles: &[f64],
) -> Result<Vec<Vec<f64>>, String> {
    if let Some(&q) = quantiles.iter().find(|&&q| !(0.0..=1.0).contains(&q)) {
        return Err(format!("quantile {q} out of range [0, 1]"));
    }
    let samples = forecast(y, horizon, freq, n_samples, seed)?;
    let ns = samples.len();
    Ok(quantiles.iter().map(|&q| {
        (0..horizon).map(|h| {
            let mut col: Vec<f64> = samples.iter().map(|s| s[h]).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let idx = round(q * (ns - 1) as f64) as usize;
            col[idx]
        }).collect()
    }).collect())
}

/// Point forecast: mean over `n_samples` paths.
pub fn forecast_mean(
    y: &[f64],
    horizon: usize,
    freq: &str,
    n_samples: usize,
    seed: u64,
) -> Result<Vec<f64>, String> {
    let samples = forecast(y, horizon, freq, n_samples, seed)?;
    let n = samples.len() as f64;
    Ok((0..horizon).map(|h| samples.iter().map(|s| s[h]).sum::<f64>() / n).collect())
}

// ── Confidence report ──────────────────────────────────────────────────────

/// Self-evaluation of how well FLAIR's assumptions fit the input series.
/// All fields are derived solely from `y` and `freq` — no forecast is run.
#[derive(Debug, Clone)]
pub struct Confidence {
    /// `s[0]^2 / Σs^2` of the seasonal matrix (period × n_complete).
    /// Measures how close the data is to rank-1 structure.
    /// 1.0 = perfect single-component seasonality; ~1/P = flat/random.
    ///
    /// `None` in two distinct cases:
    /// - `freq` has no intra-period structure (`"A"` / `"Y"`, period = 1) —
    ///   FLAIR skips seasonal decomposition and runs Level-only AR; this is
    ///   expected and not a sign of poor fit.
    /// - Series is too short to form `MIN_COMPLETE` complete periods —
    ///   seasonal decomposition is unreliable regardless of freq.
    pub rank1: Option<f64>,

    /// Seasonal strength after removing random-matrix baseline.
    /// `(r1 - 1/min(P,N)) / (1 - 1/min(P,N))`, clamped to [0, 1].
    /// 1.0 = strong clean seasonality; 0.0 = no detectable seasonal structure.
    /// `None` in the same cases as `rank1`.
    pub gamma: Option<f64>,

    /// GCV-optimal LOO error from the Ridge regression on the Level series.
    /// Lower = Level is more predictable from its own history.
    /// Scale is that of the Box-Cox-transformed, mean-subtracted Level.
    /// `None` when the series is too short to fit the Ridge model.
    pub gcv: Option<f64>,

    /// `true` if the core numerical primitives (Box-Cox round-trip and Ridge
    /// in-sample fit) pass their internal sanity checks on synthetic data.
    /// Computed once per `confidence()` call at negligible cost.
    /// A `false` here indicates a build or platform numerical issue.
    pub impl_ok: bool,
}

/// Evaluate how well FLAIR's assumptions fit `y` without running a forecast.
///
/// Returns a [`Confidence`] struct with three independent scores derived
/// purely from the input.  Use this to decide whether a forecast is likely
/// to be meaningful before committing to `n_samples` Monte-Carlo paths.
pub fn confidence(y_raw: &[f64], freq: &str) -> Confidence {
    let n = y_raw.len();

    // Apply the same shift as forecast() so all values > 0
    let y_floor = y_raw.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_shift = (1.0 - y_floor).max(1.0);
    let y: Vec<f64> = y_raw.iter().map(|&v| (if v.is_nan() { 0.0 } else { v }) + y_shift).collect();

    let (big_p, _, _, _) = select_period(&y, n, freq);
    let n_complete = n / big_p;

    if n_complete < MIN_COMPLETE || big_p < 2 {
        // Series too short for seasonal decomposition — try Ridge only
        let gcv = ridge_gcv_only(&y);
        return Confidence { rank1: None, gamma: None, gcv, impl_ok: check_impl() };
    }

    let usable = n_complete * big_p;
    let y_trim = &y[n - usable..];
    let mat_dm: Vec<Vec<f64>> = (0..big_p).map(|ph| (0..n_complete).map(|ci| y_trim[ci * big_p + ph]).collect()).collect();
    let s = svd::singvals(&mat_dm);
    let total: f64 = s.iter().map(|&v| v * v).sum();

    let (rank1, gamma) = if total < EPS {
        (None, None)
    } else {
        let r1 = s[0] * s[0] / total;
        let r_rand = 1.0 / big_p.min(n_complete) as f64;
        let g = ((r1 - r_rand) / (1.0 - r_rand).max(EPS)).clamp(0.0, 1.0);
        (Some(r1), Some(g))
    };

    // Level series → Box-Cox → Ridge GCV
    let l: Vec<f64> = (0..n_complete)
        .map(|ci| (0..big_p).map(|ph| y_trim[ci * big_p + ph]).sum())
        .collect();
    let lam = bc_lambda(&l);
    let l_bc = bc(&l, lam);
    let last_l = l_bc[n_complete - 1];
    let l_innov: Vec<f64> = l_bc.iter().map(|&v| v - last_l).collect();

    let gcv = if n_complete > 2 {
        let x_rows: Vec<Vec<f64>> = (1..n_complete)
            .map(|ti| vec![1.0, ti as f64 / n_complete as f64, l_innov[ti - 1]])
            .collect();
        let (_, _, gcv_min) = ridge_sa(&x_rows, &l_innov[1..]);
        Some(gcv_min)
    } else {
        None
    };

    Confidence { rank1, gamma, gcv, impl_ok: check_impl() }
}

/// Minimal Ridge GCV for very short series (no seasonal decomposition).
fn ridge_gcv_only(y: &[f64]) -> Option<f64> {
    let n = y.len();
    if n < 3 { return None; }
    let x_rows: Vec<Vec<f64>> = (1..n)
        .map(|i| vec![1.0, i as f64 / n as f64, y[i - 1]])
        .collect();
    let (_, _, gcv_min) = ridge_sa(&x_rows, &y[1..]);
    Some(gcv_min)
}

/// Sanity-check the core numerical primitives (Box-Cox and Ridge) on synthetic
/// data.  Returns `true` if both pass, `false` on any numerical failure.
fn check_impl() -> bool {
    // Box-Cox round-trip
    let orig = [0.5f64, 1.0, 2.0, 5.0, 10.0];
    for &lam in &[0.0f64, 0.5, 1.0] {
        let recovered = bc_inv(&bc(&orig, lam), lam);
        if orig.iter().zip(recovered.iter()).any(|(&o, &r)| (o - r).abs() > 1e-9) {
            return false;
        }
    }
    // Ridge in-sample fit on perfect linear data
    let x_rows: Vec<Vec<f64>> = (0..30).map(|i| vec![1.0, i as f64 / 30.0]).collect();
    let y_lin: Vec<f64> = x_rows.iter().map(|r| 2.0 + 3.0 * r[1]).collect();
    let (beta, _, _) = ridge_sa(&x_rows, &y_lin);
    let rmse = sqrt(x_rows.iter().zip(y_lin.iter())
        .map(|(r, &yi)| {
            let pred: f64 = r.iter().zip(beta.iter()).map(|(&xi, &bi)| xi * bi).sum();
            pow(yi - pred, 2.0)
        })
        .sum::<f64>() / 30.0);
    rmse < 0.1
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::fs;

    #[test]
    fn impl_ok_passes() {
        assert!(check_impl(), "core numerical primitives failed sanity check");
    }

    #[test]
    fn output_shape() {
        let y: Vec<f64> = (0..200).map(|i| sin(i as f64 * 0.26) * 3.0 + 10.0).collect();
        let s = forecast(&y, 12, "M", 50, 0).unwrap();
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
        assert!(forecast(&[], 5, "H", 10, 0).is_err());
        assert!(forecast(&[1.0, 2.0], 0, "H", 10, 0).is_err());
        assert!(forecast(&[1.0, 2.0], 5, "H", 0, 0).is_err());
    }

    #[test]
    fn forecast_mean_shape() {
        let y: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let m = forecast_mean(&y, 7, "D", 20, 5).unwrap();
        assert_eq!(m.len(), 7);
        assert!(m.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn forecast_quantiles_shape_and_order() {
        let y: Vec<f64> = (0..200).map(|i| sin(i as f64 * 0.26) * 3.0 + 10.0).collect();
        let qs = [0.1, 0.5, 0.9];
        let q = forecast_quantiles(&y, 12, "M", 100, 0, &qs).unwrap();
        assert_eq!(q.len(), 3);
        assert!(q.iter().all(|row| row.len() == 12));
        for h in 0..12 {
            assert!(q[0][h] <= q[1][h] && q[1][h] <= q[2][h]);
        }
    }

    #[test]
    fn forecast_quantiles_invalid_q() {
        let y: Vec<f64> = (0..50).map(|i| i as f64).collect();
        assert!(forecast_quantiles(&y, 5, "M", 10, 0, &[0.5, 1.5]).is_err());
    }

    // ── dataset-iter tests ────────────────────────────────────────────────
    //
    // Each entry in DATASETS describes a CSV in examples/dataset/ and how to
    // parse it.  The same suite of checks runs over every dataset:
    //   - forecast() and confidence() complete without error
    //   - all output values are finite
    //   - impl_ok is true
    //
    // ParseMode:
    //   Col(n)      – take the n-th comma-separated column (0-based), skip 1 header row
    //   ColSkip(n,s)– same but skip s header rows
    //   JapanTokyo  – japan_demand.csv: col 4 (Tokyo), skip 1 header row
    //   WorldBank   – world_bank.csv: find "Japan" row, extract numeric cols

    #[allow(dead_code)]
    enum ParseMode { Col(usize), ColSkip(usize, usize), JapanTokyo, WorldBank }

    struct Dataset {
        file: &'static str,
        freq: &'static str,
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
                    .filter_map(|l| l.split(',').nth(4)?.trim().parse::<f64>().ok())
                    .collect()
            }
            ParseMode::WorldBank => {
                let line = content.lines()
                    .find(|l| l.contains("\"Japan\""))
                    .expect("Japan row not found in world_bank.csv");
                line.split(',').skip(4)
                    .filter_map(|v| {
                        let s = v.trim().trim_matches('"');
                        if s.is_empty() { None } else { s.parse::<f64>().ok() }
                    })
                    .collect()
            }
        }
    }

    fn datasets() -> Vec<Dataset> {
        vec![
            Dataset { file: "air_passengers.csv",  freq: "M",  mode: ParseMode::Col(2) },
            Dataset { file: "nottem.csv",           freq: "M",  mode: ParseMode::Col(2) },
            Dataset { file: "sunspot_year.csv",     freq: "A",  mode: ParseMode::Col(2) },
            Dataset { file: "noaa_temp_annual.csv", freq: "A",  mode: ParseMode::Col(1) },
            Dataset { file: "noaa_temp_monthly.csv",freq: "M",  mode: ParseMode::Col(1) },
            Dataset { file: "world_bank.csv",       freq: "A",  mode: ParseMode::WorldBank },
            Dataset { file: "japan_demand.csv",     freq: "H",  mode: ParseMode::JapanTokyo },
        ]
    }

    #[test]
    fn dataset_iter_no_crash() {
        for ds in datasets() {
            let y = load(&ds);
            assert!(!y.is_empty(), "{}: empty", ds.file);

            let c = confidence(&y, ds.freq);
            assert!(c.impl_ok, "{}: impl_ok false", ds.file);

            let fc = forecast_mean(&y, 12, ds.freq, 30, 0)
                .unwrap_or_else(|e| panic!("{}: forecast error: {e}", ds.file));
            assert_eq!(fc.len(), 12, "{}: wrong horizon", ds.file);
            assert!(
                fc.iter().all(|v| v.is_finite()),
                "{}: non-finite output", ds.file
            );
        }
    }
}
