//! Gavish-Donoho 2014 optimal Frobenius shrinkage for rank-1 Level.
//!
//! Reference: Gavish & Donoho (2014) "The Optimal Hard Threshold for Singular
//! Values is 4/sqrt(3)", IEEE Trans. Inf. Theory 60(8), 5040-5053.
//!
//! `optshrink_factor(svd_s, P, n_complete)` returns a scalar `c in (0, 1]`
//! such that `L * c` is the minimax-optimal rank-1 Level estimate under the
//! spiked rectangular model. Falls back to `1.0` when the spectrum is
//! degenerate or the top singular value is subcritical.
//!
//! The Marchenko-Pastur median `mu_beta` is computed via numerical integration
//! over the MP support using `double_exponential::integrate` and a Brent solver.

use libm::{sqrt, pow, fabs};
use crate::{
    constants::{EPS, EPS_BOXCOX},
    double_exponential::integrate,
};

/// Returns the median of the Marchenko-Pastur distribution at aspect ratio `beta`.
///
/// Returns the median of the Marchenko-Pastur distribution at aspect ratio `beta`.
///
/// PDF: f(x) = sqrt((y_plus - x)(x - y_minus)) / (2*pi*beta*x),
/// support [y_minus, y_plus] where y_pm = (1 +/- sqrt(beta))^2.
/// Solved numerically: integrate f from y_minus to m, find m where CDF(m) = 0.5.
pub fn mp_median(beta: f64) -> f64 {
    if beta <= EPS {
        return 1.0;
    }
    let b = beta.min(1.0);
    let sb = sqrt(b);
    let y_minus = (1.0 - sb) * (1.0 - sb);
    let y_plus  = (1.0 + sb) * (1.0 + sb);

    let mp_cdf = |m: f64| -> f64 {
        if m <= y_minus { return 0.0; }
        if m >= y_plus  { return 1.0; }
        let result = integrate(
            |x| {
                let num = (y_plus - x) * (x - y_minus);
                if num <= 0.0 { 0.0 } else { sqrt(num) / (x * b) }
            },
            y_minus,
            m,
            1e-8,
        );
        // Normalization: total integral of sqrt(...) / (x*b) over support equals 2*pi.
        (result.integral / (2.0 * core::f64::consts::PI)).clamp(0.0, 1.0)
    };

    brent_solve(mp_cdf, y_minus + EPS, y_plus - EPS, 1e-10)
        .unwrap_or((y_minus + y_plus) * 0.5)
}

/// Brent root-finding: solves f(x) = 0.5 given f(xa)*f(xb) < 0.
fn brent_solve<F: Fn(f64) -> f64>(f: F, xa: f64, xb: f64, xtol: f64) -> Option<f64> {
    let target = 0.5;
    let g = |x| f(x) - target;
    let mut a = xa;
    let mut b = xb;
    let mut fa = g(a);
    let mut fb = g(b);
    if fa * fb > 0.0 { return None; }
    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;
    for _ in 0..500 {
        if fb * fc > 0.0 { c = a; fc = fa; d = b - a; e = d; }
        if fabs(fc) < fabs(fb) { a = b; b = c; c = a; fa = fb; fb = fc; fc = fa; }
        let tol = 2.0 * f64::EPSILON * fabs(b) + 0.5 * xtol;
        let m = 0.5 * (c - b);
        if fabs(m) <= tol || fb == 0.0 { return Some(b); }
        if fabs(e) >= tol && fabs(fa) > fabs(fb) {
            let s = fb / fa;
            let (p, q) = if a == c {
                (2.0 * m * s, 1.0 - s)
            } else {
                let q = fa / fc;
                let r = fb / fc;
                (s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0)),
                 (q - 1.0) * (r - 1.0) * (s - 1.0))
            };
            let (p, q) = if p > 0.0 { (p, -q) } else { (-p, q) };
            if 2.0 * p < (3.0 * m * q - fabs(tol * q)).min(fabs(e * q)) {
                e = d; d = p / q;
            } else {
                d = m; e = m;
            }
        } else {
            d = m; e = m;
        }
        a = b; fa = fb;
        b += if fabs(d) > tol { d } else if m > 0.0 { tol } else { -tol };
        fb = g(b);
    }
    Some(b)
}

/// Gavish-Donoho optimal Frobenius shrinkage factor.
///
/// `svd_s`: singular values of the (P × n_complete) period-folded matrix,
///          descending order (from `svd::singvals`).
/// `big_p`: number of rows (P).
/// `nc`:    number of columns (n_complete).
///
/// Returns `c` in `(0, 1]`. Multiply L by this factor before the Level Ridge.
pub fn optshrink_factor(svd_s: &[f64], big_p: usize, nc: usize) -> f64 {
    if svd_s.len() < 2 || big_p.min(nc) < 2 {
        return 1.0;
    }
    let sigma_1 = svd_s[0];
    if sigma_1 < EPS {
        return 1.0;
    }
    let sigma_med = {
        let mut s = svd_s.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let n = s.len();
        if n % 2 == 0 { (s[n / 2 - 1] + s[n / 2]) / 2.0 } else { s[n / 2] }
    };
    if sigma_med < EPS {
        return 1.0;
    }

    let beta = (big_p.min(nc) as f64) / (big_p.max(nc) as f64);
    let mu_beta = mp_median(beta);
    let sigma_noise = sigma_med / sqrt(mu_beta);
    let threshold = (1.0 + sqrt(beta)) * sigma_noise;

    if sigma_1 <= threshold {
        return 1.0;
    }

    // Gavish-Donoho 2014 Corollary 1 / SIAM 2017 eq. 3.2:
    //   sigma* = (1/sqrt(2)) * sqrt(A + sqrt(A^2 - 4*beta*sigma_noise^4))
    let a = sigma_1 * sigma_1 - (1.0 + beta) * sigma_noise * sigma_noise;
    let disc = a * a - 4.0 * beta * pow(sigma_noise, 4.0);
    if disc <= 0.0 {
        return 1.0;
    }
    let sigma_star = sqrt(a + sqrt(disc)) / sqrt(2.0_f64);
    (sigma_star / sigma_1).clamp(EPS_BOXCOX, 1.0)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Reference values pre-computed via Python flaircast 0.6.1 `_mp_median`.
    #[test]
    fn mp_median_known_betas() {
        // (beta, expected_mu, tol)
        let cases = [
            (0.01_f64, 0.99667, 5e-3),
            (0.1,      0.96657, 5e-3),
            (0.25,     0.91600, 5e-3),
            (0.5,      0.83047, 5e-3),
            (1.0,      0.65278, 5e-3),
        ];
        for (beta, expected, tol) in cases {
            let got = mp_median(beta);
            assert!(
                (got - expected).abs() < tol,
                "mp_median({beta}) = {got:.5}, expected ~{expected:.5} (tol={tol})"
            );
        }
    }

    /// mp_median must always lie within the MP support [(1-sqrt(beta))^2, (1+sqrt(beta))^2].
    #[test]
    fn mp_median_in_support() {
        for i in 1..=20usize {
            let beta = i as f64 * 0.05;
            let mu = mp_median(beta);
            let y_minus = pow(1.0 - sqrt(beta.min(1.0)), 2.0);
            let y_plus  = pow(1.0 + sqrt(beta.min(1.0)), 2.0);
            assert!(mu > y_minus && mu < y_plus,
                "beta={beta}: mu={mu} not in ({y_minus}, {y_plus})");
        }
    }

    /// Degenerate inputs must return 1.0.
    #[test]
    fn optshrink_degenerate() {
        assert_eq!(optshrink_factor(&[], 10, 10), 1.0);
        assert_eq!(optshrink_factor(&[5.0], 10, 10), 1.0);   // len < 2
        assert_eq!(optshrink_factor(&[5.0, 1.0], 1, 10), 1.0); // min(P,nc) < 2
        assert_eq!(optshrink_factor(&[0.0, 0.0], 5, 5), 1.0);  // sigma_1 = 0
    }

    /// When the top singular value is at or below the noise threshold, return 1.0.
    #[test]
    fn optshrink_subcritical_returns_one() {
        let s = vec![1.01, 1.0, 0.99, 0.98];
        assert_eq!(optshrink_factor(&s, 4, 4), 1.0);
    }

    /// A dominant rank-1 signal must yield factor in (0, 1).
    #[test]
    fn optshrink_strong_signal() {
        let s = vec![100.0, 1.2, 1.0, 0.9, 0.8];
        let f = optshrink_factor(&s, 5, 50);
        assert!(f > 0.0 && f < 1.0, "expected factor in (0,1), got {f}");
    }

    /// optshrink_factor agrees between nalgebra SVD and flair SVD on a rank-1 dominant 12x10 matrix.
    /// The condition number s[0]/s[1] is ~230, so small singular values may differ by a few percent;
    /// we verify the factor (which uses s[0] and median(s)) rather than individual singular values.
    #[test]
    fn optshrink_nalgebra_svd() {
        use nalgebra::DMatrix;

        let p = 12usize;
        let nc = 10usize;
        let mut mat = vec![vec![0.0f64; nc]; p];
        for ph in 0..p {
            for ci in 0..nc {
                let signal = (1.0 + 0.1 * ph as f64) * (10.0 + ci as f64);
                let noise  = 0.3 * ((ph * nc + ci) as f64 * 1.7321 % 1.0 - 0.5);
                mat[ph][ci] = signal + noise;
            }
        }

        let dm = DMatrix::from_fn(p, nc, |r, c| mat[r][c]);
        let svd_na = dm.svd(false, false);
        let s_na: Vec<f64> = svd_na.singular_values.iter().map(|&v| v as f64).collect();
        let s_rs: Vec<f64> = crate::svd::svdvals(&mat);

        let rel0 = (s_na[0] - s_rs[0]).abs() / s_na[0];
        assert!(rel0 < 0.001, "s[0]: nalgebra={:.4} flair={:.4} rel={rel0:.4}", s_na[0], s_rs[0]);

        let f_na = optshrink_factor(&s_na, p, nc);
        let f_rs = optshrink_factor(&s_rs, p, nc);
        let diff = (f_na - f_rs).abs();
        assert!(diff < 0.05, "optshrink factor: nalgebra={f_na:.4} flair={f_rs:.4} diff={diff:.4}");

        assert!(f_na > 0.0 && f_na < 1.0, "nalgebra factor out of (0,1): {f_na}");
        assert!(f_rs > 0.0 && f_rs < 1.0, "flair factor out of (0,1): {f_rs}");
    }
}
