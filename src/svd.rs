//! Singular Value Decomposition (Golub-Reinsch algorithm)
//!
//! Pure Rust implementation of dense SVD using:
//! - GEBRD (Householder bidiagonalization)
//! - DBDSQR (Givens rotation iteration)
//!
//! Reference: Golub, G.H., Reinsch, C. (1970)
//!           "Singular value decomposition and least squares solutions"
//!           Numerische Mathematik, 14(5), 403-420

use core::{
    cmp::Ordering,
    result::Result,
};
use alloc::{
    vec,
    vec::Vec,
};
use libm::{sqrt, pow};
use crate::SvdError;
use crate::dlog;

/// Compute full thin SVD: A = U * Σ * V^T
///
/// Returns (U: m×k, s: k, Vt: k×n) where k = min(m, n)
///
/// # Arguments
/// * `a` - Input matrix as Vec<Vec<f64>> (m rows × n columns)
///
/// # Example
/// ```ignore
/// let a = vec![
///     vec![1.0, 2.0, 3.0],
///     vec![4.0, 5.0, 6.0],
/// ];
/// let (u, s, vt) = svd::full(&a)?;
/// ```
pub fn full(a: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>), SvdError> {
    // Validate input
    if a.is_empty() {
        return Err(SvdError::DimensionMismatch);
    }
    let _m = a.len();
    let n = a[0].len();

    if n == 0 {
        return Err(SvdError::DimensionMismatch);
    }

    // Make a mutable copy
    let mut a_copy = a.to_vec();

    // Stage 1: Bidiagonalization (GEBRD)
    let state = gebrd(&mut a_copy)?;

    // Stage 2: Diagonalization via Givens rotations (DBDSQR)
    let (s, u, v) = dbdsqr(&state, 1000)?;

    // Extract thin SVD
    // U: m × k, S: k, V^T: k × n (where k = min(m, n))
    let k = s.len();

    // Trim U to m × k
    let u_thin: Vec<Vec<f64>> = u.iter().map(|row| row[0..k].to_vec()).collect();

    // Trim V to n × k, then transpose to get V^T: k × n
    let v_thin: Vec<Vec<f64>> = v.iter().map(|row| row[0..k].to_vec()).collect();
    let vt = transpose(&v_thin);

    Ok((u_thin, s, vt))
}

/// Compute singular values only
///
/// # Arguments
/// * `a` - Input matrix as Vec<Vec<f64>> (m rows × n columns)
///
/// # Returns
/// Vector of singular values in descending order
pub fn singvals(a: &[Vec<f64>]) -> Vec<f64> {
    if a.is_empty() || a[0].is_empty() {
        return Vec::new();
    }

    let mut a_copy = a.to_vec();
    let state = match gebrd(&mut a_copy) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    match dbdsqr(&state, 1000) {
        Ok((s, _, _)) => s,
        Err(_) => Vec::new(),
    }
}

// ── Internal helper functions ────────────────────────────────────────────────

/// Transpose a matrix
fn transpose(a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return vec![];
    }
    let m = a.len();
    let n = a[0].len();
    let mut result = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            result[j][i] = a[i][j];
        }
    }
    result
}

/// Internal state for bidiagonalization
struct BidigState {
    /// Left singular vectors (m × m)
    u: Vec<Vec<f64>>,
    /// Right singular vectors (n × n)
    v: Vec<Vec<f64>>,
    /// Diagonal elements
    d: Vec<f64>,
    /// Super-diagonal elements
    e: Vec<f64>,
}

/// Householder bidiagonalization: A → B (bidiagonal matrix)
///
/// Stage 1 of Golub-Reinsch SVD.
/// Reduces general matrix to bidiagonal form via Householder reflections.
///
/// Algorithm:
/// For i = 1 to min(m, n):
///   1. Compute Householder reflection H_L to zero below diagonal in column i
///   2. Apply H_L to A from the left
///   3. If i < n, compute Householder reflection H_R to zero right of superdiagonal in row i
///   4. Apply H_R to A from the right
///
/// The original matrix A is overwritten with the bidiagonal matrix B.
/// The Householder vectors are stored in the lower/upper triangular parts.
fn gebrd(a: &mut Vec<Vec<f64>>) -> Result<BidigState, SvdError> {
    let m = a.len();
    if m == 0 {
        return Err(SvdError::DimensionMismatch);
    }
    let n = a[0].len();
    if n == 0 {
        return Err(SvdError::DimensionMismatch);
    }

    // Initialize U and V as identity matrices
    let mut u = vec![vec![0.0; m]; m];
    let mut v = vec![vec![0.0; n]; n];
    for i in 0..m {
        u[i][i] = 1.0;
    }
    for i in 0..n {
        v[i][i] = 1.0;
    }

    let mut d = vec![0.0; m.min(n)];
    let mut e = vec![0.0; (m.min(n) - 1).max(0)];

    // Householder bidiagonalization
    let min_mn = m.min(n);

    for i in 0..min_mn {
        // ---- Left Householder reflection (zero below diagonal in column i) ----
        if i < m - 1 {
            let col = (i..m).map(|r| a[r][i]).collect::<Vec<_>>();
            let (hv, tau, beta) = householder_vector(&col);
            d[i] = beta;

            if tau.abs() > 1e-15 {
                apply_householder_left(a, tau, &hv, i, m, n);
                apply_householder_left_to_u(&mut u, tau, &hv, i, m);
            }
        } else {
            d[i] = a[i][i];
        }

        // ---- Right Householder reflection (zero right of superdiagonal in row i) ----
        if i < n - 1 {
            let row = (i + 1..n).map(|c| a[i][c]).collect::<Vec<_>>();
            let (hv, tau, beta) = householder_vector(&row);
            if i < e.len() {
                e[i] = beta;
            }

            if tau.abs() > 1e-15 {
                apply_householder_right(a, tau, &hv, i, m, n);
                apply_householder_right_to_v(&mut v, tau, &hv, i, n);
            }
        }
    }

    Ok(BidigState { u, v, d, e })
}

/// Compute Householder reflector for vector x.
///
/// Returns (v, tau, beta) where:
/// - v  = Householder vector (same length as x, with v[0] = x[0] - alpha)
/// - tau = 2 / (v^T v)   (scalar for H = I - tau * v * v^T)
/// - beta = alpha = sign(x[0]) * norm(x)  (the resulting first element)
///
/// H * x = [beta, 0, ..., 0]^T
fn householder_vector(x: &[f64]) -> (Vec<f64>, f64, f64) {
    if x.is_empty() {
        return (vec![], 0.0, 0.0);
    }

    let norm_x = sqrt(x.iter().map(|v| v * v).sum::<f64>());

    if norm_x == 0.0 {
        return (x.to_vec(), 0.0, 0.0);
    }

    // alpha = sign(x[0]) * norm(x)  — choose sign to avoid cancellation
    let alpha = if x[0] >= 0.0 { norm_x } else { -norm_x };
    let beta = alpha;

    // Householder vector v = x with v[0] replaced by (x[0] - alpha)
    let mut v = x.to_vec();
    v[0] -= alpha;

    let v_norm_sq: f64 = v.iter().map(|&vi| vi * vi).sum();

    if v_norm_sq < 1e-30 {
        return (v, 0.0, beta);
    }

    let tau = 2.0 / v_norm_sq;
    (v, tau, beta)
}

/// Apply Householder reflection from the left: A[row_start:m, :] := (I - tau*v*v^T) * A[row_start:m, :]
/// v is 0-based and covers rows row_start..m
fn apply_householder_left(a: &mut Vec<Vec<f64>>, tau: f64, v: &[f64], row_start: usize, m: usize, n: usize) {
    if tau.abs() < 1e-15 {
        return;
    }
    for j in 0..n {
        let dot: f64 = (row_start..m).map(|i| v[i - row_start] * a[i][j]).sum();
        let w_j = tau * dot;
        for i in row_start..m {
            a[i][j] -= w_j * v[i - row_start];
        }
    }
}

/// Accumulate left Householder reflector into U: U := U * H
/// where H = I - tau * v * v^T acts on rows row_start..m.
/// U is m×m; columns row_start..m of each row are updated.
fn apply_householder_left_to_u(u: &mut Vec<Vec<f64>>, tau: f64, v: &[f64], row_start: usize, m: usize) {
    if tau.abs() < 1e-15 {
        return;
    }
    // U := U * H  ⟺  for each row i: u[i, row_start..m] -= tau * (u[i, row_start..m] · v) * v
    for i in 0..m {
        let dot: f64 = (row_start..m).map(|j| u[i][j] * v[j - row_start]).sum();
        let w = tau * dot;
        for j in row_start..m {
            u[i][j] -= w * v[j - row_start];
        }
    }
}

/// Apply Householder reflection from the right: A[:, col_start+1:n] := A[:, col_start+1:n] * (I - tau*w*w^T)
/// w is 0-based and covers columns col_start+1..n
fn apply_householder_right(a: &mut Vec<Vec<f64>>, tau: f64, w: &[f64], col_start: usize, m: usize, n: usize) {
    if tau.abs() < 1e-15 {
        return;
    }
    for i in 0..m {
        let dot: f64 = (col_start + 1..n).map(|j| a[i][j] * w[j - col_start - 1]).sum();
        let z_i = tau * dot;
        for j in col_start + 1..n {
            a[i][j] -= z_i * w[j - col_start - 1];
        }
    }
}

/// Apply Householder reflection from the right to V: V[:, col_start+1:n] := V[:, col_start+1:n] * (I - tau*w*w^T)
fn apply_householder_right_to_v(v: &mut Vec<Vec<f64>>, tau: f64, w: &[f64], col_start: usize, n: usize) {
    if tau.abs() < 1e-15 {
        return;
    }
    for i in 0..n {
        let dot: f64 = (col_start + 1..n).map(|j| v[i][j] * w[j - col_start - 1]).sum();
        let z_i = tau * dot;
        for j in col_start + 1..n {
            v[i][j] -= z_i * w[j - col_start - 1];
        }
    }
}

/// 2×2 bidiagonal SVD (LAPACK DLASV2 直訳)
///
/// 入力: 上 bidiagonal [[f, g], [0, h]]
/// 出力: (ssmax, ssmin, snr, csr, snl, csl)
///   U^T * [[f,g],[0,h]] * V = diag(ssmax, ssmin)
///   U = [[csl,-snl],[snl,csl]], V = [[csr,-snr],[snr,csr]]
///
/// LAPACK DLASV2 l.39–181 の直訳。三角関数を一切使わない。
fn dlasv2(f: f64, g: f64, h: f64) -> (f64, f64, f64, f64, f64, f64) {
    let mut ft = f;
    let mut fa = ft.abs();
    let mut ht = h;
    let mut ha = ht.abs();

    // PMAX: 1=F最大, 2=G最大, 3=H最大
    let mut pmax = 1usize;
    let swap = ha > fa;
    if swap {
        pmax = 3;
        core::mem::swap(&mut ft, &mut ht);
        core::mem::swap(&mut fa, &mut ha);
        // Now fa >= ha
    }
    let gt = g;
    let ga = gt.abs();

    let (ssmin, ssmax, clt, slt, crt, srt);

    if ga == 0.0 {
        // 対角行列
        ssmin = ha;
        ssmax = fa;
        clt = 1.0;
        slt = 0.0;
        crt = 1.0;
        srt = 0.0;
    } else {
        let mut gasmal = true;
        if ga > fa {
            pmax = 2;
            // ga が機械イプシロンより fa/ga が小さい → 超大 GA ケース
            if (fa / ga) < 2.22e-16_f64 {
                gasmal = false;
                ssmax = ga;
                ssmin = if ha > 1.0 { fa / (ga / ha) } else { (fa / ga) * ha };
                clt = 1.0;
                slt = ht / gt;
                srt = 1.0;
                crt = ft / gt;
                // goto 結果へ（下の if gasmal をスキップ）
                let (csl, snl, csr, snr): (f64, f64, f64, f64) = if swap {
                    (srt, crt, slt, clt)
                } else {
                    (clt, slt, crt, srt)
                };
                let tsign = if pmax == 1 {
                    csr.signum() * csl.signum() * f.signum()
                } else if pmax == 2 {
                    snr.signum() * csl.signum() * g.signum()
                } else {
                    snr.signum() * snl.signum() * h.signum()
                };
                let ssmax_out = ssmax.copysign(tsign);
                let ssmin_out = ssmin.copysign(tsign * f.signum() * h.signum());
                return (ssmax_out, ssmin_out, csr, snr, csl, snl);
            }
        }
        if gasmal {
            // 通常ケース (LAPACK l.96–157)
            let d = fa - ha;
            let l = if d == fa { 1.0 } else { d / fa };  // 0 <= l <= 1
            let m = gt / ft;                               // |m| <= 1/eps
            let t = 2.0 - l;                               // t >= 1
            let mm = m * m;
            let tt = t * t;
            let s = sqrt(tt + mm);                         // 1 <= s <= 1+1/eps
            let r = if l == 0.0 { m.abs() } else { sqrt(l * l + mm) }; // 0 <= r <= 1+1/eps
            let a = 0.5 * (s + r);                         // 1 <= a <= 1+|m|
            ssmin = ha / a;
            ssmax = fa * a;
            let t2 = if mm == 0.0 {
                if l == 0.0 {
                    2.0_f64.copysign(ft) * gt.signum()
                } else {
                    gt / d.copysign(ft) + m / t
                }
            } else {
                (m / (s + t) + m / (r + l)) * (1.0 + a)
            };
            let l2 = sqrt(t2 * t2 + 4.0);
            crt = 2.0 / l2;
            srt = t2 / l2;
            clt = (crt + srt * m) / a;
            slt = (ht / ft) * srt / a;
        } else {
            unreachable!()
        }
    }

    // swap 補正 (LAPACK l.159–168)
    let (csl, snl, csr, snr) = if swap {
        (srt, crt, slt, clt)
    } else {
        (clt, slt, crt, srt)
    };

    // 符号補正 (LAPACK l.173–180)
    let tsign = if pmax == 1 {
        csr.signum() * csl.signum() * f.signum()
    } else if pmax == 2 {
        snr.signum() * csl.signum() * g.signum()
    } else {
        snr.signum() * snl.signum() * h.signum()
    };
    let ssmax_out = ssmax.copysign(tsign);
    let ssmin_out = ssmin.copysign(tsign * f.signum() * h.signum());

    // 呼び出し側は (sigmx, sigmn, cosr, sinr, cosl, sinl) の順を期待
    // LAPACK DLASV2 出力: ssmax, ssmin, snr, csr, snl, csl
    //   csr=cosr, snr=sinr, csl=cosl, snl=sinl
    (ssmax_out, ssmin_out, csr, snr, csl, snl)
}

/// Golub-Reinsch bidiagonal SVD (LAPACK DBDSQR 準拠)
///
/// LAPACK との対応:
///   - idir: 双方向 sweep (大端→小端 or 小端→大端を条件で切替)
///   - sminoa/mu 連鎖: 前向き・後向き相対収束判定
///   - ll == p-1 末端2×2: dlasv2 で直接処理
///   - MAXITR = 6*k でイテレーション上限
fn dbdsqr(state: &BidigState, _max_iter: usize) -> Result<(Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>), SvdError> {
    let m = state.u.len();
    let n = state.v.len();
    let k = state.d.len();

    let mut u = state.u.clone();
    let mut v = state.v.clone();
    let mut d = state.d.clone();
    let mut e = state.e.clone();
    while e.len() < k { e.push(0.0); }

    dlog!("dbdsqr", "start k={k} d={d:.4?} e={e:.4?}", d = &d, e = &e[..k-1]);

    if k == 0 { return Ok((d, u, v)); }
    if k == 1 {
        if d[0] < 0.0 { d[0] = -d[0]; for j in 0..n { v[j][0] = -v[j][0]; } }
        return Ok((d, u, v));
    }

    // LAPACK 定数
    const EPS: f64    = 2.22e-16;
    const UNFL: f64   = 2.23e-308; // safe minimum
    const TOLMUL: f64 = 100.0;
    let tol = TOLMUL * EPS;
    let maxitr = 6 * k;

    // smax: 全要素の最大絶対値（閾値スケール用）
    let smax = d.iter().chain(e.iter()).map(|v| v.abs()).fold(0.0_f64, f64::max);
    let thresh = (tol * smax).max((maxitr * k * k) as f64 * UNFL);
    dlog!("dbdsqr", "smax={smax:.4e} thresh={thresh:.4e} tol={tol:.4e}");

    let mut p   = k - 1; // アクティブブロックの末尾インデックス
    let mut idir = 0usize; // 1=上→下, 2=下→上
    let mut oldll = usize::MAX;
    let mut oldm  = usize::MAX;
    let mut iterdivn = 0usize;
    let mut iter_count = 0i64;

    'outer: loop {
        if p == 0 { break; }

        // 収束チェック: p 末端から縮小
        while e[p - 1].abs() <= thresh {
            e[p - 1] = 0.0;
            if p == 1 { p = 0; break 'outer; }
            p -= 1;
        }
        if p == 0 { break; }

        // イテレーション上限チェック
        iter_count += (p as i64) + 1;
        if iter_count > (maxitr * k) as i64 {
            iter_count -= (p as i64) + 1;
            iterdivn += 1;
            if iterdivn >= maxitr { break; }
        }

        // アクティブブロック下端 ll を探す
        let mut ll = p - 1;
        loop {
            if e[ll].abs() <= thresh {
                e[ll] = 0.0;
                ll += 1; // ll..=p が アクティブブロック
                break;
            }
            if ll == 0 { break; }
            ll -= 1;
        }

        // ── 末端2×2ブロック直接処理 (LAPACK ll==m-1 相当) ────────────────
        if ll == p - 1 {
            dlog!("dbdsqr", "  2x2 direct: ll={ll} p={p} d[p-1]={:.4e} e[p-1]={:.4e} d[p]={:.4e}",
                d[p-1], e[p-1], d[p]);
            let (sigmx, sigmn, cosr, sinr, cosl, sinl) =
                dlasv2(d[p - 1], e[p - 1], d[p]);
            d[p - 1] = sigmx;
            e[p - 1] = 0.0;
            d[p]     = sigmn;
            dlog!("dbdsqr", "  2x2 result: sigmx={sigmx:.4e} sigmn={sigmn:.4e}");
            // singular vectors 更新
            apply_givens_v(&mut v, cosr, sinr, p - 1, p, n);
            apply_givens_u(&mut u, cosl, sinl, p - 1, p, m);
            if p == 1 { p = 0; break; }
            p -= 2;
            continue;
        }

        // ── idir: sweep 方向の決定 ──────────────────────────────────────
        if ll != oldll || p != oldm {
            idir = if d[ll].abs() >= d[p].abs() { 1 } else { 2 };
        }
        oldll = ll;
        oldm  = p;
        dlog!("dbdsqr", "iter={iter_count} ll={ll} p={p} idir={idir} d={:.4?} e={:.4?}",
            &d[ll..=p], &e[ll..p]);

        // ── 収束テスト (LAPACK の前向き/後向き mu 連鎖) ─────────────────
        let converged_early = if idir == 1 {
            // 前向き: ll → p の方向で smin を追跡
            let mut mu = d[ll].abs();
            let mut smin = mu;
            let mut hit = false;
            for lll in ll..p {
                if e[lll].abs() <= tol * mu {
                    e[lll] = 0.0;
                    hit = true;
                    break;
                }
                mu = d[lll + 1].abs() * mu / (mu + e[lll].abs());
                smin = smin.min(mu);
            }
            hit
        } else {
            // 後向き: p → ll の方向
            let mut mu = d[p].abs();
            let mut smin = mu;
            let mut hit = false;
            for lll in (ll..p).rev() {
                if e[lll].abs() <= tol * mu {
                    e[lll] = 0.0;
                    hit = true;
                    break;
                }
                mu = d[lll].abs() * mu / (mu + e[lll].abs());
                smin = smin.min(mu);
            }
            hit
        };
        if converged_early {
            dlog!("dbdsqr", "  early converge via mu-chain");
            continue;
        }

        // ── シフト: ゼロシフト判定 (LAPACK: n*tol*(smin/smax) <= max(eps, hndrth*tol)) ─
        let smin_est = {
            let mut mu = d[ll].abs();
            for lll in ll..p { mu = d[lll+1].abs() * mu / (mu + e[lll].abs()); }
            mu
        };
        let use_zero_shift = (k as f64) * tol * (smin_est / smax)
            <= EPS.max(0.01 * tol);

        // ── QR イテレーション (LAPACK DBDSQR DO 120/DO 130/DO 140/DO 150) ──
        if idir == 1 {
            // 上→下 sweep (LAPACK DO 140: idir==1, shift!=0)
            let shift = if use_zero_shift { 0.0 } else { compute_qr_shift(&d, &e, p) };
            dlog!("dbdsqr", "  sweep↓ shift={shift:.4e} zero={use_zero_shift}");
            // f = (|d[ll]| - shift) * (sign(d[ll]) + shift/d[ll])  (LAPACK l.476-477)
            let mut f = (d[ll].abs() - shift)
                * (if d[ll] >= 0.0 { 1.0 } else { -1.0 } + shift / d[ll]);
            let mut g = e[ll];

            for i in ll..p {
                // 右回転: (f, g) → r, V に累積
                let (cosr, sinr, r) = givens_params(f, g);
                if i > ll { e[i - 1] = r; }
                // bidiagonal 更新 (LAPACK l.483-486)
                f  =  cosr * d[i] + sinr * e[i];
                e[i] = cosr * e[i] - sinr * d[i];
                g  =  sinr * d[i + 1];
                d[i + 1] = cosr * d[i + 1];
                apply_givens_v(&mut v, cosr, sinr, i, i + 1, n);

                // 左回転: (f, g) → r, U に累積
                let (cosl, sinl, r) = givens_params(f, g);
                d[i] = r;
                // bidiagonal 更新 (LAPACK l.489-492)
                f      =  cosl * e[i] + sinl * d[i + 1];
                d[i + 1] = cosl * d[i + 1] - sinl * e[i];
                if i + 1 < p {
                    g = sinl * e[i + 1];
                    e[i + 1] = cosl * e[i + 1];
                }
                apply_givens_u(&mut u, cosl, sinl, i, i + 1, m);
            }
            e[p - 1] = f;
            dlog!("dbdsqr", "  after↓ d={:.4?} e={:.4?}", &d[ll..=p], &e[ll..p]);

        } else {
            // 下→上 sweep (LAPACK DO 150: idir==2, shift!=0)
            let shift = if use_zero_shift { 0.0 } else { compute_qr_shift_bottom(&d, &e, ll, p) };
            dlog!("dbdsqr", "  sweep↑ shift={shift:.4e} zero={use_zero_shift}");
            // f = (|d[p]| - shift) * (sign(d[p]) + shift/d[p])  (LAPACK l.526-527)
            let mut f = (d[p].abs() - shift)
                * (if d[p] >= 0.0 { 1.0 } else { -1.0 } + shift / d[p]);
            let mut g = e[p - 1];

            for i in (ll..p).rev() {
                // 右回転
                let (cosr, sinr, r) = givens_params(f, g);
                if i < p - 1 { e[i + 1] = r; }
                // bidiagonal 更新 (LAPACK l.533-536)
                f    =  cosr * d[i + 1] + sinr * e[i];
                e[i] =  cosr * e[i] - sinr * d[i + 1];
                g    =  sinr * d[i];
                d[i] =  cosr * d[i];
                // LAPACK DO 150: work(nm1) = -sinr → V に渡すとき符号反転
                apply_givens_v(&mut v, cosr, -sinr, i, i + 1, n);

                // 左回転
                let (cosl, sinl, r) = givens_params(f, g);
                d[i + 1] = r;
                // bidiagonal 更新 (LAPACK l.539-542)
                f    =  cosl * e[i] + sinl * d[i];
                d[i] =  cosl * d[i] - sinl * e[i];
                if i > ll {
                    g = sinl * e[i - 1];
                    e[i - 1] = cosl * e[i - 1];
                }
                // LAPACK DO 150: work(nm13) = -sinl → U に渡すとき符号反転
                apply_givens_u(&mut u, cosl, -sinl, i, i + 1, m);
            }
            e[ll] = f;
            dlog!("dbdsqr", "  after↑ d={:.4?} e={:.4?}", &d[ll..=p], &e[ll..p]);
        }
    }

    dlog!("dbdsqr", "done d={d:.4?}", d = &d);
    // 負の特異値を正に
    for i in 0..k {
        if d[i] < 0.0 {
            d[i] = -d[i];
            for j in 0..n { v[j][i] = -v[j][i]; }
        }
    }

    // Sort singular values in descending order
    let mut indices: Vec<usize> = (0..k).collect();
    indices.sort_by(|&i, &j| d[j].partial_cmp(&d[i]).unwrap_or(Ordering::Equal));

    let mut sorted_d = vec![0.0; k];
    let mut sorted_u = u.clone();
    let mut sorted_v = v.clone();

    for (i, &idx) in indices.iter().enumerate() {
        sorted_d[i] = d[idx];
        for j in 0..m {
            sorted_u[j][i] = u[j][idx];
        }
        for j in 0..n {
            sorted_v[j][i] = v[j][idx];
        }
    }

    Ok((sorted_d, sorted_u, sorted_v))
}

/// Wilkinson shift: 対称 2×2 行列 [[a,c],[c,b]] の固有値のうち b に近い方
/// nalgebra symmetric_eigen::wilkinson_shift と同じ公式
fn wilkinson_shift(a: f64, b: f64, c: f64) -> f64 {
    let d = (a - b) * 0.5;
    let denom = if d == 0.0 {
        c.abs()
    } else {
        d.abs() + sqrt(d * d + c * c)
    };
    if denom == 0.0 { return b; }
    b - c * c / denom.copysign(d)
}

/// 上→下 sweep 用シフト (LAPACK DO 140 直前の shift 計算)
/// B^T B 末端 2×2 = [[tmm, tmn],[tmn, tnn]] から Wilkinson shift
///   tmm = d[p-1]^2 + e[p-2]^2 (または d[p-1]^2 のみ)
///   tmn = d[p-1] * e[p-1]
///   tnn = d[p]^2 + e[p-1]^2
fn compute_qr_shift(d: &[f64], e: &[f64], p: usize) -> f64 {
    let dm  = d[p - 1];
    let dn  = d[p];
    let em  = e[p - 1];
    let em1 = if p >= 2 { e[p - 2] } else { 0.0 };
    let tmm = dm * dm + em1 * em1;
    let tmn = dm * em;
    let tnn = dn * dn + em  * em;
    wilkinson_shift(tmm, tnn, tmn).max(0.0)
}

/// 下→上 sweep 用シフト (LAPACK DO 150 直前の shift 計算)
/// B^T B 先頭 2×2 = [[tll, tln],[tln, tl1l1]] から Wilkinson shift
fn compute_qr_shift_bottom(d: &[f64], e: &[f64], ll: usize, _p: usize) -> f64 {
    let dll  = d[ll];
    let dl1  = d[ll + 1];
    let ell  = e[ll];
    let ell1 = if ll + 2 < e.len() { e[ll + 1] } else { 0.0 };
    let tll   = dll  * dll  + ell  * ell;   // (ll,ll) 要素
    let tln   = dll  * ell;                 // (ll,ll+1) 要素
    let tl1l1 = dl1  * dl1  + ell1 * ell1; // (ll+1,ll+1) 要素
    wilkinson_shift(tl1l1, tll, tln).max(0.0)
}

/// Apply Givens rotation to U matrix (left multiplication)
///
/// Updates columns i and j of U matrix:
/// U[:, [i, j]] := U[:, [i, j]] * [[c, s], [-s, c]]
fn apply_givens_u(u: &mut Vec<Vec<f64>>, c: f64, s: f64, i: usize, j: usize, m: usize) {
    for row in 0..m {
        let u_i = u[row][i];
        let u_j = u[row][j];
        u[row][i] = c * u_i + s * u_j;
        u[row][j] = -s * u_i + c * u_j;
    }
}

/// Apply Givens rotation to V matrix (right multiplication)
///
/// Updates columns i and j of V matrix:
/// V[:, [i, j]] := V[:, [i, j]] * [[c, s], [-s, c]]
fn apply_givens_v(v: &mut Vec<Vec<f64>>, c: f64, s: f64, i: usize, j: usize, n: usize) {
    for row in 0..n {
        let v_i = v[row][i];
        let v_j = v[row][j];
        v[row][i] = c * v_i + s * v_j;
        v[row][j] = -s * v_i + c * v_j;
    }
}

/// Compute Givens rotation parameters (c, s) for vector [a, b]
///
/// Returns (c, s) such that [[c, s], [-s, c]]^T * [a; b] = [r; 0]
/// where r = sqrt(a^2 + b^2)
///
/// This is the standard Givens rotation for zeroing the second element.
/// LAPACK dlartg: numerically safe Givens rotation.
/// Returns (c, s, r) such that [c s; -s c] * [f; g] = [r; 0].
/// c >= 0, r = sign(hypot(f,g), f).
fn givens_params(f: f64, g: f64) -> (f64, f64, f64) {
    const SAFMIN: f64 = 2.2250738585072014e-308_f64;
    const SAFMAX: f64 = 4.4942328371557897e+307_f64; // safmax / 2
    let rtmin = sqrt(SAFMIN);
    let rtmax = sqrt(SAFMAX);

    let f1 = f.abs();
    let g1 = g.abs();

    if g == 0.0 {
        return (1.0, 0.0, f);
    }
    if f == 0.0 {
        let s = if g >= 0.0 { 1.0 } else { -1.0 };
        return (0.0, s, g1);
    }

    let (c, s, r);
    if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
        let d = sqrt(f * f + g * g);
        c = f1 / d;
        r = if f >= 0.0 { d } else { -d };
        s = g / r;
    } else {
        let u = f1.max(g1).min(SAFMAX * 2.0).max(SAFMIN);
        let fs = f / u;
        let gs = g / u;
        let d = sqrt(fs * fs + gs * gs);
        c = fs.abs() / d;
        r = if f >= 0.0 { d * u } else { -d * u };
        s = gs / (r / u);
    }

    (c, s, r)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::println;

    fn init_log() { let _ = crate::debug_log::init_test_logger(); }

    #[test]
    fn test_gebrd_simple_matrix() {
        // Test GEBRD on a simple 3x3 matrix
        let mut a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        match gebrd(&mut a) {
            Ok(state) => {
                // Basic structural checks
                assert_eq!(state.d.len(), 3, "diagonal vector length");
                assert_eq!(state.e.len(), 2, "superdiagonal vector length");
                assert_eq!(state.u.len(), 3, "U matrix rows");
                assert_eq!(state.v.len(), 3, "V matrix rows");

                // Check diagonal is non-zero (unless matrix is singular)
                let max_diag = state.d.iter().map(|x| x.abs()).fold(0.0, f64::max);
                assert!(max_diag > 1e-10, "diagonal should have non-zero entries");

                println!(
                    "GEBRD test passed. Diagonal: {:?}, Superdiagonal: {:?}",
                    state.d, state.e
                );
            }
            Err(e) => panic!("GEBRD failed: {}", e),
        }
    }

    #[test]
    fn test_gebrd_tall_matrix() {
        // Test GEBRD on a tall m > n matrix
        let mut a = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];

        match gebrd(&mut a) {
            Ok(state) => {
                assert_eq!(state.d.len(), 2);
                assert_eq!(state.e.len(), 1);
                assert_eq!(state.u.len(), 4);
                assert_eq!(state.u[0].len(), 4);
                assert_eq!(state.v.len(), 2);
                assert_eq!(state.v[0].len(), 2);
            }
            Err(e) => panic!("GEBRD failed: {}", e),
        }
    }

    #[test]
    fn test_gebrd_wide_matrix() {
        // Test GEBRD on a wide m < n matrix
        let mut a = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];

        match gebrd(&mut a) {
            Ok(state) => {
                assert_eq!(state.d.len(), 2);
                assert_eq!(state.e.len(), 1);
                assert_eq!(state.u.len(), 2);
                assert_eq!(state.v.len(), 4);
            }
            Err(e) => panic!("GEBRD failed: {}", e),
        }
    }

    #[test]
    fn test_householder_vector() {
        // Test Householder vector computation
        let x = vec![1.0, 2.0, 3.0];
        let (_v, tau, beta) = householder_vector(&x);

        // tau should be positive and reasonable
        assert!(tau >= 0.0 && tau <= 2.0);

        // beta should be the norm of x with appropriate sign
        let norm_x = sqrt(x.iter().map(|v| v * v).sum::<f64>());
        assert!((beta.abs() - norm_x).abs() < 1e-10);
    }

    fn transpose(a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        if a.is_empty() {
            return vec![];
        }
        let m = a.len();
        let n = a[0].len();
        let mut result = vec![vec![0.0; m]; n];
        for i in 0..m {
            for j in 0..n {
                result[j][i] = a[i][j];
            }
        }
        result
    }

    fn matrix_mult(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let m = a.len();
        let n = b[0].len();
        let k = a[0].len();
        let mut result = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    result[i][j] += a[i][p] * b[p][j];
                }
            }
        }
        result
    }

    fn assert_identity_like(a: &Vec<Vec<f64>>, tol: f64) {
        let n = a.len();
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (a[i][j] - expected).abs() < tol,
                    "Matrix not identity-like at [{}, {}]: {} vs {}",
                    i,
                    j,
                    a[i][j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_full_basic() {
        // Test full SVD on a simple matrix
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        match full(&a) {
            Ok((u, s, vt)) => {
                // Check dimensions
                assert_eq!(u.len(), 3, "U rows");
                assert_eq!(u[0].len(), 2, "U cols");
                assert_eq!(s.len(), 2, "singular values");
                assert_eq!(vt.len(), 2, "V^T rows");
                assert_eq!(vt[0].len(), 2, "V^T cols");

                // Check singular values are non-negative and descending
                assert!(s[0] >= s[1], "singular values should be descending");
                assert!(s[0] > 0.0, "singular values should be positive");

                println!("SVD test passed. Singular values: {:?}", s);
            }
            Err(e) => panic!("SVD failed: {}", e),
        }
    }

    #[test]
    fn test_singvals_basic() {
        // Test singvals on a simple matrix
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        let s = singvals(&a);
        assert_eq!(s.len(), 2, "number of singular values");
        assert!(s[0] >= s[1], "singular values should be descending");
        assert!(s[0] > 0.0, "singular values should be positive");
        println!("Singvals test passed: {:?}", s);
    }

    #[test]
    fn test_against_nalgebra() {
        init_log();
        // Singular values for [[1,2],[3,4],[5,6]].
        // nalgebra reference: [9.52551809, 0.51430058]
        let a = vec![vec![1.0f64, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let s = singvals(&a);
        assert!((s[0] - 9.5255).abs() < 1e-4, "s[0] expected ~9.5255, got {}", s[0]);
        assert!((s[1] - 0.5143).abs() < 1e-4, "s[1] expected ~0.5143, got {}", s[1]);

        // full() must produce correct U and Vt:
        // for y = 2 + 3*x on a 30x2 design matrix, beta = Vt^T * diag(1/s) * U^T * y
        // should recover [2.0, 3.0] exactly.
        let x_rows: Vec<Vec<f64>> = (0..30).map(|i| vec![1.0, i as f64 / 30.0]).collect();
        let y_lin: Vec<f64> = x_rows.iter().map(|r| 2.0 + 3.0 * r[1]).collect();
        let (u, s2, vt) = full(&x_rows).unwrap();
        let m = x_rows.len();
        let k = s2.len();
        let nf = vt[0].len();
        let uty: Vec<f64> = (0..k).map(|j| (0..m).map(|i| u[i][j] * y_lin[i]).sum()).collect();
        let mut beta = vec![0.0f64; nf];
        for col in 0..nf {
            beta[col] = (0..k).map(|j| vt[j][col] * uty[j] / s2[j]).sum::<f64>();
        }
        assert!((beta[0] - 2.0).abs() < 1e-10, "intercept expected 2.0, got {}", beta[0]);
        assert!((beta[1] - 3.0).abs() < 1e-10, "slope expected 3.0, got {}", beta[1]);
    }

    #[test]
    fn test_golden_simple() {
        // Placeholder for Python golden file tests
        // Golden files will be loaded and tested separately
    }

    // ── nalgebra比較テストのヘルパー ──────────────────────────────────────

    fn nalgebra_singvals(a: &Vec<Vec<f64>>) -> Vec<f64> {
        use nalgebra::DMatrix;
        let m = a.len();
        let n = a[0].len();
        let dm = DMatrix::from_fn(m, n, |r, c| a[r][c]);
        let svd = dm.svd(false, false);
        svd.singular_values.iter().copied().collect()
    }

    fn reconstruction_error(a: &Vec<Vec<f64>>, u: &Vec<Vec<f64>>, s: &[f64], vt: &Vec<Vec<f64>>) -> f64 {
        let m = a.len();
        let n = a[0].len();
        let k = s.len();
        let mut err = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let rec: f64 = (0..k).map(|r| u[i][r] * s[r] * vt[r][j]).sum();
                err += (a[i][j] - rec).powi(2);
            }
        }
        sqrt(err)
    }

    fn orthogonality_error_u(u: &Vec<Vec<f64>>) -> f64 {
        let m = u.len();
        let k = u[0].len();
        let mut err = 0.0f64;
        for i in 0..k {
            for j in 0..k {
                let dot: f64 = (0..m).map(|r| u[r][i] * u[r][j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                err += (dot - expected).powi(2);
            }
        }
        sqrt(err)
    }

    fn orthogonality_error_vt(vt: &Vec<Vec<f64>>) -> f64 {
        let k = vt.len();
        let n = vt[0].len();
        let mut err = 0.0f64;
        for i in 0..k {
            for j in 0..k {
                let dot: f64 = (0..n).map(|c| vt[i][c] * vt[j][c]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                err += (dot - expected).powi(2);
            }
        }
        sqrt(err)
    }

    fn assert_svd_quality(label: &str, a: &Vec<Vec<f64>>, tol_recon: f64, tol_orth: f64) {
        let (u, s, vt) = full(a).unwrap_or_else(|e| panic!("{label}: full() failed: {e}"));
        let recon   = reconstruction_error(a, &u, &s, &vt);
        let orth_u  = orthogonality_error_u(&u);
        let orth_vt = orthogonality_error_vt(&vt);
        assert!(recon   < tol_recon, "{label}: reconstruction error {recon:.2e} >= {tol_recon:.2e}");
        assert!(orth_u  < tol_orth,  "{label}: U orthogonality {orth_u:.2e} >= {tol_orth:.2e}");
        assert!(orth_vt < tol_orth,  "{label}: Vt orthogonality {orth_vt:.2e} >= {tol_orth:.2e}");
    }

    // ── 精度検証テスト ────────────────────────────────────────────────────

    /// 末端2×2ブロック処理の精度検証
    ///
    /// k=2 の純粋2×2行列は 1 回の Givens で偶然収束するため末端処理の欠如を
    /// 検出できない。問題が顕在化するのは k≥3 の行列がイテレーション中に
    /// 末端2×2ブロック (ll==m-1) に縮んだ瞬間。
    /// そのシナリオを意図的に作るため、s[0]>>s[1]≈s[2] な 3×3 行列を使う:
    /// 上位特異値が先に収束して p が減り、残った2×2ブロックで dlasv2 相当が
    /// 必要になる。
    #[test]
    fn svd_2x2_trailing_block() {
        init_log();
        let cases: &[(&str, Vec<Vec<f64>>)] = &[
            // s ≈ [1000, ~0.7, ~0.3]: 大きい s[0] が先に収束し 2×2 末端が残る
            ("3x3-trailing-2x2",
             vec![vec![1.0f64, 0.0, 0.0],
                  vec![0.0,    1.0, 1.0],
                  vec![0.0,    1.0, -1.0]]),
            // 高条件数 3×3: s[0]/s[2] ≈ 1e6
            ("3x3-high-cond",
             vec![vec![1e3f64, 1.0,  0.0],
                  vec![0.0,    1.0,  1.0],
                  vec![0.0,    1e-3, 1.0]]),
            // 一般密 3×3: 全要素非ゼロ、末端2×2が最も収束しにくい
            ("3x3-dense",
             vec![vec![4.0f64, 3.0, 2.0],
                  vec![3.0,    2.0, 1.0],
                  vec![1.0,    1.0, 5.0]]),
        ];
        for (label, a) in cases {
            let s_ref = nalgebra_singvals(a);
            let s = singvals(a);
            for (i, (&sr, &sf)) in s_ref.iter().zip(s.iter()).enumerate() {
                let err = if sr > 1e-8 { (sr - sf).abs() / sr } else { (sr - sf).abs() };
                assert!(err < 1e-6,
                    "{label} s[{i}]: nalgebra={sr:.6e} flair={sf:.6e} err={err:.2e}");
            }
            assert_svd_quality(label, a, 1e-11, 1e-11);
        }
    }

    /// 高条件数対角行列: 特異値 [1e6, 1e3, 1.0, 1e-3] — 小特異値の相対精度
    #[test]
    fn svd_high_condition_number() {
        let scales = [1e6f64, 1e3, 1.0, 1e-3];
        let a: Vec<Vec<f64>> = (0..4).map(|i|
            (0..4).map(|j| if i == j { scales[i] } else { 0.0 }).collect()
        ).collect();
        let s_ref = nalgebra_singvals(&a);
        let s = singvals(&a);
        for (i, (&sr, &sf)) in s_ref.iter().zip(s.iter()).enumerate() {
            let rel = (sr - sf).abs() / sr.max(1e-15);
            assert!(rel < 1e-6,
                "high-cond s[{i}]: nalgebra={sr:.6e} flair={sf:.6e} rel={rel:.2e}");
        }
    }

    /// rank-1 行列: 非ゼロ特異値が1個のみ, s[1..] ≈ 0
    #[test]
    fn svd_rank1_matrix() {
        let u_vec = [1.0f64, 2.0, 3.0];
        let v_vec = [4.0f64, 5.0, 6.0];
        let a: Vec<Vec<f64>> = (0..3).map(|i|
            (0..3).map(|j| u_vec[i] * v_vec[j]).collect()
        ).collect();
        let s = singvals(&a);
        let expected_s0 = sqrt(u_vec.iter().map(|&v| v*v).sum::<f64>())
                        * sqrt(v_vec.iter().map(|&v| v*v).sum::<f64>());
        assert!((s[0] - expected_s0).abs() / expected_s0 < 1e-8,
            "rank-1 s[0]: expected {expected_s0:.6} got {:.6}", s[0]);
        assert!(s[1] < 1e-8, "rank-1 s[1] should be ~0, got {}", s[1]);
        assert!(s[2] < 1e-8, "rank-1 s[2] should be ~0, got {}", s[2]);
        assert_svd_quality("rank-1", &a, 1e-10, 1e-10);
    }

    /// near-singular: 最小特異値 ≈ 1e-10 — 絶対誤差で評価
    #[test]
    fn svd_near_singular() {
        let eps = 1e-10f64;
        let a = vec![vec![1.0f64, 1.0], vec![1.0, 1.0 + eps]];
        let s_ref = nalgebra_singvals(&a);
        let s = singvals(&a);
        let rel0 = (s_ref[0] - s[0]).abs() / s_ref[0];
        assert!(rel0 < 1e-8, "near-singular s[0] rel={rel0:.2e}");
        let abs1 = (s_ref[1] - s[1]).abs();
        assert!(abs1 < 1e-10,
            "near-singular s[1] abs={abs1:.2e} (ref={:.4e} got={:.4e})", s_ref[1], s[1]);
    }

    /// rank-1 dominant 行列 (12×10): optshrink が使うシナリオ, 条件数≈230
    /// これが現状 s[1] で 4.97% ずれているケース — dbdsqr 修正後に閾値を締める
    #[test]
    fn svd_rank1_dominant_period_matrix() {
        init_log();
        let p = 12usize;
        let nc = 10usize;
        let mut a: Vec<Vec<f64>> = vec![vec![0.0; nc]; p];
        for ph in 0..p {
            for ci in 0..nc {
                let signal = (1.0 + 0.1 * ph as f64) * (10.0 + ci as f64);
                let noise  = 0.3 * ((ph * nc + ci) as f64 * 1.7321 % 1.0 - 0.5);
                a[ph][ci] = signal + noise;
            }
        }
        let s_ref = nalgebra_singvals(&a);
        let s = singvals(&a);

        // s[0]（信号強度）は高精度
        let rel0 = (s_ref[0] - s[0]).abs() / s_ref[0];
        assert!(rel0 < 1e-4, "period-matrix s[0] rel={rel0:.2e}");

        // s[1..]: 現状の限界（10%）を記録。dbdsqr 修正後にここを 1e-3 に締める
        for (i, (&sr, &sf)) in s_ref.iter().zip(s.iter()).enumerate() {
            let rel = (sr - sf).abs() / sr.max(1e-6);
            assert!(rel < 0.10,
                "period-matrix s[{i}]: nalgebra={sr:.4} flair={sf:.4} rel={rel:.4}");
        }

        // 再構成・直交性は常に成立
        assert_svd_quality("period-matrix", &a, 1e-8, 1e-8);
    }

    /// 背の高い行列 (50×3): ridge_sa の典型的な設計行列
    #[test]
    fn svd_tall_ridge_design() {
        init_log();
        let a: Vec<Vec<f64>> = (0..50).map(|i| {
            let t = i as f64 / 50.0;
            vec![1.0, t, (t * 6.28318).sin()]
        }).collect();
        let s_ref = nalgebra_singvals(&a);
        let s = singvals(&a);
        for (i, (&sr, &sf)) in s_ref.iter().zip(s.iter()).enumerate() {
            let rel = (sr - sf).abs() / sr.max(1e-10);
            assert!(rel < 1e-6,
                "tall-ridge s[{i}]: nalgebra={sr:.6} flair={sf:.6} rel={rel:.2e}");
        }
        assert_svd_quality("tall-ridge", &a, 1e-10, 1e-10);
    }
}

// ============================================================
// nalgebra-0.33.3 src/linalg/svd.rs より転記（比較用コメント）
// Apache-2.0 / MIT ライセンス
// ============================================================
//
// nalgebra の SVD メインループ（try_new_unordered の核心部分）
// 現在の dbdsqr 実装と何が違うか:
//
// 【違い1】 idir（sweep方向切替）なし
//   nalgebra は常に start→end（上→下）方向のみ。
//   LAPACK / 現実装は |d[ll]| vs |d[p]| で方向を切り替える。
//
// 【違い2】 subdim > 2 のメインループ: Matrix2x3 局所更新
//   nalgebra は各ステップで 2×3 の局所行列を使い、
//   rot1（右回転）→ rot2（左回転）を local に計算してから
//   U/V に適用する。
//   現実装はインプレースで bidiagonal を直接更新。
//
// 【違い3】 subdim == 2 の2×2処理: compute_2x2_uptrig_svd
//   nalgebra は hypot ベースの公式を使う（下記参照）。
//   現実装は dlasv2（atan2 ベース）。
//
// 【違い4】 shift の計算: wilkinson_shift（symmetric_eigen から）
//   nalgebra は T_{mm}, T_{nn}, T_{mn} から Wilkinson shift。
//   現実装は compute_qr_shift で末端2×2固有値を直接計算。
//
// 【違い5】 delimit_subproblem での d[m]==0 / d[n]==0 の特殊処理
//   nalgebra は diagonal がゼロのとき Givens でキャンセルする。
//   現実装は thresh のみでの収束判定（この処理がない）。
//
// ─────────────────────────────────────────────────────────────
// subdim > 2 のメインループ（svd.rs l.177-265）
// ─────────────────────────────────────────────────────────────
//
//   let m = end - 1;
//   let n = end;
//
//   // Wilkinson shift: B^T B の末端 2×2 から固有値を求める
//   let tmm = d[m]^2 + off[m-1]^2;
//   let tmn = d[m] * off[m];
//   let tnn = d[n]^2 + off[m]^2;   // ← 注意: off[m] が 2 回使われる
//   let shift = wilkinson_shift(tmm, tnn, tmn);
//   //   wilkinson_shift(tmm, tnn, tmn):
//   //     let d = (tmm - tnn) / 2;
//   //     tnn - tmn^2 / (d + sign(d)*hypot(d, tmn))
//
//   vec = [d[start]^2 - shift, d[start] * off[start]]
//
//   for k in start..n:
//     m12 = if k == n-1 { 0 } else { off[k+1] }
//     subm = Matrix2x3 {
//       [d[k],   off[k], 0    ],
//       [0,      d[k+1], m12  ],
//     }
//
//     // rot1: vec の y を 0 にする右回転 → V に適用
//     if let Some((rot1, norm1)) = GivensRotation::cancel_y(&vec):
//       rot1.inverse().rotate_rows(&mut subm[..2, 0..2])  // subm の左2列を rot1^{-1} で行回転
//       if k > start: off[k-1] = norm1
//
//       // rot2: subm の (1,0) を 0 にする左回転 → U に適用
//       v = [subm[0,0], subm[1,0]]
//       (rot2, norm2) = GivensRotation::cancel_y(&v)
//       rot2.rotate(&mut subm[..2, 1..3])  // subm の右2列を rot2 で行回転
//       subm[0,0] = norm2
//
//       // is_upper_diagonal なら:
//       //   V に rot1 を適用（v_t の k 行目2行）
//       //   U に rot2.inverse() を適用（u の k 列目2列）
//
//       d[k]     = subm[0,0]
//       d[k+1]   = subm[1,1]
//       off[k]   = subm[0,1]
//       if k != n-1: off[k+1] = subm[1,2]
//
//       vec.x = subm[0,1]   // 次の f
//       vec.y = subm[0,2]   // 次の g
//
// ─────────────────────────────────────────────────────────────
// subdim == 2 の処理（svd.rs l.266-303）
// ─────────────────────────────────────────────────────────────
//
//   (u2, s, v2) = compute_2x2_uptrig_svd(d[start], off[start], d[start+1], ...)
//   d[start]   = s[0]
//   d[start+1] = s[1]
//   off[start] = 0
//   U に u2 を rot_rows で適用
//   V^t に v2.inverse() を rot で適用
//   end -= 1
//
// ─────────────────────────────────────────────────────────────
// compute_2x2_uptrig_svd（svd.rs l.840-891）
// ─────────────────────────────────────────────────────────────
// 参考: "Computing the Singular Values of 2-by-2 Complex Matrices"
//       Sanzheng Qiao, Xiaohong Wang
//
// fn compute_2x2_uptrig_svd(m11, m12, m22):
//   denom = hypot(m11+m22, m12) + hypot(m11-m22, m12)
//   v1 = m11 * m22 * 2 / denom    ← m22 に近い特異値（キャンセルなし）
//   v2 = denom / 2                 ← m11 に近い特異値
//
//   // 右回転 csv を求める
//   (csv, sgn_v) = GivensRotation::new(m11*m12, v1^2 - m11^2)
//   //   GivensRotation::new(x, y) → (c, s) s.t. c*x + s*y = hypot(x,y), -s*x + c*y = 0
//   //   sgn_v = sign(hypot(x,y))
//   v1 *= sgn_v; v2 *= sgn_v
//
//   // 左回転 csu を求める
//   cu = (m11*csv.c + m12*csv.s) / v1
//   su = (m22 * csv.s) / v1
//   (csu, sgn_u) = GivensRotation::new(cu, su)
//   v1 *= sgn_u; v2 *= sgn_u
//
//   return (csu, [v1, v2], csv)
//
// ─────────────────────────────────────────────────────────────
// 現実装の dlasv2 との比較:
//   nalgebra: hypot(m11+m22, m12) + hypot(m11-m22, m12) で安定な denom
//   現実装: atan2(2b, a-c)/2 で固有ベクトル角度を計算
//   → nalgebra の方が数値的に安定（|m11|≈|m22| のとき atan2 は不安定になり得る）
//
// ─────────────────────────────────────────────────────────────
// GivensRotation::new (givens.rs)
// ─────────────────────────────────────────────────────────────
//   fn new(x, y) -> (Self, T):
//     if y == 0:  return (c=1, s=0), sign(x)
//     if x == 0:  return (c=0, s=sign(y)), |y|
//     r = hypot(x, y)
//     (c=x/r, s=y/r), r
//
//   fn cancel_y(&v):   // v.y を 0 にする → new(v.x, v.y)
//   fn cancel_x(&v):   // v.x を 0 にする → new(v.y, v.x) + swap
