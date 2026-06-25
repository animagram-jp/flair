// SPDX-License-Identifier: Apache-2.0
//
// Derived from nalgebra v0.35.0 (Copyright 2020 Sébastien Crozet, Apache-2.0).
// Ported to Vec<f64>-based standalone implementation. See NOTICE for details.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use libm::{sqrt, fabs, hypot};

/// Thin SVD — nalgebra-0.35.0 algorithm ported to `Vec<f64>`.
///
/// Origin:
/// - <https://github.com/dimforge/nalgebra/blob/v0.35.0/src/linalg/svd.rs>
/// - <https://github.com/dimforge/nalgebra/blob/v0.35.0/src/linalg/bidiagonal.rs>
/// - <https://github.com/dimforge/nalgebra/blob/v0.35.0/src/linalg/givens.rs>
///
/// nalgebra's type system (`ComplexField`, `DefaultAllocator`, `OMatrix`/`OVector` generics)
/// pulls in a large dependency tree, so all types are replaced with plain `Vec<f64>`.
/// The algorithm itself (Householder bidiagonalization → Givens QR iteration) is a
/// direct port of the nalgebra implementation.
///
/// Public API:
/// - [`svd`]     : thin SVD → `(U[m×n], s[n], Vt[n×n])`
/// - [`svdvals`] : singular values only (equivalent to `scipy.linalg.svdvals`)

// ─────────────────────────────────────────────────────────────────────────────
// Givens rotation
//
// Origin: nalgebra/src/linalg/givens.rs (Apache-2.0)
// https://github.com/dimforge/nalgebra/blob/v0.35.0/src/linalg/givens.rs
// ─────────────────────────────────────────────────────────────────────────────

/// Givens rotation `G` such that `G * [a, b]^T = [r, 0]^T`.
#[derive(Clone, Copy, Debug)]
struct Givens {
    c: f64,
    s: f64,
}

impl Givens {
    fn identity() -> Self {
        Self { c: 1.0, s: 0.0 }
    }

    /// Constructs a Givens rotation from unnormalized (c, s) and returns the norm.
    ///
    /// Port of nalgebra `GivensRotation::new` / `try_new` (eps=0) for f64:
    /// `sign0 = signum(c)`, `denom = hypot(c, s)`,
    /// `norm = sign0·denom`, `c_out = |c|/denom`, `s_out = s/norm`.
    /// Returns identity rotation and norm=0 when `denom == 0`.
    fn new(c: f64, s: f64) -> (Self, f64) {
        let mod0 = fabs(c);
        let sign0 = if c >= 0.0 { 1.0 } else { -1.0 };
        let denom = sqrt(mod0 * mod0 + s * s);
        if denom > 0.0 {
            let norm = sign0 * denom;
            (Self { c: mod0 / denom, s: s / norm }, norm)
        } else {
            (Self::identity(), 0.0)
        }
    }

    /// Returns `(G, r)` such that `G * [a, b]^T = [r, 0]^T`, or `None` if `b == 0`.
    ///
    /// Port of nalgebra `GivensRotation::cancel_y` for f64:
    /// `c = |a|/denom`, `s = -b/(signum(a)·denom)`, `r = signum(a)·denom`.
    fn cancel_y(a: f64, b: f64) -> Option<(Self, f64)> {
        if b == 0.0 {
            return None;
        }
        let mod0 = fabs(a);
        let sign0 = if a >= 0.0 { 1.0 } else { -1.0 };
        let denom = sqrt(mod0 * mod0 + b * b);
        let c = mod0 / denom;
        let s = -b / (sign0 * denom);
        let r = sign0 * denom;
        Some((Self { c, s }, r))
    }

    /// Returns `(G, r)` such that `G * [a, b]^T = [0, r]^T`, or `None` if `a == 0`.
    ///
    /// Port of nalgebra `GivensRotation::cancel_x` for f64:
    /// `c = |b|/denom`, `s = a·signum(b)/denom`, `r = signum(b)·denom`.
    fn cancel_x(a: f64, b: f64) -> Option<(Self, f64)> {
        if a == 0.0 {
            return None;
        }
        let mod1 = fabs(b);
        let sign1 = if b >= 0.0 { 1.0 } else { -1.0 };
        let denom = sqrt(mod1 * mod1 + a * a);
        let c = mod1 / denom;
        let s = (a * sign1) / denom;
        let r = sign1 * denom;
        Some((Self { c, s }, r))
    }

    /// Inverse (transpose) of this rotation.
    fn inverse(self) -> Self {
        Self { c: self.c, s: -self.s }
    }

    /// Right-applies this rotation to column pair `(ci, cj)`: `lhs <- lhs * G`.
    ///
    /// Port of nalgebra `GivensRotation::rotate_rows` for f64:
    /// `new_ci = c*ci + s*cj`, `new_cj = -s*ci + c*cj`.
    fn rotate_rows(self, mat: &mut [Vec<f64>], ci: usize, cj: usize) {
        for row in mat.iter_mut() {
            let a = row[ci];
            let b = row[cj];
            row[ci] = self.c * a + self.s * b;
            row[cj] = -self.s * a + self.c * b;
        }
    }

    /// Left-applies this rotation to row pair `(ri, rj)`: `rhs <- G * rhs`.
    ///
    /// Port of nalgebra `GivensRotation::rotate` for f64:
    /// `new_ri = c*ri - s*rj`, `new_rj = s*ri + c*rj`
    /// (note opposite sign convention from `rotate_rows`).
    fn rotate(self, mat: &mut [Vec<f64>], ri: usize, rj: usize) {
        let ncols = mat[0].len();
        for k in 0..ncols {
            let a = mat[ri][k];
            let b = mat[rj][k];
            mat[ri][k] = self.c * a - self.s * b;
            mat[rj][k] = self.s * a + self.c * b;
        }
    }
}

/// Right-applies `G` to column pair `(c0, c1)` of a local 2×3 submatrix.
///
/// Equivalent to nalgebra `rot.rotate_rows(subm.fixed_columns_mut::<2>(c0))`.
fn subm_rotate_rows(subm: &mut [[f64; 3]; 2], g: Givens, c0: usize, c1: usize) {
    for r in 0..2 {
        let a = subm[r][c0];
        let b = subm[r][c1];
        subm[r][c0] = g.c * a + g.s * b;
        subm[r][c1] = -g.s * a + g.c * b;
    }
}

/// Left-applies `G` to rows 0..2 of a local 2×3 submatrix over columns `cstart..cend`.
///
/// Equivalent to nalgebra `rot.rotate(subm.fixed_columns_mut::<2>(cstart))`.
fn subm_rotate(subm: &mut [[f64; 3]; 2], g: Givens, cstart: usize, cend: usize) {
    for col in cstart..cend {
        let a = subm[0][col];
        let b = subm[1][col];
        subm[0][col] = g.c * a - g.s * b;
        subm[1][col] = g.s * a + g.c * b;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Householder reflection axis construction
//
// Origin: nalgebra/src/linalg/householder.rs `reflection_axis_mut` (Apache-2.0)
// https://github.com/dimforge/nalgebra/blob/v0.35.0/src/linalg/householder.rs
//
// `ComplexField::to_exp` / `unscale_mut` / `normalize_mut` are expanded for f64.
// Complex conjugate is identity for reals and omitted.
// ─────────────────────────────────────────────────────────────────────────────

/// Replaces `column` in-place with the unit Householder axis that maps
/// `column` to `(±‖column‖, 0, …, 0)`.
///
/// Returns `(reflection_norm, not_zero)`.
/// `reflection_norm` is the signed norm placed on the diagonal/off-diagonal.
/// When `not_zero == false` the reflection is unnecessary.
///
/// Port of nalgebra `reflection_axis_mut` for f64.
fn reflection_axis_mut(column: &mut [f64]) -> (f64, bool) {
    let reflection_sq_norm: f64 = column.iter().map(|x| x * x).sum();
    let reflection_norm = sqrt(reflection_sq_norm);

    // to_exp() for reals: modulus = |x|, sign = signum(x) (sign = 1 when x = 0)
    let x0 = column[0];
    let sign = if x0 >= 0.0 { 1.0 } else { -1.0 };
    let modulus = fabs(x0);
    let signed_norm = sign * reflection_norm;
    let factor = (reflection_sq_norm + modulus * reflection_norm) * 2.0;
    column[0] += signed_norm;

    if factor != 0.0 {
        let inv = 1.0 / sqrt(factor);
        for c in column.iter_mut() {
            *c *= inv;
        }
        // Second normalization pass to enforce exact unit length (see nalgebra comment).
        let nrm: f64 = sqrt(column.iter().map(|x| x * x).sum::<f64>());
        if nrm > 0.0 {
            let inv2 = 1.0 / nrm;
            for c in column.iter_mut() {
                *c *= inv2;
            }
        }
        (-signed_norm, true)
    } else {
        (signed_norm, false)
    }
}

/// Applies a Householder reflection defined by a unit axis to the columns of `mat`.
///
/// `col_new = sign · (I − 2·axis·axisᵀ) · col_old`
///
/// Port of nalgebra `Reflection::reflect_with_sign` for f64 (bias = 0).
/// Acts on `mat[from_row..][from_col..ncols]`.
fn reflect_cols_with_sign(
    mat: &mut [Vec<f64>],
    axis: &[f64],
    from_row: usize,
    from_col: usize,
    ncols: usize,
    sign: f64,
) {
    for j in from_col..ncols {
        let dot: f64 = axis
            .iter()
            .enumerate()
            .map(|(i, ai)| ai * mat[from_row + i][j])
            .sum();
        let factor = sign * (-2.0) * dot;
        for (i, ai) in axis.iter().enumerate() {
            mat[from_row + i][j] = factor * ai + sign * mat[from_row + i][j];
        }
    }
}

/// Applies a Householder reflection defined by a unit axis to the rows of `mat` (right-side / Vt).
///
/// `row_new = sign · row_old · (I − 2·axis·axisᵀ)`
///
/// Port of nalgebra `Reflection::reflect_rows_with_sign` for f64 (bias = 0).
/// Acts on `mat[from_row..nrows][from_col..]`.
fn reflect_rows_with_sign(
    mat: &mut [Vec<f64>],
    axis: &[f64],
    from_row: usize,
    nrows: usize,
    from_col: usize,
    sign: f64,
) {
    for i in from_row..nrows {
        let dot: f64 = axis
            .iter()
            .enumerate()
            .map(|(j, aj)| aj * mat[i][from_col + j])
            .sum();
        let factor = sign * (-2.0) * dot;
        for (j, aj) in axis.iter().enumerate() {
            mat[i][from_col + j] = factor * aj + sign * mat[i][from_col + j];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Householder bidiagonalization
//
// Origin: nalgebra/src/linalg/bidiagonal.rs
//         `Bidiagonal::new` / `clear_column_unchecked` / `clear_row_unchecked` (Apache-2.0)
// https://github.com/dimforge/nalgebra/blob/v0.35.0/src/linalg/bidiagonal.rs
//
// `OMatrix` replaced with `Vec<Vec<f64>>`. Only the upper-diagonal (m >= n) path is
// implemented, which is the only case used in flair.
// ─────────────────────────────────────────────────────────────────────────────

/// Reduces `a` to upper bidiagonal form using Householder reflections.
///
/// Returns:
/// - `diag`    : diagonal elements (signed, length n)
/// - `off`     : superdiagonal elements (signed, length n-1)
/// - `u_axes`  : U-side unit axes `(col, from_row, axis)`
/// - `vt_axes` : Vt-side unit axes `(row, from_col, axis)`
fn bidiagonalize(
    a: &[Vec<f64>],
) -> (Vec<f64>, Vec<f64>, Vec<(usize, usize, Vec<f64>)>, Vec<(usize, usize, Vec<f64>)>) {
    let m = a.len();
    let n = a[0].len();
    let mut work: Vec<Vec<f64>> = a.to_vec();
    let mut diag = vec![0.0f64; n];
    let mut off = vec![0.0f64; n.saturating_sub(1)];
    let mut u_axes: Vec<(usize, usize, Vec<f64>)> = Vec::new();
    let mut vt_axes: Vec<(usize, usize, Vec<f64>)> = Vec::new();

    assert!(m >= n, "bidiagonalize: requires m >= n");

    for k in 0..n {
        // Left Householder: zero out column k below row k (clear_column_unchecked)
        let mut axis: Vec<f64> = (k..m).map(|i| work[i][k]).collect();
        let (reflection_norm, not_zero) = reflection_axis_mut(&mut axis);
        diag[k] = reflection_norm;
        if not_zero {
            let sign = if reflection_norm >= 0.0 { 1.0 } else { -1.0 };
            reflect_cols_with_sign(&mut work, &axis, k, k + 1, n, sign);
            u_axes.push((k, k, axis));
        }

        // Right Householder: zero out row k right of column k+1 (clear_row_unchecked)
        if k + 1 < n {
            let mut axis: Vec<f64> = (k + 1..n).map(|j| work[k][j]).collect();
            let (reflection_norm, not_zero) = reflection_axis_mut(&mut axis);
            // nalgebra calls axis.conjugate_mut() here; identity for reals.
            off[k] = reflection_norm;
            if not_zero {
                let sign = if reflection_norm >= 0.0 { 1.0 } else { -1.0 };
                reflect_rows_with_sign(&mut work, &axis, k + 1, m, k + 1, sign);
                vt_axes.push((k, k + 1, axis));
            }
        }
    }

    (diag, off, u_axes, vt_axes)
}

/// Assembles U (m×n) from the stored unit axes.
///
/// Port of nalgebra `Bidiagonal::u()`: applies each reflection in reverse order
/// with `sign = signum(diag[i])`.
fn assemble_u(
    m: usize,
    n: usize,
    diag: &[f64],
    u_axes: &[(usize, usize, Vec<f64>)],
) -> Vec<Vec<f64>> {
    let mut u = vec![vec![0.0f64; n]; m];
    for i in 0..n {
        u[i][i] = 1.0;
    }
    for &(col, from_row, ref axis) in u_axes.iter().rev() {
        let sign = if diag[col] >= 0.0 { 1.0 } else { -1.0 };
        reflect_cols_with_sign(&mut u, axis, from_row, col, n, sign);
    }
    u
}

/// Assembles Vt (n×n) from the stored unit axes.
///
/// Port of nalgebra `Bidiagonal::v_t()`: applies each reflection in reverse order
/// with `sign = signum(off[i])`.
fn assemble_vt(
    n: usize,
    off: &[f64],
    vt_axes: &[(usize, usize, Vec<f64>)],
) -> Vec<Vec<f64>> {
    let mut vt = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        vt[i][i] = 1.0;
    }
    for &(row, from_col, ref axis) in vt_axes.iter().rev() {
        let sign = if off[row] >= 0.0 { 1.0 } else { -1.0 };
        reflect_rows_with_sign(&mut vt, axis, row, n, from_col, sign);
    }
    vt
}

// ─────────────────────────────────────────────────────────────────────────────
// Wilkinson shift
//
// Origin: nalgebra/src/linalg/symmetric_eigen.rs `wilkinson_shift` (Apache-2.0)
// https://github.com/dimforge/nalgebra/blob/v0.35.0/src/linalg/symmetric_eigen.rs
// ─────────────────────────────────────────────────────────────────────────────

/// Wilkinson shift: eigenvalue of `[[tmm, tmn], [tmn, tnn]]` closest to `tnn`.
#[inline]
fn wilkinson_shift(tmm: f64, tnn: f64, tmn: f64) -> f64 {
    let d = (tmm - tnn) * 0.5;
    let sign_d = if d >= 0.0 { 1.0 } else { -1.0 };
    tnn - sign_d * tmn * tmn / (fabs(d) + sqrt(d * d + tmn * tmn))
}

// ─────────────────────────────────────────────────────────────────────────────
// 2×2 upper-triangular SVD
//
// Origin: nalgebra/src/linalg/svd.rs `compute_2x2_uptrig_svd` (Apache-2.0)
// Reference: Qiao & Wang, "Computing the Singular Values of 2-by-2 Complex Matrices"
// http://www.cas.mcmaster.ca/sqrl/papers/sqrl5.pdf
// ─────────────────────────────────────────────────────────────────────────────

/// SVD of a 2×2 upper-triangular matrix `[[m11, m12], [0, m22]]`.
///
/// Returns `(u_rot, [s1, s2], vt_rot)`.
fn svd_2x2_uptrig(
    m11: f64,
    m12: f64,
    m22: f64,
    compute_u: bool,
    compute_v: bool,
) -> (Option<Givens>, [f64; 2], Option<Givens>) {
    let denom = hypot(m11 + m22, m12) + hypot(m11 - m22, m12);

    // v1 is the singular value closest to m22 (avoids cancellation; see paper NOTE).
    let mut v1 = m11 * m22 * 2.0 / denom;
    let mut v2 = 0.5 * denom;

    let mut u_rot = None;
    let mut v_rot = None;

    if compute_u || compute_v {
        // Origin uses GivensRotation::new (not cancel_y); sgn_v is multiplied directly.
        let (csv, sgn_v) = Givens::new(m11 * m12, v1 * v1 - m11 * m11);
        v1 *= sgn_v;
        v2 *= sgn_v;

        if compute_v {
            v_rot = Some(csv);
        }

        let cu = (m11 * csv.c + m12 * csv.s) / v1;
        let su = (m22 * csv.s) / v1;
        let (csu, sgn_u) = Givens::new(cu, su);
        v1 *= sgn_u;
        v2 *= sgn_u;

        if compute_u {
            u_rot = Some(csu);
        }
    }

    (u_rot, [v1, v2], v_rot)
}

// ─────────────────────────────────────────────────────────────────────────────
// Subproblem splitting + Givens zeroing
//
// Origin: nalgebra/src/linalg/svd.rs `delimit_subproblem` /
//         `cancel_horizontal_off_diagonal_elt` /
//         `cancel_vertical_off_diagonal_elt` (Apache-2.0)
// ─────────────────────────────────────────────────────────────────────────────

fn cancel_horizontal(
    diag: &mut [f64],
    off: &mut [f64],
    u: &mut Option<Vec<Vec<f64>>>,
    _vt: &mut Option<Vec<Vec<f64>>>,
    i: usize,
    end: usize,
) {
    let mut vx = off[i];
    let mut vy = diag[i + 1];
    off[i] = 0.0;
    for k in i..end {
        if let Some((rot, norm)) = Givens::cancel_x(vx, vy) {
            diag[k + 1] = norm;
            // upper_diagonal = true → update U side
            if let Some(u) = u {
                rot.inverse().rotate_rows(u, i, k + 1);
            }
            if k + 1 != end {
                vx = -rot.s * off[k + 1];
                vy = diag[k + 2];
                off[k + 1] *= rot.c;
            }
        } else {
            break;
        }
    }
}

fn cancel_vertical(
    diag: &mut [f64],
    off: &mut [f64],
    _u: &mut Option<Vec<Vec<f64>>>,
    vt: &mut Option<Vec<Vec<f64>>>,
    i: usize,
) {
    let mut vx = diag[i];
    let mut vy = off[i];
    off[i] = 0.0;
    for k in (0..=i).rev() {
        if let Some((rot, norm)) = Givens::cancel_y(vx, vy) {
            diag[k] = norm;
            // upper_diagonal = true → update Vt side
            if let Some(vt) = vt {
                rot.rotate(vt, k, i + 1);
            }
            if k > 0 {
                vx = diag[k - 1];
                vy = rot.s * off[k - 1];
                off[k - 1] *= rot.c;
            }
        } else {
            break;
        }
    }
}

fn delimit_subproblem(
    diag: &mut Vec<f64>,
    off: &mut Vec<f64>,
    u: &mut Option<Vec<Vec<f64>>>,
    vt: &mut Option<Vec<Vec<f64>>>,
    end: usize,
    eps: f64,
) -> (usize, usize) {
    let mut n = end;
    while n > 0 {
        let m = n - 1;
        if fabs(off[m]) <= eps * (fabs(diag[n]) + fabs(diag[m])) {
            off[m] = 0.0;
        } else if fabs(diag[m]) <= eps {
            diag[m] = 0.0;
            cancel_horizontal(diag, off, u, vt, m, m + 1);
            if m != 0 {
                cancel_vertical(diag, off, u, vt, m - 1);
            }
        } else if fabs(diag[n]) <= eps {
            diag[n] = 0.0;
            cancel_vertical(diag, off, u, vt, m);
        } else {
            break;
        }
        n -= 1;
    }

    if n == 0 {
        return (0, 0);
    }

    let mut new_start = n - 1;
    while new_start > 0 {
        let m = new_start - 1;
        if fabs(off[m]) <= eps * (fabs(diag[new_start]) + fabs(diag[m])) {
            off[m] = 0.0;
            break;
        } else if fabs(diag[m]) <= eps {
            diag[m] = 0.0;
            cancel_horizontal(diag, off, u, vt, m, n);
            if m != 0 {
                cancel_vertical(diag, off, u, vt, m - 1);
            }
            break;
        }
        new_start -= 1;
    }

    (new_start, n)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Thin SVD: `A = U * diag(s) * Vt`, singular values in descending order.
///
/// - `a`: m×n matrix (slice of row vectors); accepts m >= n and m < n
/// - Returns: `(U[m×k], s[k], Vt[k×n])` where k = min(m, n)
///
/// Equivalent to `numpy.linalg.svd(a, full_matrices=False)`.
pub fn svd(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let m = a.len();
    let n = if m > 0 { a[0].len() } else { 0 };
    if m >= n {
        svd_impl(a, true, true)
    } else {
        // m < n: compute on A^T (n×m) and swap U, Vt
        // svd(A^T) = (U', s, Vt')  →  svd(A) = (Vt'^T, s, U'^T)
        let at = transpose(a, m, n);
        let (u_t, s, vt_t) = svd_impl(&at, true, true);
        let u = transpose(&vt_t, n, m);   // Vt'^T: m×k
        let vt = transpose(&u_t, m, n);   // U'^T:  k×n
        (u, s, vt)
    }
}

/// Returns singular values only (U and Vt are not computed).
///
/// Equivalent to `scipy.linalg.svdvals(a)`.
pub fn svdvals(a: &[Vec<f64>]) -> Vec<f64> {
    let m = a.len();
    let n = if m > 0 { a[0].len() } else { 0 };
    if m >= n {
        let (_, s, _) = svd_impl(a, false, false);
        s
    } else {
        let at = transpose(a, m, n);
        let (_, s, _) = svd_impl(&at, false, false);
        s
    }
}

fn transpose(a: &[Vec<f64>], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0f64; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j][i] = a[i][j];
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal implementation
//
// Origin: nalgebra/src/linalg/svd.rs `SVD::try_new_unordered` (Apache-2.0)
// ─────────────────────────────────────────────────────────────────────────────

fn svd_impl(
    a: &[Vec<f64>],
    compute_u: bool,
    compute_v: bool,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let m = a.len();
    assert!(m > 0, "cannot compute SVD of empty matrix");
    let n = a[0].len();
    assert!(m >= n, "svd_impl: requires m >= n (thin SVD)");
    let dim = n;

    // Step 1: scale by amax (nalgebra `try_new_unordered` prologue)
    let amax = a
        .iter()
        .flat_map(|r: &Vec<f64>| r.iter())
        .cloned()
        .fold(0.0_f64, |acc: f64, x: f64| if fabs(x) > acc { fabs(x) } else { acc });
    let scale = if amax == 0.0 { 1.0 } else { amax };
    let a_scaled: Vec<Vec<f64>> = a
        .iter()
        .map(|r: &Vec<f64>| r.iter().map(|x| x / scale).collect())
        .collect();

    // Step 2: Householder bidiagonalization
    let (mut diag, mut off, u_refl, vt_refl) = bidiagonalize(&a_scaled);

    // Step 3: assemble U and Vt
    let mut u_mat: Option<Vec<Vec<f64>>> = if compute_u {
        Some(assemble_u(m, dim, &diag, &u_refl))
    } else {
        None
    };
    let mut vt_mat: Option<Vec<Vec<f64>>> = if compute_v {
        Some(assemble_vt(dim, &off, &vt_refl))
    } else {
        None
    };

    // Signs are absorbed into U/Vt; take absolute values on the bidiagonal.
    for d in diag.iter_mut() {
        *d = fabs(*d);
    }
    for e in off.iter_mut() {
        *e = fabs(*e);
    }

    // Step 4: QR iteration (Golub-Reinsch implicit-shift)
    // Origin: nalgebra/src/linalg/svd.rs `SVD::try_new_unordered` main loop
    let eps = f64::EPSILON * 5.0;

    let (mut start, mut end) = delimit_subproblem(
        &mut diag, &mut off, &mut u_mat, &mut vt_mat, dim - 1, eps,
    );

    // nalgebra uses max_niter=0 (unlimited); we cap with a large bound as a safety net.
    let max_niter = 100 * dim + 1000;
    let mut niter = 0usize;

    while end != start {
        let subdim = end - start + 1;

        if subdim > 2 {
            // Implicit QR step with Wilkinson shift
            let m_idx = end - 1;
            let n_idx = end;

            let dm = diag[m_idx];
            let dn = diag[n_idx];
            let fm = off[m_idx];
            let tmm = dm * dm + if m_idx > 0 { off[m_idx - 1] * off[m_idx - 1] } else { 0.0 };
            let tmn = dm * fm;
            let tnn = dn * dn + fm * fm;
            let shift = wilkinson_shift(tmm, tnn, tmn);

            let mut vx = diag[start] * diag[start] - shift;
            let mut vy = diag[start] * off[start];

            for k in start..n_idx {
                let m12 = if k == n_idx - 1 { 0.0 } else { off[k + 1] };

                // Local 2×3 submatrix (nalgebra `Matrix2x3`):
                //   [[d[k], off[k], 0], [0, d[k+1], m12]]
                // nalgebra applies rotate_rows / rotate to fixed-column views of this submatrix.
                let mut subm: [[f64; 3]; 2] = [
                    [diag[k], off[k], 0.0],
                    [0.0, diag[k + 1], m12],
                ];

                if let Some((rot1, norm1)) = Givens::cancel_y(vx, vy) {
                    // rot1.inverse().rotate_rows(subm.cols(0,1)): right-apply to column pair (0,1)
                    let inv1 = rot1.inverse();
                    subm_rotate_rows(&mut subm, inv1, 0, 1);

                    if k > start {
                        off[k - 1] = norm1;
                    }

                    let (rot2, norm2) = Givens::cancel_y(subm[0][0], subm[1][0])
                        .unwrap_or((Givens::identity(), subm[0][0]));

                    // rot2.rotate(subm.cols(1,2)): left-apply to rows 0..2, columns 1..3
                    subm_rotate(&mut subm, rot2, 1, 3);
                    subm[0][0] = norm2;

                    // Update Vt (upper_diagonal=true → rot1.rotate(v_t.rows(k, k+1)))
                    if let Some(ref mut vt) = vt_mat {
                        rot1.rotate(vt, k, k + 1);
                    }
                    // Update U (upper_diagonal=true → rot2.inverse().rotate_rows(u.cols(k, k+1)))
                    if let Some(ref mut u) = u_mat {
                        rot2.inverse().rotate_rows(u, k, k + 1);
                    }

                    diag[k] = subm[0][0];
                    diag[k + 1] = subm[1][1];
                    off[k] = subm[0][1];
                    if k != n_idx - 1 {
                        off[k + 1] = subm[1][2];
                    }
                    vx = subm[0][1];
                    vy = subm[0][2];
                } else {
                    break;
                }
            }
        } else if subdim == 2 {
            // Closed-form SVD for the remaining 2×2 block
            let (u2, s2, v2) = svd_2x2_uptrig(
                diag[start],
                off[start],
                diag[start + 1],
                compute_u,
                compute_v,
            );
            diag[start] = s2[0];
            diag[start + 1] = s2[1];
            off[start] = 0.0;

            if let (Some(u), Some(rot)) = (&mut u_mat, u2) {
                rot.rotate_rows(u as &mut Vec<Vec<f64>>, start, start + 1);
            }
            if let (Some(vt), Some(rot)) = (&mut vt_mat, v2) {
                rot.inverse().rotate(vt as &mut Vec<Vec<f64>>, start, start + 1);
            }
            end -= 1;
        }

        let sub = delimit_subproblem(
            &mut diag, &mut off, &mut u_mat, &mut vt_mat, end, eps,
        );
        start = sub.0;
        end = sub.1;

        niter += 1;
        assert!(niter < max_niter, "svd: QR iteration did not converge");
    }

    // Step 5: restore scale
    for d in diag.iter_mut() {
        *d *= scale;
    }

    // Flip sign of any negative singular values, reflecting into the corresponding U column.
    for i in 0..dim {
        if diag[i] < 0.0 {
            diag[i] = -diag[i];
            if let Some(ref mut u) = u_mat {
                for row in (u as &mut Vec<Vec<f64>>).iter_mut() {
                    row[i] = -row[i];
                }
            }
        }
    }

    // Step 6: sort descending (nalgebra `sort_by_singular_values`)
    let mut idx: Vec<usize> = (0..dim).collect();
    idx.sort_unstable_by(|&a, &b| {
        diag[b].partial_cmp(&diag[a]).unwrap()
    });

    let sorted_s: Vec<f64> = idx.iter().map(|&i| diag[i]).collect();

    let sorted_u = u_mat.map(|u| {
        let mut su = vec![vec![0.0f64; dim]; m];
        for (new_j, &old_j) in idx.iter().enumerate() {
            for i in 0..m {
                su[i][new_j] = u[i][old_j];
            }
        }
        su
    });

    let sorted_vt = vt_mat.map(|vt| {
        let mut svt = vec![vec![0.0f64; n]; dim];
        for (new_i, &old_i) in idx.iter().enumerate() {
            svt[new_i] = vt[old_i].clone();
        }
        svt
    });

    (
        sorted_u.unwrap_or_default(),
        sorted_s,
        sorted_vt.unwrap_or_default(),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — compared against nalgebra reference
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::println;

    fn nalgebra_singvals(a: &[Vec<f64>]) -> Vec<f64> {
        use nalgebra::DMatrix;
        let m = a.len();
        let n = a[0].len();
        let dm = DMatrix::from_fn(m, n, |r, c| a[r][c]);
        let sv = dm.svd(false, false);
        sv.singular_values.iter().copied().collect()
    }

    fn reconstruction_error(a: &[Vec<f64>], u: &[Vec<f64>], s: &[f64], vt: &[Vec<f64>]) -> f64 {
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
        err.sqrt()
    }

    fn orthogonality_error_u(u: &[Vec<f64>]) -> f64 {
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
        err.sqrt()
    }

    fn orthogonality_error_vt(vt: &[Vec<f64>]) -> f64 {
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
        err.sqrt()
    }

    fn assert_svd_quality(label: &str, a: &[Vec<f64>], tol_recon: f64, tol_orth: f64) {
        let (u, s, vt) = svd(a);
        let recon   = reconstruction_error(a, &u, &s, &vt);
        let orth_u  = orthogonality_error_u(&u);
        let orth_vt = orthogonality_error_vt(&vt);
        assert!(recon   < tol_recon, "{label}: reconstruction error {recon:.2e} >= {tol_recon:.2e}");
        assert!(orth_u  < tol_orth,  "{label}: U orthogonality {orth_u:.2e} >= {tol_orth:.2e}");
        assert!(orth_vt < tol_orth,  "{label}: Vt orthogonality {orth_vt:.2e} >= {tol_orth:.2e}");
    }

    #[test]
    fn test_svd_basic() {
        let a = vec![vec![1.0_f64, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let (u, s, vt) = svd(&a);
        assert_eq!(u.len(), 3);
        assert_eq!(u[0].len(), 2);
        assert_eq!(s.len(), 2);
        assert_eq!(vt.len(), 2);
        assert_eq!(vt[0].len(), 2);
        assert!(s[0] >= s[1], "singular values not in descending order");
        assert!(s[0] > 0.0);
        println!("test_svd_basic: s = {:?}", s);
    }

    #[test]
    fn test_svdvals_basic() {
        let a = vec![vec![1.0_f64, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let s = svdvals(&a);
        assert_eq!(s.len(), 2);
        assert!(s[0] >= s[1]);
        assert!(s[0] > 0.0);
        println!("test_svdvals_basic: s = {:?}", s);
    }

    #[test]
    fn test_against_nalgebra() {
        // nalgebra reference: [9.52551809, 0.51430058]
        let a = vec![vec![1.0_f64, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let s = svdvals(&a);
        assert!((s[0] - 9.5255).abs() < 1e-4, "s[0] expected ~9.5255, got {}", s[0]);
        assert!((s[1] - 0.5143).abs() < 1e-4, "s[1] expected ~0.5143, got {}", s[1]);

        // Verify β = Vt^T * diag(1/s) * U^T * y recovers OLS coefficients.
        let x_rows: Vec<Vec<f64>> = (0..30).map(|i| vec![1.0, i as f64 / 30.0]).collect();
        let y_lin: Vec<f64> = x_rows.iter().map(|r| 2.0 + 3.0 * r[1]).collect();
        let (u, s2, vt) = svd(&x_rows);
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

    /// Accuracy of the trailing 2×2 block handler.
    #[test]
    fn svd_2x2_trailing_block() {
        let cases: &[(&str, Vec<Vec<f64>>)] = &[
            ("3x3-trailing-2x2",
             vec![vec![1.0f64, 0.0, 0.0],
                  vec![0.0,    1.0, 1.0],
                  vec![0.0,    1.0, -1.0]]),
            ("3x3-high-cond",
             vec![vec![1e3f64, 1.0,  0.0],
                  vec![0.0,    1.0,  1.0],
                  vec![0.0,    1e-3, 1.0]]),
            ("3x3-dense",
             vec![vec![4.0f64, 3.0, 2.0],
                  vec![3.0,    2.0, 1.0],
                  vec![1.0,    1.0, 5.0]]),
        ];
        for (label, a) in cases {
            let s_ref = nalgebra_singvals(a);
            let s = svdvals(a);
            for (i, (&sr, &sf)) in s_ref.iter().zip(s.iter()).enumerate() {
                let err = if sr > 1e-8 { (sr - sf).abs() / sr } else { (sr - sf).abs() };
                assert!(err < 1e-6,
                    "{label} s[{i}]: nalgebra={sr:.6e} flair={sf:.6e} err={err:.2e}");
            }
            assert_svd_quality(label, a, 1e-11, 1e-11);
        }
    }

    /// High-condition-number diagonal matrix.
    #[test]
    fn svd_high_condition_number() {
        let scales = [1e6f64, 1e3, 1.0, 1e-3];
        let a: Vec<Vec<f64>> = (0..4).map(|i|
            (0..4).map(|j| if i == j { scales[i] } else { 0.0 }).collect()
        ).collect();
        let s_ref = nalgebra_singvals(&a);
        let s = svdvals(&a);
        for (i, (&sr, &sf)) in s_ref.iter().zip(s.iter()).enumerate() {
            let rel = (sr - sf).abs() / sr.max(1e-15);
            assert!(rel < 1e-6,
                "high-cond s[{i}]: nalgebra={sr:.6e} flair={sf:.6e} rel={rel:.2e}");
        }
        assert_svd_quality("high-cond", &a, 1e-11, 1e-11);
    }

    /// Rank-1 matrix.
    #[test]
    fn svd_rank1_matrix() {
        let u_vec = [1.0f64, 2.0, 3.0];
        let v_vec = [4.0f64, 5.0, 6.0];
        let a: Vec<Vec<f64>> = (0..3).map(|i|
            (0..3).map(|j| u_vec[i] * v_vec[j]).collect()
        ).collect();
        let s = svdvals(&a);
        let expected_s0 = (u_vec.iter().map(|&v| v * v).sum::<f64>()).sqrt()
                        * (v_vec.iter().map(|&v| v * v).sum::<f64>()).sqrt();
        assert!((s[0] - expected_s0).abs() / expected_s0 < 1e-8,
            "rank-1 s[0]: expected {expected_s0:.6} got {:.6}", s[0]);
        assert!(s[1] < 1e-8, "rank-1 s[1] should be ~0, got {}", s[1]);
        assert!(s[2] < 1e-8, "rank-1 s[2] should be ~0, got {}", s[2]);
        assert_svd_quality("rank-1", &a, 1e-10, 1e-10);
    }

    /// Near-singular matrix.
    #[test]
    fn svd_near_singular() {
        let eps = 1e-10f64;
        let a = vec![vec![1.0f64, 1.0], vec![1.0, 1.0 + eps]];
        let s_ref = nalgebra_singvals(&a);
        let s = svdvals(&a);
        let rel0 = (s_ref[0] - s[0]).abs() / s_ref[0];
        assert!(rel0 < 1e-8, "near-singular s[0] rel={rel0:.2e}");
        let abs1 = (s_ref[1] - s[1]).abs();
        assert!(abs1 < 1e-10,
            "near-singular s[1] abs={abs1:.2e} (ref={:.4e} got={:.4e})", s_ref[1], s[1]);
    }

    /// Rank-1 dominant 12×10 matrix (optshrink scenario).
    #[test]
    fn svd_rank1_dominant_period_matrix() {
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
        let s = svdvals(&a);
        let rel0 = (s_ref[0] - s[0]).abs() / s_ref[0];
        assert!(rel0 < 1e-4, "period-matrix s[0] rel={rel0:.2e}");
        for (i, (&sr, &sf)) in s_ref.iter().zip(s.iter()).enumerate() {
            let rel = (sr - sf).abs() / sr.max(1e-6);
            assert!(rel < 0.10,
                "period-matrix s[{i}]: nalgebra={sr:.4} flair={sf:.4} rel={rel:.4}");
        }
        assert_svd_quality("period-matrix", &a, 1e-8, 1e-8);
    }

    /// Tall matrix 50×3 (ridge_sa scenario).
    #[test]
    fn svd_tall_ridge_design() {
        let a: Vec<Vec<f64>> = (0..50).map(|i| {
            let t = i as f64 / 50.0;
            vec![1.0, t, (t * 6.28318).sin()]
        }).collect();
        let s_ref = nalgebra_singvals(&a);
        let s = svdvals(&a);
        for (i, (&sr, &sf)) in s_ref.iter().zip(s.iter()).enumerate() {
            let rel = (sr - sf).abs() / sr.max(1e-10);
            assert!(rel < 1e-6,
                "tall-ridge s[{i}]: nalgebra={sr:.6} flair={sf:.6} rel={rel:.2e}");
        }
        assert_svd_quality("tall-ridge", &a, 1e-10, 1e-10);
    }

    /// Full reconstruction vs nalgebra: verifies (U, s, Vt) reconstruction and orthogonality.
    #[test]
    fn svd_full_reconstruction_vs_nalgebra() {
        let cases: &[(&str, Vec<Vec<f64>>)] = &[
            ("3x2", vec![vec![1.0f64,2.0],vec![3.0,4.0],vec![5.0,6.0]]),
            ("4x3", vec![vec![1.0f64,2.0,3.0],vec![4.0,5.0,6.0],
                         vec![7.0,8.0,9.0],vec![10.0,11.0,12.0]]),
            ("5x5-rand", vec![vec![2.0f64,1.0,0.5,0.25,0.1],
                              vec![1.0,3.0,1.0,0.5, 0.2],
                              vec![0.5,1.0,4.0,1.0, 0.3],
                              vec![0.25,0.5,1.0,5.0,0.4],
                              vec![0.1,0.2,0.3,0.4,6.0]]),
        ];
        for (label, a) in cases {
            let s_ref = nalgebra_singvals(a);
            let s = svdvals(a);
            for (i, (&sr, &sf)) in s_ref.iter().zip(s.iter()).enumerate() {
                let rel = (sr - sf).abs() / sr.max(1e-12);
                assert!(rel < 1e-8,
                    "{label} s[{i}]: nalgebra={sr:.8e} flair={sf:.8e} rel={rel:.2e}");
            }
            assert_svd_quality(label, a, 1e-10, 1e-10);
        }
    }
}
