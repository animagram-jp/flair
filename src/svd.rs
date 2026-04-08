//! Singular Value Decomposition (Golub-Reinsch algorithm)
//!
//! Pure Rust implementation of dense SVD using:
//! - GEBRD (Householder bidiagonalization)
//! - DBDSQR (Givens rotation iteration)
//!
//! Reference: Golub, G.H., Reinsch, C. (1970)
//!           "Singular value decomposition and least squares solutions"
//!           Numerische Mathematik, 14(5), 403-420

use core::cmp::Ordering;
use alloc::{
    vec,
    vec::Vec,
};
use crate::SvdError as Error;
use libm::{sqrt, pow};

type Result<T> = core::result::Result<T, Error>;

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
pub fn full(a: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>)> {
    // Validate input
    if a.is_empty() {
        return Err(Error::InvalidInput("empty matrix"));
    }
    let _m = a.len();
    let n = a[0].len();

    if n == 0 {
        return Err(Error::InvalidInput("empty matrix"));
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
fn gebrd(a: &mut Vec<Vec<f64>>) -> Result<BidigState> {
    let m = a.len();
    if m == 0 {
        return Err(Error::InvalidInput("empty matrix"));
    }
    let n = a[0].len();
    if n == 0 {
        return Err(Error::InvalidInput("empty matrix"));
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

/// Golub-Reinsch iteration on bidiagonal matrix
///
/// Stage 2 of Golub-Reinsch SVD.
/// Applies Givens rotations to iteratively diagonalize the bidiagonal matrix.
///
/// Algorithm:
/// 1. Initialize U and V as identity matrices
/// 2. For each bidiagonal element that is not converged:
///    a. Find submatrix with largest superdiagonal element
///    b. Apply QR iteration (Givens rotations) to reduce off-diagonal
///    c. Accumulate rotations into U and V
/// 3. Continue until all superdiagonal elements are sufficiently small
///
/// Returns (singular_values, U, V^T)
fn dbdsqr(state: &BidigState, max_iter: usize) -> Result<(Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let m = state.u.len();
    let n = state.v.len();
    let k = state.d.len();

    // Initialize result matrices
    let mut u = state.u.clone();
    let mut v = state.v.clone();
    let mut d = state.d.clone();
    let mut e = state.e.clone();

    // Pad e with zeros to make it length k
    while e.len() < k {
        e.push(0.0);
    }

    const EPS: f64 = 2.22e-16; // Machine epsilon for f64
    const TOL: f64 = 100.0; // Convergence tolerance factor

    let mut iter = 0;
    let mut p = k - 1; // Current block size

    while p > 0 && iter < max_iter {
        iter += 1;

        // [Fix 1] Check convergence at p first, then find active block bottom q.
        // Deflate trailing converged singular values before searching for q.
        if e[p - 1].abs() <= TOL * EPS * (d[p - 1].abs() + d[p].abs()) {
            e[p - 1] = 0.0;
            p -= 1;
            if p == 0 {
                break;
            }
            continue;
        }

        // Find the largest q < p such that e[q-1] is negligible (active block: q..=p)
        let mut q = p - 1;
        while q > 0 {
            if e[q - 1].abs() <= TOL * EPS * (d[q - 1].abs() + d[q].abs()) {
                e[q - 1] = 0.0;
                break;
            }
            q -= 1;
        }

        // QR iteration on the bidiagonal matrix from q to p
        // Compute shift using Wilkinson's criterion
        let shift = compute_qr_shift(&d, &e, q, p);

        // Apply Givens rotations: chase the bulge from q to p.
        // Each step does: right rotation (into V) then left rotation (into U).
        // [Fix 2] Uniform loop — no special-cased final iteration.
        let mut x = d[q] * d[q] - shift;
        let mut y = d[q] * e[q];

        for i in q..p {
            // Right rotation G_R: zeros the bulge entry, applied to V
            let (cr, sr) = givens_params(x, y);
            apply_givens_v(&mut v, cr, sr, i, i + 1, n);

            // Update bidiagonal: right rotation on columns i, i+1
            let f = cr * d[i] + sr * e[i];
            e[i]     = -sr * d[i] + cr * e[i];
            let g = sr * d[i + 1];
            d[i + 1] = cr * d[i + 1];
            d[i] = f;

            // Left rotation G_L: zeros the sub-diagonal bulge g, applied to U
            let (cl, sl) = givens_params(f, g);
            apply_givens_u(&mut u, cl, sl, i, i + 1, m);

            // Update bidiagonal: left rotation on rows i, i+1
            d[i]     = cl * d[i] + sl * g;
            let h    = cl * e[i] + sl * d[i + 1];
            d[i + 1] = -sl * e[i] + cl * d[i + 1];
            e[i] = h;

            // Carry the new bulge for the next iteration
            if i + 1 < p {
                x = e[i];
                y = sl * e[i + 1];
                e[i + 1] = cl * e[i + 1];
            }
        }
    }

    // [Fix 3] Ensure non-negative singular values after all iterations complete.
    for i in 0..k {
        if d[i] < 0.0 {
            d[i] = -d[i];
            for j in 0..n {
                v[j][i] = -v[j][i];
            }
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

/// Compute QR shift using Wilkinson's shift strategy
fn compute_qr_shift(d: &[f64], e: &[f64], q: usize, p: usize) -> f64 {
    if p == q {
        return 0.0;
    }

    // Use 2x2 block at bottom of matrix to compute shift
    let d_p = d[p];
    let d_pm = d[p - 1];
    let e_pm = if p - 1 < e.len() { e[p - 1] } else { 0.0 };

    let a = d_pm * d_pm;
    let b = e_pm * e_pm;
    let c = d_p * d_p;

    let trace = a + c;
    let det = a * c - b * b;

    // Eigenvalues of 2x2 block
    let discriminant = pow(trace / 2.0, 2.0) - det;
    if discriminant < 0.0 {
        return trace / 2.0;
    }

    let sqrt_disc = sqrt(discriminant);
    let ev1 = trace / 2.0 + sqrt_disc;
    let ev2 = trace / 2.0 - sqrt_disc;

    // Return eigenvalue closer to c
    if (ev1 - c).abs() < (ev2 - c).abs() {
        ev1
    } else {
        ev2
    }
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
fn givens_params(a: f64, b: f64) -> (f64, f64) {
    if b.abs() < 1e-30 {
        return (1.0, 0.0);
    }

    if a.abs() < 1e-30 {
        return (0.0, -b.abs() / b);
    }

    let r = sqrt(a * a + b * b);

    let c = a / r;
    let s = b / r;

    (c, s)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::println;

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
}
