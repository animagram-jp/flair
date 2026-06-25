// SPDX-License-Identifier: Apache-2.0
//
// このファイルは nalgebra v0.35.0 (Copyright 2020 Sébastien Crozet, Apache-2.0)
// の派生物（`Vec<f64>` への移植・改変）です。詳細は同梱の NOTICE を参照。

/// SVD (特異値分解) — nalgebra-0.35.0 のアルゴリズムを `Vec<f64>` に移植。
///
/// # ライセンス
/// 元コードは nalgebra v0.35.0 (Apache-2.0) に基づく。
/// 原典: <https://github.com/dimforge/nalgebra/blob/v0.35.0/src/linalg/svd.rs>
///       <https://github.com/dimforge/nalgebra/blob/v0.35.0/src/linalg/bidiagonal.rs>
///       <https://github.com/dimforge/nalgebra/blob/v0.35.0/src/linalg/givens.rs>
///
/// # 端部処理の方針
/// nalgebra の型システム（`ComplexField` trait、`DefaultAllocator`、
/// `OMatrix`/`OVector` ジェネリクス）は依存が芋づる式に広がるため、
/// すべて `Vec<f64>` ベースのスタンドアロン実装に置き換えた。
/// **アルゴリズムのロジック**（Householder 二重対角化 → Givens 回転 QR イテレーション）
/// は nalgebra の実装を直接移植している。
///
/// # 提供する関数
/// - [`svd`]     : thin SVD → `(U[m×n], s[n], Vt[n×n])`
/// - [`svdvals`] : 特異値のみ（`scipy.linalg.svdvals` 相当）

// ─────────────────────────────────────────────────────────────────────────────
// Givens 回転
//
// 出典: nalgebra/src/linalg/givens.rs (Apache-2.0)
// https://github.com/dimforge/nalgebra/blob/v0.35.0/src/linalg/givens.rs
// ─────────────────────────────────────────────────────────────────────────────

/// Givens 回転 `G` — `G * [a, b]^T = [r, 0]^T` を成立させる `(c, s)`。
#[derive(Clone, Copy, Debug)]
struct Givens {
    c: f64,
    s: f64,
}

impl Givens {
    fn identity() -> Self {
        Self { c: 1.0, s: 0.0 }
    }

    /// 正規化前の cosine `c`・sine `s` 成分から Givens 回転を構築し、norm を返す。
    ///
    /// nalgebra `GivensRotation::new` / `try_new`（eps=0）の f64 特殊化:
    /// `sign0 = signum(c)`, `denom = hypot(c, s)`,
    /// `norm = sign0·denom`, `c_out = |c|/denom`, `s_out = s/norm`。
    /// `denom == 0` のときは恒等回転と norm=0 を返す。
    fn new(c: f64, s: f64) -> (Self, f64) {
        let mod0 = c.abs();
        let sign0 = if c >= 0.0 { 1.0 } else { -1.0 };
        let denom = (mod0 * mod0 + s * s).sqrt();
        if denom > 0.0 {
            let norm = sign0 * denom;
            (Self { c: mod0 / denom, s: s / norm }, norm)
        } else {
            (Self::identity(), 0.0)
        }
    }

    /// `G * [a, b]^T = [r, 0]^T` を満たす回転と r を返す（`b == 0` なら None）。
    ///
    /// nalgebra `GivensRotation::cancel_y` の f64 特殊化:
    /// `c = |a|/denom`, `s = -b/(signum(a)·denom)`, `r = signum(a)·denom`,
    /// `denom = hypot(a, b)`。
    fn cancel_y(a: f64, b: f64) -> Option<(Self, f64)> {
        if b == 0.0 {
            return None;
        }
        let mod0 = a.abs();
        let sign0 = if a >= 0.0 { 1.0 } else { -1.0 };
        let denom = (mod0 * mod0 + b * b).sqrt();
        let c = mod0 / denom;
        let s = -b / (sign0 * denom);
        let r = sign0 * denom;
        Some((Self { c, s }, r))
    }

    /// `G * [a, b]^T = [0, r]^T` を満たす回転と r を返す（`a == 0` なら None）。
    ///
    /// nalgebra `GivensRotation::cancel_x` の f64 特殊化:
    /// `c = |b|/denom`, `s = a·signum(b)/denom`, `r = signum(b)·denom`,
    /// `denom = hypot(a, b)`。
    fn cancel_x(a: f64, b: f64) -> Option<(Self, f64)> {
        if a == 0.0 {
            return None;
        }
        let mod1 = b.abs();
        let sign1 = if b >= 0.0 { 1.0 } else { -1.0 };
        let denom = (mod1 * mod1 + a * a).sqrt();
        let c = mod1 / denom;
        let s = (a * sign1) / denom;
        let r = sign1 * denom;
        Some((Self { c, s }, r))
    }

    /// 逆回転（転置）。
    fn inverse(self) -> Self {
        Self { c: self.c, s: -self.s }
    }

    /// 列ペア `(ci, cj)` への右作用 `lhs <- lhs * G`。
    ///
    /// nalgebra `GivensRotation::rotate_rows` の f64 特殊化:
    /// `new_ci = c*ci + s*cj`, `new_cj = -s*ci + c*cj`
    fn rotate_rows(self, mat: &mut [Vec<f64>], ci: usize, cj: usize) {
        for row in mat.iter_mut() {
            let a = row[ci];
            let b = row[cj];
            row[ci] = self.c * a + self.s * b;
            row[cj] = -self.s * a + self.c * b;
        }
    }

    /// 行ペア `(ri, rj)` への左作用 `rhs <- G * rhs`。
    ///
    /// nalgebra `GivensRotation::rotate` の f64 特殊化:
    /// `new_ri = c*ri - s*rj`, `new_rj = s*ri + c*rj`
    /// （`rotate_rows` とは s の符号配置が異なる点に注意）
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

/// 局所 2×3 サブ行列 `subm` の列ペア `(c0, c1)` への右作用 `subm <- subm * G`。
///
/// nalgebra `rot.rotate_rows(subm.fixed_columns_mut::<2>(c0))` 相当（2 行固定）。
fn subm_rotate_rows(subm: &mut [[f64; 3]; 2], g: Givens, c0: usize, c1: usize) {
    for r in 0..2 {
        let a = subm[r][c0];
        let b = subm[r][c1];
        subm[r][c0] = g.c * a + g.s * b;
        subm[r][c1] = -g.s * a + g.c * b;
    }
}

/// 局所 2×3 サブ行列 `subm` の 2 行への左作用 `subm <- G * subm`、列範囲 `cstart..cend`。
///
/// nalgebra `rot.rotate(subm.fixed_columns_mut::<2>(cstart))` 相当。
fn subm_rotate(subm: &mut [[f64; 3]; 2], g: Givens, cstart: usize, cend: usize) {
    for col in cstart..cend {
        let a = subm[0][col];
        let b = subm[1][col];
        subm[0][col] = g.c * a - g.s * b;
        subm[1][col] = g.s * a + g.c * b;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Householder 反射 axis の生成
//
// 出典: nalgebra/src/linalg/householder.rs `reflection_axis_mut` (Apache-2.0)
// https://github.com/dimforge/nalgebra/blob/v0.35.0/src/linalg/householder.rs
//
// 端部処理: `ComplexField::to_exp` / `unscale_mut` / `normalize_mut` を
// f64 専用に展開。複素共役は実数では恒等なので省略。
// ─────────────────────────────────────────────────────────────────────────────

/// `column` を「`column` を `(±‖column‖, 0, …, 0)` に写す Householder 反射の
/// **単位 axis**」に置き換える。
///
/// 返り値: `(reflection_norm, not_zero)`。
/// `reflection_norm` は反射後の第1成分（= diag/off に入る符号付きノルム）。
/// `not_zero == false` のとき反射は不要（axis は使われない）。
///
/// nalgebra `reflection_axis_mut` の f64 特殊化。
fn reflection_axis_mut(column: &mut [f64]) -> (f64, bool) {
    let reflection_sq_norm: f64 = column.iter().map(|x| x * x).sum();
    let reflection_norm = reflection_sq_norm.sqrt();

    // to_exp(): real では (modulus=|x|, sign=signum*(±1), x=0 なら sign=1)
    let x0 = column[0];
    let sign = if x0 >= 0.0 { 1.0 } else { -1.0 };
    let modulus = x0.abs();
    let signed_norm = sign * reflection_norm;
    let factor = (reflection_sq_norm + modulus * reflection_norm) * 2.0;
    column[0] += signed_norm;

    if factor != 0.0 {
        let inv = 1.0 / factor.sqrt();
        for c in column.iter_mut() {
            *c *= inv;
        }
        // 2 段階目の正規化（nalgebra コメント参照: 単位ベクトル性を厳密化）
        let nrm: f64 = column.iter().map(|x| x * x).sum::<f64>().sqrt();
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

/// 単位 axis による Householder 反射を行列の列群に適用する。
///
/// `col_new = sign · (I − 2·axis·axisᵀ) · col_old`
/// nalgebra `Reflection::reflect_with_sign` の f64 特殊化（bias = 0）。
/// `mat` の行範囲 `from_row..` × 列範囲 `from_col..` に作用する。
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
        // factor = sign · (-2) · dot ; col <- factor·axis + sign·col
        let factor = sign * (-2.0) * dot;
        for (i, ai) in axis.iter().enumerate() {
            mat[from_row + i][j] = factor * ai + sign * mat[from_row + i][j];
        }
    }
}

/// 単位 axis による Householder 反射を行列の行群に適用する（右側 / Vt 用）。
///
/// `row_new = sign · row_old · (I − 2·axis·axisᵀ)`
/// nalgebra `Reflection::reflect_rows_with_sign` の f64 特殊化（bias = 0）。
/// `mat` の行範囲 `from_row..` × 列範囲 `from_col..` に作用する。
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
// Householder 二重対角化
//
// 出典: nalgebra/src/linalg/bidiagonal.rs `Bidiagonal::new` /
//       `clear_column_unchecked` / `clear_row_unchecked` (Apache-2.0)
// https://github.com/dimforge/nalgebra/blob/v0.35.0/src/linalg/bidiagonal.rs
//
// 端部処理: `OMatrix` を `Vec<Vec<f64>>` に置き換え。
// アルゴリズム（reflection_axis_mut → reflect_with_sign）は原典に厳密一致。
// ─────────────────────────────────────────────────────────────────────────────

/// `a` を Householder 反射で上二重対角形式に変換し、
/// U, Vt 生成に必要な**単位 axis** 群を返す。
///
/// 返り値:
/// - `diag`: 対角要素（符号付き、長さ n）
/// - `off` : 超対角要素（符号付き、長さ n-1）
/// - `u_axes` : U 側の単位 axis 群（各 `(col, from_row, axis)`）
/// - `vt_axes`: Vt 側の単位 axis 群（各 `(row, from_col, axis)`）
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

    // nalgebra の `Bidiagonal::new` は upper_diagonal = (m >= n) で分岐する。
    // flair の対象行列は m >= n なので upper_diagonal = true のパスのみ実装する。
    assert!(m >= n, "bidiagonalize: m >= n が必要");

    for k in 0..n {
        // ── 左 Householder: 列 k の行 k.. をゼロ化（clear_column_unchecked）──
        let mut axis: Vec<f64> = (k..m).map(|i| work[i][k]).collect();
        let (reflection_norm, not_zero) = reflection_axis_mut(&mut axis);
        diag[k] = reflection_norm;
        if not_zero {
            // 右側部分行列 work[k.., k+1..] に sign = signum(reflection_norm) で反射
            let sign = if reflection_norm >= 0.0 { 1.0 } else { -1.0 };
            reflect_cols_with_sign(&mut work, &axis, k, k + 1, n, sign);
            u_axes.push((k, k, axis));
        }

        // ── 右 Householder: 行 k の列 k+1.. をゼロ化（clear_row_unchecked）──
        if k + 1 < n {
            let mut axis: Vec<f64> = (k + 1..n).map(|j| work[k][j]).collect();
            let (reflection_norm, not_zero) = reflection_axis_mut(&mut axis);
            // nalgebra は axis.conjugate_mut() するが real では恒等。
            off[k] = reflection_norm;
            if not_zero {
                // 下側部分行列 work[k+1.., k+1..] に
                // sign = signum(reflection_norm).conjugate() = signum で反射（行作用）
                let sign = if reflection_norm >= 0.0 { 1.0 } else { -1.0 };
                reflect_rows_with_sign(&mut work, &axis, k + 1, m, k + 1, sign);
                vt_axes.push((k, k + 1, axis));
            }
        }
    }

    (diag, off, u_axes, vt_axes)
}

/// 保存された単位 axis 群から U (m×n) を組み立てる。
///
/// nalgebra `Bidiagonal::u()` に厳密一致:
/// 逆順に各反射を `reflect_with_sign(sign = signum(diag[i]))` で適用する。
fn assemble_u(
    m: usize,
    n: usize,
    diag: &[f64],
    u_axes: &[(usize, usize, Vec<f64>)],
) -> Vec<Vec<f64>> {
    // res = identity_generic(m, min(m,n)=n) の m×n 単位行列
    let mut u = vec![vec![0.0f64; n]; m];
    for i in 0..n {
        u[i][i] = 1.0;
    }
    for &(col, from_row, ref axis) in u_axes.iter().rev() {
        let sign = if diag[col] >= 0.0 { 1.0 } else { -1.0 };
        // res.view_range_mut(from_row.., col..) に列作用
        reflect_cols_with_sign(&mut u, axis, from_row, col, n, sign);
    }
    u
}

/// 保存された単位 axis 群から Vt (n×n) を組み立てる。
///
/// nalgebra `Bidiagonal::v_t()` に厳密一致:
/// 逆順に各反射を `reflect_rows_with_sign(sign = signum(off[i]))` で適用する。
fn assemble_vt(
    n: usize,
    off: &[f64],
    vt_axes: &[(usize, usize, Vec<f64>)],
) -> Vec<Vec<f64>> {
    // res = identity_generic(min(m,n)=n, n) の n×n 単位行列
    let mut vt = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        vt[i][i] = 1.0;
    }
    for &(row, from_col, ref axis) in vt_axes.iter().rev() {
        let sign = if off[row] >= 0.0 { 1.0 } else { -1.0 };
        // res.view_range_mut(row.., from_col..) に行作用
        reflect_rows_with_sign(&mut vt, axis, row, n, from_col, sign);
    }
    vt
}

// ─────────────────────────────────────────────────────────────────────────────
// Wilkinson シフト
//
// 出典: nalgebra/src/linalg/symmetric_eigen.rs `wilkinson_shift` (Apache-2.0)
// https://github.com/dimforge/nalgebra/blob/v0.35.0/src/linalg/symmetric_eigen.rs
// ─────────────────────────────────────────────────────────────────────────────

/// 2×2 対称行列 `[[tmm, tmn], [tmn, tnn]]` の `tnn` に近い固有値（Wilkinson シフト）。
#[inline]
fn wilkinson_shift(tmm: f64, tnn: f64, tmn: f64) -> f64 {
    let d = (tmm - tnn) * 0.5;
    let sign_d = if d >= 0.0 { 1.0 } else { -1.0 };
    tnn - sign_d * tmn * tmn / (d.abs() + (d * d + tmn * tmn).sqrt())
}

// ─────────────────────────────────────────────────────────────────────────────
// 2×2 上三角 SVD
//
// 出典: nalgebra/src/linalg/svd.rs `compute_2x2_uptrig_svd` (Apache-2.0)
// 論文: Qiao & Wang, "Computing the Singular Values of 2-by-2 Complex Matrices"
// http://www.cas.mcmaster.ca/sqrl/papers/sqrl5.pdf
// ─────────────────────────────────────────────────────────────────────────────

/// 2×2 上三角行列 `[[m11, m12], [0, m22]]` の SVD。
///
/// 返り値: `(u_rot, [s1, s2], vt_rot)` — s1, s2 は特異値。
fn svd_2x2_uptrig(
    m11: f64,
    m12: f64,
    m22: f64,
    compute_u: bool,
    compute_v: bool,
) -> (Option<Givens>, [f64; 2], Option<Givens>) {
    let denom = (m11 + m22).hypot(m12) + (m11 - m22).hypot(m12);

    // v1 は m22 に最も近い特異値（cancellation 回避のため; 原典 NOTE 参照）。
    let mut v1 = m11 * m22 * 2.0 / denom;
    let mut v2 = 0.5 * denom;

    let mut u_rot = None;
    let mut v_rot = None;

    if compute_u || compute_v {
        // 原典は GivensRotation::new（cancel_y ではない）。norm（sgn_v）をそのまま乗算。
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
// サブ問題の区切り + Givens でのゼロ化
//
// 出典: nalgebra/src/linalg/svd.rs `delimit_subproblem` /
//       `cancel_horizontal_off_diagonal_elt` /
//       `cancel_vertical_off_diagonal_elt` (Apache-2.0)
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
            // upper_diagonal = true → U 側を更新
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
            // upper_diagonal = true → Vt 側を更新
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
        if off[m].abs() <= eps * (diag[n].abs() + diag[m].abs()) {
            off[m] = 0.0;
        } else if diag[m].abs() <= eps {
            diag[m] = 0.0;
            cancel_horizontal(diag, off, u, vt, m, m + 1);
            if m != 0 {
                cancel_vertical(diag, off, u, vt, m - 1);
            }
        } else if diag[n].abs() <= eps {
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
        if off[m].abs() <= eps * (diag[new_start].abs() + diag[m].abs()) {
            off[m] = 0.0;
            break;
        } else if diag[m].abs() <= eps {
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
// 公開 API
// ─────────────────────────────────────────────────────────────────────────────

/// Thin SVD: `A = U * diag(s) * Vt`、特異値は降順。
///
/// - `a` : m×n 行列（行ベクトルのスライス）、`m >= n` を前提とする
/// - 返り値: `(U[m×n], s[n], Vt[n×n])`
///
/// `numpy.linalg.svd(a, full_matrices=False)` と等価。
pub fn svd(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    svd_impl(a, true, true)
}

/// 特異値のみを返す（U, Vt は計算しない）。
///
/// `scipy.linalg.svdvals(a)` と等価。
pub fn svdvals(a: &[Vec<f64>]) -> Vec<f64> {
    let (_, s, _) = svd_impl(a, false, false);
    s
}

// ─────────────────────────────────────────────────────────────────────────────
// 内部実装
//
// 出典: nalgebra/src/linalg/svd.rs `SVD::try_new_unordered` (Apache-2.0)
// ─────────────────────────────────────────────────────────────────────────────

fn svd_impl(
    a: &[Vec<f64>],
    compute_u: bool,
    compute_v: bool,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let m = a.len();
    assert!(m > 0, "空行列は SVD できない");
    let n = a[0].len();
    assert!(m >= n, "svd: m >= n が必要 (thin SVD)");
    let dim = n;

    // ── Step 1: camax スケーリング（nalgebra `try_new_unordered` 冒頭）────
    let amax = a
        .iter()
        .flat_map(|r| r.iter())
        .cloned()
        .fold(0.0_f64, |acc, x| acc.max(x.abs()));
    let scale = if amax == 0.0 { 1.0 } else { amax };
    let a_scaled: Vec<Vec<f64>> = a
        .iter()
        .map(|r| r.iter().map(|x| x / scale).collect())
        .collect();

    // ── Step 2: Householder 二重対角化 ────────────────────────────────────
    let (mut diag, mut off, u_refl, vt_refl) = bidiagonalize(&a_scaled);

    // ── Step 3: U, Vt を組み立てる ────────────────────────────────────────
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

    // 対角・超対角要素を絶対値化（符号は U/Vt に反映済み）
    for d in diag.iter_mut() {
        *d = d.abs();
    }
    for e in off.iter_mut() {
        *e = e.abs();
    }

    // ── Step 4: QR イテレーション（Golub-Reinsch implicit-shift）────────
    // 出典: nalgebra/src/linalg/svd.rs `SVD::try_new_unordered` メインループ
    let eps = f64::EPSILON * 5.0;

    let (mut start, mut end) = delimit_subproblem(
        &mut diag, &mut off, &mut u_mat, &mut vt_mat, dim - 1, eps,
    );

    // nalgebra `SVD::new` は max_niter=0（無制限）で呼ぶが、ここでは
    // 万一の非収束に備えた保険として十分大きな上限を設ける。
    let max_niter = 100 * dim + 1000;
    let mut niter = 0usize;

    while end != start {
        let subdim = end - start + 1;

        if subdim > 2 {
            // Wilkinson シフト付き implicit QR ステップ
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

                // 2×3 サブ行列 subm（nalgebra `Matrix2x3`）:
                //   [[d[k], off[k], 0], [0, d[k+1], m12]]
                // 原典は subm.fixed_columns_mut::<2>(j) のビューに対し
                // rotate_rows（列ペア右作用）/ rotate（2 行左作用）を呼ぶ。
                let mut subm: [[f64; 3]; 2] = [
                    [diag[k], off[k], 0.0],
                    [0.0, diag[k + 1], m12],
                ];

                if let Some((rot1, norm1)) = Givens::cancel_y(vx, vy) {
                    // rot1.inverse().rotate_rows(subm.cols(0,1))
                    //   列ペア (0,1) への右作用 [c,+s / -s,c]（inverse で s 反転）
                    let inv1 = rot1.inverse();
                    subm_rotate_rows(&mut subm, inv1, 0, 1);

                    if k > start {
                        off[k - 1] = norm1;
                    }

                    // rot2 = cancel_y(subm[0][0], subm[1][0])
                    let (rot2, norm2) = Givens::cancel_y(subm[0][0], subm[1][0])
                        .unwrap_or((Givens::identity(), subm[0][0]));

                    // rot2.rotate(subm.cols(1,2)): 2 行 (0,1) への左作用 [c,-s / s,c]
                    //   ただし列範囲は 1..3 に限定
                    subm_rotate(&mut subm, rot2, 1, 3);
                    subm[0][0] = norm2;

                    // Vt 更新（upper_diagonal=true → rot1.rotate(v_t.rows(k,k+1))）
                    if let Some(ref mut vt) = vt_mat {
                        rot1.rotate(vt, k, k + 1);
                    }
                    // U 更新（upper_diagonal=true → rot2.inverse().rotate_rows(u.cols(k,k+1))）
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
            // 残り 2×2 の closed-form SVD
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
                rot.rotate_rows(u, start, start + 1);
            }
            if let (Some(vt), Some(rot)) = (&mut vt_mat, v2) {
                rot.inverse().rotate(vt, start, start + 1);
            }
            end -= 1;
        }

        let sub = delimit_subproblem(
            &mut diag, &mut off, &mut u_mat, &mut vt_mat, end, eps,
        );
        start = sub.0;
        end = sub.1;

        niter += 1;
        assert!(niter < max_niter, "svd: QR イテレーションが収束しませんでした");
    }

    // ── Step 5: スケールを戻す ────────────────────────────────────────────
    for d in diag.iter_mut() {
        *d *= scale;
    }

    // 負の特異値を正にして U の対応列の符号を反転
    for i in 0..dim {
        if diag[i] < 0.0 {
            diag[i] = -diag[i];
            if let Some(ref mut u) = u_mat {
                for row in u.iter_mut() {
                    row[i] = -row[i];
                }
            }
        }
    }

    // ── Step 6: 降順ソート（nalgebra `sort_by_singular_values`）─────────
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
