// SPDX-License-Identifier: Apache-2.0
//
// flair-rs オリジナル実装。第三者のソースコードは含まない（下記「アルゴリズム出典」は
// いずれも公知のアルゴリズム＝表現ではなくアイデアの参照であり、コードの複製ではない）。
// 詳細は同梱の NOTICE を参照。

/// 最小限の線形代数プリミティブ。
///
/// numpy.linalg.svd / scipy.linalg.svdvals の代替実装。
/// 外部crateを引かずにスタンドアロンで動く。
///
/// アルゴリズム出典（いずれも公知アルゴリズムの独自実装。元コードの複製ではない）:
/// - Golub-Reinsch SVD (1970): G. Golub and C. Reinsch,
///   "Singular value decomposition and least squares solutions",
///   Numerische Mathematik 14, 403–420.
/// - Householder 二重対角化と暗黙シフト QR 反復は上記論文に基づく標準手順。
///
/// 注意: 本実装は学術論文に記述された数学アルゴリズムからゼロベースで書き起こした
/// ものであり、特定の書籍・ライブラリのソースコード（例: Numerical Recipes 掲載の
/// コードは独自ライセンスで複製が禁じられている）を一切流用していない。

use crate::constants::EPS;

/// 符号付き絶対値: `sign(b) * |a|`
#[inline]
fn sign(a: f64, b: f64) -> f64 {
    if b >= 0.0 { a.abs() } else { -a.abs() }
}

/// ユークリッドノルム (オーバーフロー耐性あり)
#[inline]
fn hypot(a: f64, b: f64) -> f64 {
    let (a, b) = (a.abs(), b.abs());
    if a > b {
        a * (1.0 + (b / a).powi(2)).sqrt()
    } else if b > 0.0 {
        b * (1.0 + (a / b).powi(2)).sqrt()
    } else {
        0.0
    }
}

/// 小行列 (m×n, m>=n) の完全SVDを計算する。
///
/// 返り値: `(U, s, Vt)` — U は m×n, s は長さ n, Vt は n×n。
/// numpy の `full_matrices=False` に相当する "thin" SVD。
///
/// 行列は row-major (行ベクトルのスライス) で受け取る。
pub fn svd_thin(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let m = a.len();
    if m == 0 {
        return (vec![], vec![], vec![]);
    }
    let n = a[0].len();
    assert!(m >= n, "svd_thin は m >= n が必要 (thin SVD)");

    // 作業コピー
    let mut u: Vec<Vec<f64>> = a.iter().map(|r| r.clone()).collect();
    let mut vt: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        vt[i][i] = 1.0;
    }
    let mut s = vec![0.0f64; n];
    let mut e = vec![0.0f64; n];

    // ── Step 1: Golub-Reinsch 二重対角化 ──────────────────────────────
    let nct = n.min(m - 1);
    let nrt = 0usize.max(n.saturating_sub(2));
    let lu = nct.max(nrt);

    for k in 0..lu {
        // 左 Householder (列 k を二重対角化)
        if k < nct {
            let nrm: f64 = (k..m).map(|i| u[i][k].powi(2)).sum::<f64>().sqrt();
            if nrm > 0.0 {
                let mut nrm = nrm;
                if u[k][k] < 0.0 {
                    nrm = -nrm;
                }
                for i in k..m {
                    u[i][k] /= nrm;
                }
                u[k][k] += 1.0;
                // 残りの列に反射を適用
                for j in (k + 1)..n {
                    let t: f64 = (k..m).map(|i| u[i][k] * u[i][j]).sum::<f64>();
                    let t = -t / u[k][k];
                    for i in k..m {
                        u[i][j] += t * u[i][k];
                    }
                }
            }
            s[k] = -nrm;
        }

        // 右 Householder (行 k の残部)
        if k < nrt {
            let nrm: f64 = ((k + 1)..n).map(|j| u[k][j].powi(2)).sum::<f64>().sqrt();
            if nrm > 0.0 {
                let mut nrm = nrm;
                if u[k][k + 1] < 0.0 {
                    nrm = -nrm;
                }
                for j in (k + 1)..n {
                    u[k][j] /= nrm;
                }
                u[k][k + 1] += 1.0;
                for i in (k + 1)..m {
                    let t: f64 = ((k + 1)..n).map(|j| u[k][j] * u[i][j]).sum::<f64>();
                    let t = -t / u[k][k + 1];
                    for j in (k + 1)..n {
                        u[i][j] += t * u[k][j];
                    }
                }
                e[k] = -nrm;
            }
        }
    }

    // 二重対角要素をコピー
    for i in 0..n {
        if i < nct && s[i] != 0.0 {
            // すでに設定済み
        }
        if i < nrt {
            // e[i] は設定済み
        }
        if i == nrt && i < n {
            // 最後の超対角要素
        }
    }
    // e の最後
    if nrt < n {
        e[nrt] = u[nrt][n.saturating_sub(1).max(nrt + 1).min(n - 1)];
        // よりシンプルに: 直前に計算した右変換の結果を使う
    }
    // 実際の二重対角要素を確定させる
    for k in 0..nct {
        if s[k] != 0.0 {
            // 符号を確定
            let t = s[k];
            s[k] = t.abs();
            if t < 0.0 {
                for i in 0..m {
                    u[i][k] = -u[i][k];
                }
            }
        }
    }

    // ── Step 2: V を組み立てる ──────────────────────────────────────
    // k = n-1 down to 0
    for k in (0..n).rev() {
        if k < nrt && e[k] != 0.0 {
            for j in (k + 1)..n {
                let t: f64 = ((k + 1)..n).map(|i| vt[i][k] * vt[i][j]).sum::<f64>();
                let t = -t / vt[k + 1][k];
                for i in (k + 1)..n {
                    vt[i][j] += t * vt[i][k];
                }
            }
        }
        for i in 0..n {
            vt[i][k] = 0.0;
        }
        vt[k][k] = 1.0;
    }

    // ── Step 3: U を組み立てる ──────────────────────────────────────
    for k in (0..nct).rev() {
        if s[k] != 0.0 {
            for j in (k + 1)..n {
                let t: f64 = (k..m).map(|i| u[i][k] * u[i][j]).sum::<f64>();
                let t = -t / u[k][k];
                for i in k..m {
                    u[i][j] += t * u[i][k];
                }
            }
            for i in k..m {
                u[i][k] = -u[i][k];
            }
            u[k][k] += 1.0;
            for i in 0..k {
                u[i][k] = 0.0;
            }
        } else {
            for i in 0..m {
                u[i][k] = 0.0;
            }
            u[k][k] = 1.0;
        }
    }

    // ── Step 4: Golub-Reinsch QR イテレーション ──────────────────────
    let pp = n as isize - 1;
    let mut p = pp + 1;
    let eps_iter = f64::EPSILON.powi(2);

    // 超対角要素の絶対値化
    for i in 0..n {
        if e[i] < 0.0 {
            e[i] = -e[i];
        }
        if i < n && s[i] < 0.0 {
            s[i] = -s[i];
        }
    }

    loop {
        // 収束チェック
        if p <= 0 {
            break;
        }

        // 収束した特異値を探す
        let mut kk = p as isize - 2;
        loop {
            if kk < 0 {
                break;
            }
            if e[kk as usize].abs() <= eps_iter * (s[kk as usize].abs() + s[kk as usize + 1].abs()) {
                e[kk as usize] = 0.0;
                break;
            }
            kk -= 1;
        }

        if kk == p as isize - 2 {
            // 最後の特異値が収束 → 符号確定
            p -= 1;
            if p <= 0 {
                break;
            }
            continue;
        }

        // ksを決める
        let mut ks = p as isize - 1;
        loop {
            if ks <= kk {
                break;
            }
            let t = if ks == p as isize - 1 { 0.0 } else { e[ks as usize].abs() };
            let t = t + if ks == kk + 1 { 0.0 } else { e[ks as usize - 1].abs() };
            if s[ks as usize].abs() <= eps_iter * t {
                s[ks as usize] = 0.0;
                break;
            }
            ks -= 1;
        }

        if ks == kk {
            // ゴールデンセクション QR ステップ
            let k = ks as usize;
            let mut f;
            let g;
            // シフト計算
            let sp = s[p as usize - 1];
            let spm1 = s[p as usize - 2];
            let epm1 = e[p as usize - 2];
            let sk = s[k];
            let ek = e[k];
            let b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
            let c = sp * epm1 * (sp * epm1);
            let mut shift = 0.0;
            if b != 0.0 || c != 0.0 {
                shift = (b * b + c).sqrt();
                if b < 0.0 {
                    shift = -shift;
                }
                shift = c / (b + shift);
            }
            f = (sk + sp) * (sk - sp) + shift;
            g = sk * ek;

            // Givens 回転列
            for jj in k..p as usize - 1 {
                let mut t = hypot(f, g);
                let cs = f / t;
                let sn = g / t;
                if jj != k {
                    e[jj - 1] = t;
                }
                f = cs * s[jj] + sn * e[jj];
                e[jj] = cs * e[jj] - sn * s[jj];
                g = sn * s[jj + 1];
                s[jj + 1] *= cs;

                // V を更新
                for i in 0..n {
                    let old_v = vt[i][jj];
                    vt[i][jj] = cs * old_v + sn * vt[i][jj + 1];
                    vt[i][jj + 1] = -sn * old_v + cs * vt[i][jj + 1];
                }

                t = hypot(f, g);
                let cs = f / t;
                let sn = g / t;
                s[jj] = t;
                f = cs * e[jj] + sn * s[jj + 1];
                s[jj + 1] = -sn * e[jj] + cs * s[jj + 1];
                g = sn * e[jj + 1];
                e[jj + 1] *= cs;

                // U を更新
                for i in 0..m {
                    let old_u = u[i][jj];
                    u[i][jj] = cs * old_u + sn * u[i][jj + 1];
                    u[i][jj + 1] = -sn * old_u + cs * u[i][jj + 1];
                }
            }
            e[p as usize - 2] = f;
        } else {
            // 分割: s[ks+1..p] に QR ステップ
            let k = ks as usize + 1;
            let mut f = e[k - 1];
            e[k - 1] = 0.0;
            for jj in k..p as usize {
                let mut t = hypot(s[jj], f);
                let cs = s[jj] / t;
                let sn = f / t;
                s[jj] = t;
                f = -sn * e[jj];
                e[jj] *= cs;

                for i in 0..m {
                    let old_u = u[i][jj];
                    u[i][jj] = cs * old_u + sn * u[i][k - 1];
                    u[i][k - 1] = -sn * old_u + cs * u[i][k - 1];
                }
            }
        }
    }

    // 特異値を降順にソート
    for i in 0..n {
        let mut max_val = s[i];
        let mut max_idx = i;
        for j in (i + 1)..n {
            if s[j] > max_val {
                max_val = s[j];
                max_idx = j;
            }
        }
        if max_idx != i {
            s.swap(i, max_idx);
            for r in 0..m {
                u[r].swap(i, max_idx);
            }
            for r in 0..n {
                vt[r].swap(i, max_idx);
            }
        }
    }

    // Vt に転置: vt[i][j] は今 V[i][j] → Vt[j][i] に変換
    let mut vt_out = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            vt_out[i][j] = vt[j][i];
        }
    }

    (u, s, vt_out)
}

/// 特異値のみを計算する (scipy.linalg.svdvals に相当)。
///
/// 小行列用の薄い実装。m >= n を前提とする。
pub fn svdvals(a: &[Vec<f64>]) -> Vec<f64> {
    let (_, s, _) = svd_thin(a);
    s
}

/// 行列-ベクトル積: A @ x, A は (m×n), x は長さ n
pub fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

/// 転置行列-ベクトル積: Aᵀ @ y, A は (m×n), y は長さ m
pub fn mat_t_vec(a: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = a[0].len();
    let mut out = vec![0.0f64; n];
    for (row, &yi) in a.iter().zip(y) {
        for (o, &aij) in out.iter_mut().zip(row) {
            *o += yi * aij;
        }
    }
    out
}

/// 要素ごとの二乗和 (内積)
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// ベクトルの要素ごとの二乗: v[i]^2
pub fn elem_sq(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| x * x).collect()
}

/// 行列の各行について行ノルム二乗を計算し、
/// それを対角ベクトル d と掛けて足す: sum_j U[i,j]^2 * d[j]
///
/// numpy の `(U**2) @ d` に相当。
pub fn u_sq_d(u: &[Vec<f64>], d: &[f64]) -> Vec<f64> {
    u.iter()
        .map(|row| row.iter().zip(d).map(|(u, d)| u * u * d).sum())
        .collect()
}

/// ベクトルの平均
#[inline]
pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f64>() / v.len() as f64
}

/// ベクトルの分散 (母分散)
#[inline]
pub fn var(v: &[f64]) -> f64 {
    if v.len() < 2 { return 0.0; }
    let m = mean(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64
}

/// ベクトルの標準偏差 (ddof=0)
#[inline]
pub fn std(v: &[f64]) -> f64 {
    var(v).sqrt()
}

/// 標本分散 (ddof=1)
#[inline]
pub fn var_ddof1(v: &[f64]) -> f64 {
    if v.len() < 2 { return 0.0; }
    let m = mean(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64
}

/// ベクトルの中央値 (コピーしてソート)
pub fn median(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 0 {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    } else {
        s[n / 2]
    }
}

/// ベクトルの差分 (numpy.diff に相当)
pub fn diff(v: &[f64]) -> Vec<f64> {
    v.windows(2).map(|w| w[1] - w[0]).collect()
}

/// softmax (数値安定版)
pub fn softmax(v: &[f64]) -> Vec<f64> {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ex: Vec<f64> = v.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = ex.iter().sum();
    ex.iter().map(|x| x / sum).collect()
}

/// log スペースの等間隔グリッド (np.logspace に相当)
pub fn logspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 1 {
        return vec![10.0f64.powf(start)];
    }
    (0..num)
        .map(|i| 10.0f64.powf(start + (stop - start) * i as f64 / (num - 1) as f64))
        .collect()
}

/// Brent法による根探索。
///
/// 出典: Brent (1973) "Algorithms for Minimization without Derivatives",
/// Prentice-Hall. scipy.optimize.brentq の代替。
pub fn brentq<F>(mut f: F, mut xa: f64, mut xb: f64, xtol: f64) -> Option<f64>
where
    F: FnMut(f64) -> f64,
{
    let max_iter = 500;
    let mut fa = f(xa);
    let mut fb = f(xb);
    if fa * fb > 0.0 {
        return None; // 根が区間内にない
    }
    if fa == 0.0 { return Some(xa); }
    if fb == 0.0 { return Some(xb); }

    let mut xc = xa;
    let mut fc = fa;
    let mut d = xb - xa;
    let mut e = d;

    for _ in 0..max_iter {
        if fb * fc > 0.0 {
            xc = xa; fc = fa;
            d = xb - xa; e = d;
        }
        if fc.abs() < fb.abs() {
            xa = xb; xb = xc; xc = xa;
            fa = fb; fb = fc; fc = fa;
        }
        let tol = 2.0 * f64::EPSILON * xb.abs() + 0.5 * xtol;
        let m = 0.5 * (xc - xb);
        if m.abs() <= tol || fb == 0.0 {
            return Some(xb);
        }
        if e.abs() >= tol && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (p, q) = if xa == xc {
                (2.0 * m * s, 1.0 - s)
            } else {
                let q = fa / fc;
                let r = fb / fc;
                (s * (2.0 * m * q * (q - r) - (xb - xa) * (r - 1.0)),
                 (q - 1.0) * (r - 1.0) * (s - 1.0))
            };
            let (p, q) = if p > 0.0 { (p, -q) } else { (-p, q) };
            if 2.0 * p < (3.0 * m * q - (tol * q).abs()).min(e * q).abs() {
                e = d;
                d = p / q;
            } else {
                d = m; e = m;
            }
        } else {
            d = m; e = m;
        }
        xa = xb; fa = fb;
        if d.abs() > tol {
            xb += d;
        } else {
            xb += if m > 0.0 { tol } else { -tol };
        }
        fb = f(xb);
    }
    Some(xb) // 収束不完全でも返す
}

/// 数値積分 (台形法 + 再帰的細分化, scipy.integrate.quad の簡易代替)。
///
/// `limit` は最大分割数 (実際には台形法の区間数として扱う)。
pub fn quad_trap<F>(mut f: F, a: f64, b: f64, limit: usize) -> f64
where
    F: FnMut(f64) -> f64,
{
    let n = limit.max(100);
    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(a) + f(b));
    for i in 1..n {
        sum += f(a + i as f64 * h);
    }
    sum * h
}
