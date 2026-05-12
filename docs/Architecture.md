// This file includes untranslated text (ja).

# Architecture

## Arguments

```rust
// y:         &[f64]   時系列データ
// freq:      &Freq    データの周波数
// horizon:   usize    何ステップ先まで予測するか
// n_samples: usize    サンプルパス本数（精度と速度のトレードオフ）
// seed:      u64      乱数シード（再現性のため）
// quantiles: &[f64]   各パーセンタイル（0.0〜1.0）

pub fn seed_from_time() -> u64;

pub fn forecast(
    y: &[f64],
    freq: &Freq,
    horizon: usize,
    n_samples: usize,
    seed: u64,
    covariates: Option<(&[f64], &[f64])>,  // (x_historical, x_future) row-major, len = n*k と horizon*k
) -> Result<(Vec<Vec<f64>>, Confidence), Error> {
    // y が空でないこと
    if y.is_empty() { return Err(Error::InvalidInput("y must not be empty")); }
    // horizon >= 1
    if horizon < 1 { return Err(Error::InvalidInput("horizon must be >= 1")); }
    // n_samples >= 1
    if n_samples < 1 { return Err(Error::InvalidInput("n_samples must be >= 1")); }
    // covariates の長さ整合性
    if let Some((x_hist, x_future)) = covariates {
        if x_hist.is_empty() || x_future.is_empty() {
            return Err(Error::InvalidInput("covariates must not be empty"));
        }
        // k = x_hist.len() / y.len() が割り切れること
        if x_hist.len() % y.len() != 0 {
            return Err(Error::InvalidInput("x_historical length must be a multiple of y length"));
        }
        let k = x_hist.len() / y.len();
        // x_future の長さ = horizon * k であること
        if x_future.len() != horizon * k {
            return Err(Error::InvalidInput("x_future length must equal horizon * k"));
        }
    }
    todo!()
}

pub fn forecast_mean(
    y: &[f64],
    freq: &Freq,
    horizon: usize,
    n_samples: usize,
    seed: u64,
    covariates: Option<(&[f64], &[f64])>,
) -> Result<(Vec<f64>, Confidence), Error> {
    let (samples, conf) = forecast(y, freq, horizon, n_samples, seed, covariates)?;
    let mean = (0..horizon)
        .map(|h| samples.iter().map(|s| s[h]).sum::<f64>() / n_samples as f64)
        .collect();
    Ok((mean, conf))
}

pub fn forecast_quantiles(
    y: &[f64],
    freq: &Freq,
    horizon: usize,
    n_samples: usize,
    seed: u64,
    quantiles: &[f64],
    covariates: Option<(&[f64], &[f64])>,
) -> Result<(Vec<Vec<f64>>, Confidence), Error> {
    // quantiles がNoneやNaN(IEEE 754)でないこと、各値が 0.0〜1.0 の範囲内であること
    if quantiles.is_empty() || quantiles.iter().any(|&q| !(0.0..=1.0).contains(&q)) {
        return Err(Error::InvalidInput("quantiles must be non-empty and each value in [0.0, 1.0]"));
    }
    let (samples, conf) = forecast(y, freq, horizon, n_samples, seed, covariates)?;
    let result = quantiles.iter().map(|&q| {
        (0..horizon).map(|h| {
            let mut col: Vec<f64> = samples.iter().map(|s| s[h]).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let idx = (q * (n_samples - 1) as f64).round() as usize;
            col[idx]
        }).collect()
    }).collect();
    Ok((result, conf))
}

pub struct Confidence {
    pub rank1: Option<f64>,  // σ₁²/Σσᵢ² — rank-1構造の強さ（1.0=完全な単一周期性、~1/P=フラット）
    pub gamma: Option<f64>,  // 季節強度（ランダム行列ベースライン除去後）。1.0=強い、0.0=なし
    pub gcv:   Option<f64>,  // RidgeのLOOCV最小誤差 — Levelの予測可能性。低いほど良い
    // rank1とgammaがNoneになるケース:
    //   - Yearly（P=1、季節分解なし）
    //   - 系列が短くてMIN_COMPLETE周期に満たない
}

pub enum Error {
    InvalidFreq(usize),   // Freq::new() に無効な間隔が渡された
    InvalidInput(&'static str),   // y が空など
    SvdError,             // SVD が収束しなかった
}

pub enum Freq {
    Secondly(usize),  // 有効値: 10
    Minutely(usize),  // 有効値: 5, 10, 15, 30
    Hourly(usize),    // 有効値: 1, 12
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Yearly,
}

impl Freq {
    pub fn secondly(n: usize) -> Result<Self, Error> {
        match n {
            10 => Ok(Freq::Secondly(n)),
            _ => Err(Error::InvalidFreq(n)),
        }
    }
    pub fn minutely(n: usize) -> Result<Self, Error> {
        match n {
            5 | 10 | 15 | 30 => Ok(Freq::Minutely(n)),
            _ => Err(Error::InvalidFreq(n)),
        }
    }
    pub fn hourly(n: usize) -> Result<Self, Error> {
        match n {
            1 | 12 => Ok(Freq::Hourly(n)),
            _ => Err(Error::InvalidFreq(n)),
        }
    }
}

fn get_period(freq: &Freq) -> usize {
    match freq {
        Freq::Secondly(10) => 6,  // 1 minute
        Freq::Minutely(5)  => 12,
        Freq::Minutely(10) => 6,
        Freq::Minutely(15) => 4,
        Freq::Minutely(30) => 48, // 1 day
        Freq::Hourly(1)    => 24,
        Freq::Daily        => 7,
        Freq::Weekly       => 52,
        Freq::Monthly      => 12,
        Freq::Quarterly    => 4,
        Freq::Yearly       => 1,
        _                  => 1,  // unreachable
    }
}

fn get_periods(freq: &Freq) -> Vec<usize> {
    match freq {
        Freq::Secondly(10) => vec![6],
        Freq::Minutely(5)  => vec![12, 288],
        Freq::Minutely(10) => vec![6, 144],
        Freq::Minutely(15) => vec![4, 96],
        Freq::Minutely(30) => vec![48, 336],
        Freq::Hourly(1)    => vec![24, 168],
        Freq::Daily        => vec![7, 365],
        Freq::Weekly       => vec![52],
        Freq::Monthly      => vec![12],
        Freq::Quarterly    => vec![4],
        Freq::Yearly       => vec![],
        _                  => vec![],
    }
}
```