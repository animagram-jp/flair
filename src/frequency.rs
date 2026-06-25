// SPDX-License-Identifier: Apache-2.0
//
// このファイルは FLAIR (flaircast) (Copyright 2026 Takato Honda, Apache-2.0)
// の flaircast/_frequency.py を Rust へ移植したものです。詳細は同梱の NOTICE を参照。

/// 周波数文字列の解決とカレンダーテーブル。
///
/// 出典: flaircast/_frequency.py (Apache-2.0)
/// https://github.com/Mellon-Inc/FLAIR/blob/main/flaircast/_frequency.py

/// pandas形式の周波数文字列を正規化する。
///
/// 変換ルール:
/// 1. "min" サフィックス → "T" (例: "30min" → "30T")
/// 2. pandas 2.2+ サフィックス (ME, QE, YE, MS...) → 単一文字
/// 3. オフセットアンカー (W-SUN, Q-DEC) → ベース文字
pub fn resolve_freq(freq: &str) -> String {
    let mut f = freq.to_uppercase().replace("MIN", "T");

    // オフセットアンカーを除去: "QE-DEC" → "QE"
    if let Some(dash) = f.find('-') {
        f = f[..dash].to_string();
    }

    // pandas 2.2+ サフィックスを変換
    let suffixes = [
        ("ME", "M"),
        ("QE", "Q"),
        ("YE", "Y"),
        ("MS", "M"),
        ("QS", "Q"),
        ("YS", "Y"),
        ("AS", "A"),
    ];
    for (suffix, base) in &suffixes {
        if f == *suffix {
            return base.to_string();
        }
        if f.len() > suffix.len() && f.ends_with(suffix) {
            f = format!("{}{}", &f[..f.len() - suffix.len()], base);
            break;
        }
    }
    f
}

/// 周波数文字列から主要周期を返す。未知の場合は 1。
pub fn get_period(freq: &str) -> usize {
    let f = resolve_freq(freq);
    match f.as_str() {
        "S" => 60,
        "T" => 60,
        "5T" => 12,
        "10T" => 6,
        "15T" => 4,
        "30T" => 48,
        "10S" => 6,
        "H" => 24,
        "D" => 7,
        "W" => 52,
        "M" => 12,
        "Q" => 4,
        "A" | "Y" => 1,
        _ => {
            // 末尾マッチ (例: "2H" → "H")
            let table = [
                ("S", 60usize),
                ("T", 60),
                ("H", 24),
                ("D", 7),
                ("W", 52),
                ("M", 12),
                ("Q", 4),
            ];
            for (k, v) in &table {
                if f.ends_with(k) {
                    return *v;
                }
            }
            1
        }
    }
}

/// 周波数文字列から MDL 候補周期リストを返す。
pub fn get_periods(freq: &str) -> Vec<usize> {
    let f = resolve_freq(freq);
    match f.as_str() {
        "10S" => vec![6, 360],
        "S" => vec![60],
        "5T" => vec![12, 288],
        "10T" => vec![6, 144],
        "15T" => vec![4, 96],
        "30T" => vec![48, 336],
        "H" => vec![24, 168],
        "D" => vec![7, 365],
        "W" => vec![52],
        "M" => vec![12],
        "Q" => vec![4],
        "A" | "Y" => vec![],
        _ => {
            let table: &[(&str, &[usize])] = &[
                ("H", &[24, 168]),
                ("D", &[7, 365]),
                ("W", &[52]),
                ("M", &[12]),
                ("Q", &[4]),
            ];
            for (k, v) in table {
                if f.ends_with(k) {
                    return v.to_vec();
                }
            }
            vec![]
        }
    }
}
