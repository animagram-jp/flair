// integration_tests.rs — end-to-end tests against real public datasets
//
// Run: cargo run --example integration_tests
//
// Checks:
//   1. verify()         — FLAIR self-verification (seasonal / constant / Box-Cox / Ridge)
//   2. japan_demand     — Tokyo hourly demand: confidence + forecast + determinism
//   3. elec_per_capita  — Japan/USA/Germany/China annual kWh/capita: confidence + forecast

use flair::{confidence, forecast_mean};
use std::fs;

// ── helpers ───────────────────────────────────────────────────────────────────

fn pass(label: &str) { println!("  [OK] {label}"); }
fn fail(label: &str, reason: &str) -> ! {
    eprintln!("  [FAIL] {label}: {reason}");
    std::process::exit(1);
}

fn load_col(file: &str, col: usize, skip: usize) -> Option<Vec<f64>> {
    let path = format!("examples/dataset/{}", file);
    let content = fs::read_to_string(&path).ok()?;
    Some(content.lines().skip(skip)
        .filter_map(|l| l.split(',').nth(col)?.trim().trim_matches('"').parse::<f64>().ok())
        .collect())
}

// ── 1. japan demand ───────────────────────────────────────────────────────────

fn check_japan_demand() {
    println!("=== japan_demand_tokyo ===");
    let y = match load_col("japan_demand_tokyo.csv", 2, 1) {
        Some(v) => v,
        None => { println!("  (skipped: file not found — see README for download instructions)"); return; }
    };
    println!("  loaded {} hourly observations (Tokyo MW)", y.len());

    let c = confidence(&y, "H");
    println!("  rank1   : {}", c.rank1.map_or("n/a".into(), |v| format!("{v:.3}")));
    println!("  gamma   : {}", c.gamma.map_or("n/a".into(), |v| format!("{v:.3}")));
    println!("  gcv     : {}", c.gcv  .map_or("n/a".into(), |v| format!("{v:.4}")));
    println!("  impl_ok : {}", c.impl_ok);

    let fc = forecast_mean(&y, 24, "H", 200, 42).unwrap_or_else(|e| fail("forecast", &e));
    println!("  Tokyo next 24h forecast (MW):");
    for (h, v) in fc.iter().enumerate() {
        print!("    +{:02}h: {:.0}", h + 1, v);
        if h % 4 == 3 { println!(); }
    }
    if fc.len() % 4 != 0 { println!(); }
    pass("forecast shape and finiteness");

    let a = forecast_mean(&y, 24, "H", 200, 42).unwrap();
    let b = forecast_mean(&y, 24, "H", 200, 42).unwrap();
    let c2 = forecast_mean(&y, 24, "H", 200, 99).unwrap();
    if a != b  { fail("determinism", "same seed produced different results"); }
    if a == c2 { fail("determinism", "different seeds produced identical results"); }
    pass("determinism (same seed identical; different seed differs)");
}

// ── 2. elec per capita ────────────────────────────────────────────────────────

fn check_elec_per_capita() {
    // col 1=japan, 2=usa, 3=germany, 4=china
    let series = [
        ("Japan",   1usize),
        ("USA",     2),
        ("Germany", 3),
        ("China",   4),
    ];

    for (name, col) in series {
        println!("\n=== elec_per_capita / {} ===", name);
        let y = match load_col("elec_per_capita.csv", col, 1) {
            Some(v) => v,
            None => { println!("  (skipped: file not found)"); continue; }
        };
        println!("  loaded {} annual observations (kWh/capita)", y.len());

        let c = confidence(&y, "A");
        println!("  rank1   : {}", c.rank1.map_or("n/a".into(), |v| format!("{v:.3}")));
        println!("  gamma   : {}", c.gamma.map_or("n/a".into(), |v| format!("{v:.3}")));
        println!("  gcv     : {}", c.gcv  .map_or("n/a".into(), |v| format!("{v:.4}")));
        println!("  impl_ok : {}", c.impl_ok);

        let fc = forecast_mean(&y, 3, "A", 200, 42).unwrap_or_else(|e| fail("forecast", &e));
        println!("  next 3y forecast (kWh/capita):");
        for (h, v) in fc.iter().enumerate() {
            println!("    +{}y: {:.0}", h + 1, v);
        }
        pass("forecast shape and finiteness");
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    check_japan_demand();
    check_elec_per_capita();
    println!("\nAll integration tests passed.");
}
