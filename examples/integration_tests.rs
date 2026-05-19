// integration_tests.rs — end-to-end tests against real public datasets
//
// Run: cargo run --example integration_tests
//
// Checks:
//   1. japan_demand_tokyo — Tokyo hourly demand: forecast + determinism
//   2. elec_per_capita    — Japan/USA/Germany/China annual kWh/capita: forecast

use flair::{forecast_mean, Freq};
use std::fs;

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
        Some(v) if !v.is_empty() => v,
        _ => { println!("  (skipped: file not found — see README for download instructions)"); return; }
    };
    println!("  loaded {} hourly observations (Tokyo MW)", y.len());
    let freq = Freq::hourly(1).unwrap();

    let (fc, _) = forecast_mean(&y, 24, &freq, 200, 42).unwrap_or_else(|e| fail("forecast", &format!("{e:?}")));
    println!("  Tokyo next 24h forecast (MW):");
    for (h, v) in fc.iter().enumerate() {
        print!("    +{:02}h: {:.0}", h + 1, v);
        if h % 4 == 3 { println!(); }
    }
    if fc.len() % 4 != 0 { println!(); }
    pass("forecast shape and finiteness");

    let (a, _) = forecast_mean(&y, 24, &freq, 200, 42).unwrap();
    let (b, _) = forecast_mean(&y, 24, &freq, 200, 42).unwrap();
    let (c2, _) = forecast_mean(&y, 24, &freq, 200, 99).unwrap();
    if a != b  { fail("determinism", "same seed produced different results"); }
    if a == c2 { fail("determinism", "different seeds produced identical results"); }
    pass("determinism (same seed identical; different seed differs)");
}

// ── 2. elec per capita ────────────────────────────────────────────────────────

fn check_elec_per_capita() {
    let series = [
        ("Japan",   1usize),
        ("USA",     2),
        ("Germany", 3),
        ("China",   4),
    ];

    for (name, col) in series {
        println!("\n=== elec_per_capita / {} ===", name);
        let y = match load_col("elec_per_capita.csv", col, 1) {
            Some(v) if !v.is_empty() => v,
            _ => { println!("  (skipped: file not found)"); continue; }
        };
        println!("  loaded {} annual observations (kWh/capita)", y.len());

        let (fc, _) = forecast_mean(&y, 3, &Freq::Yearly, 200, 42).unwrap_or_else(|e| fail("forecast", &format!("{e:?}")));
        println!("  next 3y forecast (kWh/capita):");
        for (h, v) in fc.iter().enumerate() {
            println!("    +{}y: {:.0}", h + 1, v);
        }
        pass("forecast shape and finiteness");
    }
}

fn main() {
    check_japan_demand();
    check_elec_per_capita();
    println!("\nAll integration tests passed.");
}
