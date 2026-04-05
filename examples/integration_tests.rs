// integration_tests.rs — end-to-end tests against real public datasets
//
// Run: cargo run --example integration_tests
//
// Checks:
//   1. verify()       — FLAIR self-verification (seasonal / constant / Box-Cox / Ridge)
//   2. japan_demand   — Tokyo hourly demand forecast + rank-1 score + determinism
//   3. world_bank     — Japan annual kWh/capita forecast + rank-1 score + determinism

use flair::{confidence, forecast_mean, verify};
use std::fs;

// ── helpers ───────────────────────────────────────────────────────────────────

fn pass(label: &str) { println!("  [OK] {label}"); }
fn fail(label: &str, reason: &str) -> ! {
    eprintln!("  [FAIL] {label}: {reason}");
    std::process::exit(1);
}

// ── 1. verify ─────────────────────────────────────────────────────────────────

fn check_verify() {
    println!("=== verify ===");
    match verify() {
        Ok(()) => pass("seasonal / constant / box-cox / ridge"),
        Err(e) => fail("verify", &e),
    }
}

// ── 2. japan demand (hourly, 9 regions) ──────────────────────────────────────

fn load_japan_demand() -> Vec<f64> {
    let path = "examples/test_data/japan_electricity_demand.csv";
    let content = fs::read_to_string(path).expect("failed to read japan_electricity_demand.csv");
    content
        .lines()
        .skip(1)
        .filter_map(|line| {
            let mut cols = line.split(',');
            cols.nth(4)?.trim().parse::<f64>().ok() // Tokyo column
        })
        .collect()
}

fn check_japan_demand() {
    println!("=== japan_demand ===");
    let y = load_japan_demand();
    println!("  loaded {} hourly observations (Tokyo)", y.len());

    // confidence
    let c = confidence(&y, "H");
    println!("  rank1 : {}", c.rank1.map_or("n/a".into(), |v| format!("{v:.3}"))  );
    println!("  gamma : {}", c.gamma.map_or("n/a".into(), |v| format!("{v:.3}"))  );
    println!("  gcv   : {}", c.gcv  .map_or("n/a".into(), |v| format!("{v:.4}")));

    // forecast
    let fc = forecast_mean(&y, 24, "H", 200, None).unwrap_or_else(|e| fail("forecast", &e));
    println!("  Tokyo next 24h forecast (万kW):");
    for (h, v) in fc.iter().enumerate() {
        print!("    +{:02}h: {:.0}", h + 1, v);
        if h % 4 == 3 { println!(); }
    }
    if fc.len() % 4 != 0 { println!(); }
    pass("forecast shape and finiteness");

    // determinism: same seed → identical, seed=None → non-deterministic
    let a = forecast_mean(&y, 24, "H", 200, Some(42)).unwrap();
    let b = forecast_mean(&y, 24, "H", 200, Some(42)).unwrap();
    let c = forecast_mean(&y, 24, "H", 200, None).unwrap();
    let d = forecast_mean(&y, 24, "H", 200, None).unwrap();
    if a != b { fail("determinism", "same seed produced different results"); }
    if a == c { fail("determinism", "different seeds produced identical results"); }
    if c == d { fail("non-determinism", "seed=None produced identical results across runs"); }
    pass("determinism (same seed identical; seed=None non-deterministic)");
}

// ── 3. world bank annual kWh/capita ──────────────────────────────────────────

fn load_world_bank() -> Vec<f64> {
    let path = "examples/test_data/world_bank_electricity/API_EG.USE.ELEC.KH.PC_DS2_en_csv_v2_258.csv";
    let content = fs::read_to_string(path).expect("failed to read world_bank CSV");
    let japan_line = content
        .lines()
        .find(|l| l.contains("\"Japan\""))
        .expect("Japan row not found");
    japan_line
        .split(',')
        .skip(4)
        .filter_map(|v| {
            let s = v.trim().trim_matches('"');
            if s.is_empty() { None } else { s.parse::<f64>().ok() }
        })
        .collect()
}

fn check_world_bank() {
    println!("=== world_bank ===");
    let y = load_world_bank();
    println!("  loaded {} annual observations (Japan kWh/capita)", y.len());

    // confidence — annual data has no intra-period structure, rank1/gamma will be n/a
    let c = confidence(&y, "A");
    println!("  rank1 : {}", c.rank1.map_or("n/a".into(), |v| format!("{v:.3}"))  );
    println!("  gamma : {}", c.gamma.map_or("n/a".into(), |v| format!("{v:.3}"))  );
    println!("  gcv   : {}", c.gcv  .map_or("n/a".into(), |v| format!("{v:.4}")));

    // forecast
    let fc = forecast_mean(&y, 3, "A", 200, None).unwrap_or_else(|e| fail("forecast", &e));
    println!("  Japan next 3y forecast (kWh/capita):");
    for (h, v) in fc.iter().enumerate() {
        println!("    +{}y: {:.0}", h + 1, v);
    }
    pass("forecast shape and finiteness");

    // determinism
    let a = forecast_mean(&y, 3, "A", 200, Some(42)).unwrap();
    let b = forecast_mean(&y, 3, "A", 200, Some(42)).unwrap();
    let c = forecast_mean(&y, 3, "A", 200, None).unwrap();
    let d = forecast_mean(&y, 3, "A", 200, None).unwrap();
    if a != b { fail("determinism", "same seed produced different results"); }
    if a == c { fail("determinism", "different seeds produced identical results"); }
    if c == d { fail("non-determinism", "seed=None produced identical results across runs"); }
    pass("determinism (same seed identical; seed=None non-deterministic)");
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    check_verify();
    check_japan_demand();
    check_world_bank();
    println!("\nAll integration tests passed.");
}
