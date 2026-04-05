// Example: verify determinism — same seed produces identical results across runs.
// Also verifies that different seeds produce different results.

use flair::forecast_mean;
use std::fs;

fn main() {
    let path = "examples/test_data/japan_electricity_demand.csv";
    let content = fs::read_to_string(path).expect("failed to read CSV");
    let y: Vec<f64> = content
        .lines()
        .skip(1)
        .filter_map(|line| {
            let mut cols = line.split(',');
            cols.nth(4)?.trim().parse::<f64>().ok()
        })
        .collect();

    let run1 = forecast_mean(&y, 24, "H", 200, Some(42)).unwrap();
    let run2 = forecast_mean(&y, 24, "H", 200, Some(42)).unwrap();
    let run3 = forecast_mean(&y, 24, "H", 200, Some(99)).unwrap();

    let same_seed = run1 == run2;
    let diff_seed = run1 != run3;

    println!("same seed (42 == 42): {}", if same_seed { "identical ✓" } else { "DIFFER ✗" });
    println!("diff seed (42 != 99): {}", if diff_seed { "different ✓" } else { "SAME ✗" });

    if same_seed && diff_seed {
        println!("determinism: OK");
    } else {
        eprintln!("determinism: FAILED");
        std::process::exit(1);
    }
}
