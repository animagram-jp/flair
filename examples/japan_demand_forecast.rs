// Example: hourly electricity demand forecast for Tokyo
// Data: japanesepower.org demand.csv (2016-2024, hourly, 9 regions)
// Predicts next 24 hours from the Tokyo column.

use flair::forecast_mean;
use std::fs;

fn main() {
    let path = "examples/test_data/japan_electricity_demand.csv";
    let content = fs::read_to_string(path).expect("failed to read CSV");

    // Parse Tokyo column (index 4, 0-based after Date,Time)
    let y: Vec<f64> = content
        .lines()
        .skip(1)
        .filter_map(|line| {
            let mut cols = line.split(',');
            cols.nth(4)?.trim().parse::<f64>().ok()
        })
        .collect();

    println!("Loaded {} hourly observations (Tokyo)", y.len());

    let horizon = 24;
    let result = forecast_mean(&y, horizon, "H", 200, Some(42))
        .expect("forecast failed");

    println!("Tokyo electricity demand forecast (next {horizon}h, unit: 万kW):");
    for (h, v) in result.iter().enumerate() {
        println!("  +{:02}h: {:.0}", h + 1, v);
    }
}
