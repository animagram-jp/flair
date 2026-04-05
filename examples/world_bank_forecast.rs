// Example: annual electricity consumption forecast for Japan
// Data: World Bank EG.USE.ELEC.KH.PC (kWh per capita, 1990-2022)
// Predicts next 3 years.

use flair::forecast_mean;
use std::fs;

fn main() {
    let path = "examples/test_data/world_bank_electricity/API_EG.USE.ELEC.KH.PC_DS2_en_csv_v2_258.csv";
    let content = fs::read_to_string(path).expect("failed to read CSV");

    // Header row is line 5 (0-indexed: skip 4 lines of metadata)
    // Data starts line 6. Find Japan row.
    let japan_line = content
        .lines()
        .find(|l| l.contains("\"Japan\""))
        .expect("Japan row not found");

    // Columns: Country Name, Country Code, Indicator Name, Indicator Code, 1960..2025
    // First data year with values is 1990 (index 34 in 0-based after the 4 fixed cols)
    let values: Vec<f64> = japan_line
        .split(',')
        .skip(4) // skip fixed cols
        .filter_map(|v| {
            let s = v.trim().trim_matches('"');
            if s.is_empty() { None } else { s.parse::<f64>().ok() }
        })
        .collect();

    println!("Loaded {} annual observations (Japan, kWh per capita)", values.len());

    let horizon = 3;
    let result = forecast_mean(&values, horizon, "A", 200, Some(42))
        .expect("forecast failed");

    println!("Japan electricity consumption forecast (next {horizon} years, kWh/capita):");
    for (h, v) in result.iter().enumerate() {
        println!("  +{}y: {:.0}", h + 1, v);
    }
}
