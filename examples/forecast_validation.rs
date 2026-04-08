use flair::forecast_mean;
use std::fs;

fn load_col(file: &str, col: usize, skip: usize) -> Option<Vec<f64>> {
    let path = format!("examples/dataset/{}", file);
    let content = fs::read_to_string(&path).ok()?;
    Some(content.lines().skip(skip)
        .filter_map(|l| l.split(',').nth(col)?.trim().trim_matches('"').parse::<f64>().ok())
        .collect())
}

fn mae(actual: &[f64], predicted: &[f64]) -> f64 {
    actual.iter().zip(predicted).map(|(a, p)| (a - p).abs()).sum::<f64>() / actual.len() as f64
}

fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    (actual.iter().zip(predicted).map(|(a, p)| (a - p).powi(2)).sum::<f64>() / actual.len() as f64).sqrt()
}

fn mape(actual: &[f64], predicted: &[f64]) -> f64 {
    actual.iter().zip(predicted)
        .filter(|(a, _)| a.abs() > 1e-10)
        .map(|(a, p)| ((a - p).abs() / a.abs()).min(2.0))  // cap at 200% error
        .sum::<f64>()
        / actual.iter().filter(|a| a.abs() > 1e-10).count() as f64
}

// Mean Absolute Scaled Error: MAE normalized by naive 1-step forecast error on train.
// Scale-free and well-defined even when actual ≈ 0.
fn mase(train: &[f64], actual: &[f64], predicted: &[f64]) -> f64 {
    let naive_mae: f64 = train.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>()
        / (train.len() - 1) as f64;
    if naive_mae < 1e-15 { return f64::NAN; }
    let mae = actual.iter().zip(predicted).map(|(a, p)| (a - p).abs()).sum::<f64>()
        / actual.len() as f64;
    mae / naive_mae
}

fn test_dataset(name: &str, file: &str, col: usize, skip: usize, freq: &str, horizon: usize) {
    let y = match load_col(file, col, skip) {
        Some(v) => v,
        None => { println!("\n=== {} === (skipped: file not found)", name); return; }
    };
    if y.len() < horizon * 3 {
        println!("{}: too short ({} obs < {}*3)", name, y.len(), horizon);
        return;
    }

    // Split: train on 80%, test on remaining
    let train_size = (y.len() as f64 * 0.8) as usize;
    let train = &y[..train_size];
    let test_actual = &y[train_size..];
    
    println!("\n=== {} ===", name);
    println!("  total: {} obs, train: {} obs, test: {} obs", y.len(), train.len(), test_actual.len());
    
    // Make forecast from training data for next `horizon` steps
    let fc = match forecast_mean(train, horizon.min(test_actual.len()), freq, 100, 42) {
        Ok(f) => f,
        Err(e) => {
            println!("  ERROR: {}", e);
            return;
        }
    };
    
    // Compare with actual test data
    let test_len = fc.len().min(test_actual.len());
    let test_pred = &fc[..test_len];
    let test_true = &test_actual[..test_len];
    
    println!("  validation: {} steps", test_len);
    println!("    MAE:  {:.4}", mae(test_true, test_pred));
    println!("    RMSE: {:.4}", rmse(test_true, test_pred));
    println!("    MAPE: {:.2}%", mape(test_true, test_pred) * 100.0);
    println!("    MASE: {:.4}", mase(train, test_true, test_pred));
    
    // Show actual vs predicted for first few steps
    println!("  actual vs predicted:");
    for i in 0..test_len.min(3) {
        println!("    step {}: actual={:.2}, pred={:.2}, error={:.2}", 
            i+1, test_true[i], test_pred[i], (test_true[i] - test_pred[i]).abs());
    }
}

fn main() {
    println!("=== Forecast Validation (Train/Test Split) ===\n");

    // R built-in classics
    test_dataset("air_passengers",    "air_passengers.csv",    2, 1, "M", 12);
    test_dataset("nottem",            "nottem.csv",            2, 1, "M", 12);
    test_dataset("sunspot_year",      "sunspot_year.csv",      2, 1, "A", 10);

    // NOAA global surface temperature anomaly
    test_dataset("noaa_temp_annual",  "noaa_temp_annual.csv",  1, 1, "A", 10);
    test_dataset("noaa_temp_monthly", "noaa_temp_monthly.csv", 1, 1, "M", 12);

    // japanesepower.org — Tokyo hourly electricity demand (col 2 = tokyo_mw)
    test_dataset("japan_demand_tokyo", "japan_demand_tokyo.csv", 2, 1, "H", 24);

    // World Bank WDI — electric power consumption (kWh per capita)
    test_dataset("elec_japan",   "elec_per_capita.csv", 1, 1, "A", 5);
    test_dataset("elec_usa",     "elec_per_capita.csv", 2, 1, "A", 5);
    test_dataset("elec_germany", "elec_per_capita.csv", 3, 1, "A", 5);
    test_dataset("elec_china",   "elec_per_capita.csv", 4, 1, "A", 5);
}
