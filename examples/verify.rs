// Example: run FLAIR self-verification (judge function)
// verify() checks seasonal signal, constant series, Box-Cox round-trip, and Ridge SA.

use flair::verify;

fn main() {
    match verify() {
        Ok(()) => println!("verify: OK — all checks passed"),
        Err(e) => {
            eprintln!("verify: FAILED — {e}");
            std::process::exit(1);
        }
    }
}
