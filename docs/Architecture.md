// This file includes untranslated text (ja).

# flair

## Test

```sh
# unit tests
cargo test

# integration tests
cargo run --example integration_tests --release

# forecast accuracy (80/20 train-test split, all datasets)
cargo run --example forecast_validation --release
```

## Excutable size

Measured on release build (`cargo build --release`), WSL2 / Linux x86-64.

| target | library size (rlib) |
|--------|---------------------|
| x86-64 | 612 KB |
| wasm32 | 371 KB |

## determinism

Same seed → bit-identical output. Different seeds → different output.  
For non-deterministic output, pass `flair::seed_from_time()` (requires `std` feature, enabled by default).

```
  [OK] determinism (same seed identical; different seed differs)
```
