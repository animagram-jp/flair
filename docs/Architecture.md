# flair architecture

## Rule

- [common for repositories](https://github.com/animagram-jp/.github/blob/main/Rule.md)

## Test

```sh
# unit tests
cargo test

# integration tests (crash-free + determinism on bundled datasets)
cargo run --example integration_tests --release

# forecast accuracy (80/20 train-test split, bundled datasets)
cargo run --example forecast_validation --release
```

## Python vs Rust comparison

To compare forecast output between flaircast (Python) and this crate (Rust),
run the same series with the same arguments in both and compare the mean forecast.

Python (requires `uv` and `flaircast` from PyPI):

```sh
uv run --with flaircast python3 - << 'EOF'
import math
from flaircast import forecast
import statistics

y = [100 + 1.5*i + 20*math.sin(2*math.pi*i/12) for i in range(144)]
samples = forecast(y, horizon=12, freq="M", n_samples=500, seed=0)
mean_fc = [statistics.mean(samples[:, h].tolist()) for h in range(12)]
print("Python mean forecast:", [round(v, 1) for v in mean_fc])
EOF
```

Rust (this crate, via `cargo test`):

The unit test `flair::tests::lwcp_vs_python_reference` in `src/flair.rs` runs
the same series (y = 100 + 1.5*t + 20*sin(2*pi*t/12), monthly, 144 points,
seed=0, n_samples=500) and asserts the mean forecast is within +-15 of the
flaircast 0.6.1 reference values recorded in the test.

## Executable size

Measured on release build (`cargo build --release`), WSL2 / Linux x86-64.

| target | library size (rlib) |
|--------|---------------------|
| x86-64 | 783 KB |
| wasm32 | 513 KB |