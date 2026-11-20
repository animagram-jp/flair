# Security Assessment

The most likely and the most impactful security problems of this software, and how each is handled.
Review this document before each release, together with the [quality checklist](https://github.com/animagram-jp/.github/blob/main/quality_checklist/README.md).

## Scope

| Item | Content |
|-|-|
| Library | The crate `flair-rs` (`src/`). `no_std`, no `unsafe`, no input or output. The only dependency is `libm`. `nalgebra` is used by tests only. |
| Example | `examples/` and the bundled datasets, used for integration tests and accuracy validation. |
| Pipeline | `.github/workflows/`. Publishing to crates.io authenticates through OIDC. |

## Actor

| Actor | Trust | Action |
|-|-|-|
| Calling application | Trusted. | Passes a series, a frequency, a horizon, a sample count, a seed and optional covariates, then reads the forecast and its confidence. |
| Data source of the application | Untrusted. | Produces the values of the series and the covariates. |
| Pipeline | Trusted. | Tests and publishes the crate. |

## Problem

| Problem | Likelihood | Impact | Handling |
|-|-|-|-|
| A long series, a large horizon or a large sample count consumes memory and time, and blocks the caller. | Medium | Medium | The history is capped to `MAX_COMPLETE` (500) periods. The horizon and the sample count are not capped, so an application bounds them before the call. |
| Non-finite or degenerate values in the series produce a forecast that looks valid. | Medium | Medium | Arguments that cannot be evaluated return `Error::InvalidInput`, and a failed decomposition returns `Error::Svd`. Non-finite values are clipped as in the reference implementation. The returned `Confidence` tells the application how far to rely on the forecast. |
| The seed or the samples are used as a secret. | Low | Medium | `seed_from_time` and the sampler are for reproducible forecasts and are not cryptographic. |
| A vulnerable or malicious dependency, or a leaked publishing credential, alters the published crate. | Low | High | The crate depends on `libm` only. No long-lived crates.io token exists, because publishing uses trusted publishing from the `crates-io` environment on a version tag. Actions are pinned by commit SHA. |
