# alphagate

Champion/challenger model promotion gating: decides whether a newly retrained model should replace the one currently deployed, based on a caller-supplied metric scored on out-of-sample holdout data.

What `gate()` does today:

- **Champion/challenger comparison via a pluggable comparator.** The caller supplies the metric (and its direction); a comparator makes the promote/reject call. The default, `MarginComparator`, promotes only on a strict improvement greater than `min_delta` (default `0.0`). Ties go to the incumbent. An empty champion slot promotes with reason `no_incumbent`, and non-finite scores (NaN/inf) are always rejected with `invalid_metric`.
- **Chronological / lookahead-safety checks, run before anything is scored.** Each candidate's `trained_through` must be strictly more than the configured `embargo` before `holdout.start`. The holdout must not end after `as_of`. All timestamps must be timezone-aware, and optional per-observation timestamps must be non-decreasing and inside the window. A violation raises `LookaheadError`, and nothing is scored or logged. These checks can be skipped only with an explicit `allow_untimed=True`, which is recorded in the output.
- **A logged decision record for every promotion *and* every rejection.** Each call writes exactly one immutable `GateRecord` to a sink (`JsonlSink`, `ListSink`, `NullSink`, or your own). The record holds the decision, reason code, human-readable explanation, both scores, comparator parameters, holdout window and fingerprint, and candidate metadata. If the record can't be persisted, `gate()` raises `SinkError` and returns nothing, so an unlogged decision can't be acted on.

What it does **not** do: `gate()` itself makes no statistical-significance or multiple-testing adjustment. The default comparator only compares the two point scores. Whether that difference means anything depends entirely on the metric and `min_delta` you choose. For a significance test, use `PairedComparator` (below).

## `PairedComparator` (v0.2.0): a paired significance test

`PairedComparator` decides on a one-sided [Diebold-Mariano](https://doi.org/10.1080/07350015.1995.10524599) test (Diebold & Mariano, 1995) instead of the two point scores. It reads `MetricResult.samples`: one value per observation or per period, for both models, **in the same order**. It tests the per-sample improvement of the challenger over the champion (`champion - challenger` when lower is better, `challenger - champion` when higher is better), with a Newey-West (1987) HAC standard error so autocorrelated samples (e.g. consecutive time periods) don't overstate the evidence. Pure Python, no new dependencies.

```python
from alphagate import PairedComparator, gate, spend_alpha

comparator = PairedComparator(
    alpha=0.05,                # one-sided level; promote only if p < alpha
    mode="superiority",        # or "non_inferiority"
    margin=0.0,                # >= 0; see modes below
    hac_lags=5,                # None = floor(4 * (n/100) ** (2/9))
    min_samples=30,
)
```

- `mode="superiority"`: promote iff the test rejects "mean improvement <= `margin`". Reasons: `significant_improvement` / `not_significant`.
- `mode="non_inferiority"`: promote iff the test rejects "the challenger is worse by `margin` or more". Reasons: `non_inferior` / `inferior` (meaning non-inferiority was *not demonstrated*, not that inferiority was proven).
- Fewer than `min_samples` pairs, or no samples: `insufficient_samples`. Different lengths: `samples_mismatch`. A non-finite or non-numeric sample: `invalid_metric`.
- The decision record's `comparator_stats` holds `mean_diff`, `se`, `t`, `p`, `n`, `lags`, `alpha`, `margin`.

Limits, stated plainly:

- **Pairing is the caller's job.** alphagate checks that both sample lists have the same length; it cannot check that position *i* refers to the same observation for both models.
- **The p-value uses the normal approximation** to the DM statistic, which is somewhat optimistic for small `n`. Keep `n` well above `min_samples`.
- **The test is on the mean of the samples**, which need not equal `MetricResult.value` if your metric aggregates differently.

### Multiple testing: `spend_alpha` and the trial number

Running the same test on many challengers inflates the chance that one passes by luck. `spend_alpha(total, k)` returns `total / (k (k + 1))` for the `k`-th test in an open-ended series. Those levels sum to less than `total` however many tests are eventually run, so the chance of any false promotion across the whole series stays below `total`.

**alphagate cannot verify `k`.** The caller supplies the trial number (put it in `context` so it is logged with the decision) and is responsible for counting every challenger that was ever evaluated, including ones that were abandoned or never logged. Undercounting `k` silently voids the guarantee. `paired_test(a, b, lags)` is also exported for reporting the same statistic outside a gate decision.

**Status**: early. This package is domain-agnostic: it doesn't know or care that it happens to be used for gating trading-model retraining in [WolfPack](https://github.com/nafisuddinn/wolfpack), where it's dogfooded as a real dependency, not just referenced.

## Roadmap (planned, not yet implemented)

- An optional trust signal (e.g. community/user feedback) blended into the promotion decision.

None of the above exists in `src/` yet. Don't rely on it.

## Why this exists

Retraining a model on new data doesn't guarantee the new version is better. `alphagate` enforces a simple rule before any promotion: prove it on a holdout first. That's the same discipline real ML teams use before trusting a new model with production traffic — this is a small, honest, general-purpose version of it.

## Install (once published)

```bash
pip install alphagate
```

Until then, install directly from this repo:

```bash
pip install git+https://github.com/nafisuddinn/alphagate.git
```

## License

MIT
