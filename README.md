# alphagate

Champion/challenger model promotion gating: decides whether a newly retrained model should replace the one currently deployed, based on a caller-supplied metric scored on out-of-sample holdout data.

What `gate()` does today:

- **Champion/challenger comparison via a pluggable comparator.** The caller supplies the metric (and its direction); a comparator makes the promote/reject call. The default, `MarginComparator`, promotes only on a strict improvement greater than `min_delta` (default `0.0`). Ties go to the incumbent. An empty champion slot promotes with reason `no_incumbent`, and non-finite scores (NaN/inf) are always rejected with `invalid_metric`.
- **Chronological / lookahead-safety checks, run before anything is scored.** Each candidate's `trained_through` must be strictly more than the configured `embargo` before `holdout.start`. The holdout must not end after `as_of`. All timestamps must be timezone-aware, and optional per-observation timestamps must be non-decreasing and inside the window. A violation raises `LookaheadError`, and nothing is scored or logged. These checks can be skipped only with an explicit `allow_untimed=True`, which is recorded in the output.
- **A logged decision record for every promotion *and* every rejection.** Each call writes exactly one immutable `GateRecord` to a sink (`JsonlSink`, `ListSink`, `NullSink`, or your own). The record holds the decision, reason code, human-readable explanation, both scores, comparator parameters, holdout window and fingerprint, and candidate metadata. If the record can't be persisted, `gate()` raises `SinkError` and returns nothing, so an unlogged decision can't be acted on.

What it does **not** do: `gate()` makes no statistical-significance or multiple-testing adjustment of its own. The default comparator only compares the two point scores. Whether that difference means anything depends entirely on the metric and `min_delta` you choose.

**Status**: early. This package is domain-agnostic: it doesn't know or care that it happens to be used for gating trading-model retraining in [WolfPack](https://github.com/nafisuddinn/wolfpack), where it's dogfooded as a real dependency, not just referenced.

## Roadmap (planned, not yet implemented)

- Comparators that account for multiple testing, e.g. one based on the Deflated Sharpe Ratio (Bailey & López de Prado, 2014). These would consume `MetricResult.samples`, a field that is reserved but not used by any shipped comparator.
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
