# alphagate

Champion/challenger model promotion gating — decides whether a newly retrained model is actually good enough to replace the one currently deployed, blending statistical performance with an optional trust signal (e.g. community/user feedback).

**Status**: early — full README, usage examples, and API docs land alongside the first real implementation. This package is domain-agnostic: it doesn't know or care that it happens to be used for gating trading-model retraining in [WolfPack](https://github.com/nafisuddinn/wolfpack), where it's dogfooded as a real dependency, not just referenced.

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
