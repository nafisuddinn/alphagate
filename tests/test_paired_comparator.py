"""PairedComparator, paired_test and spend_alpha.

The Diebold-Mariano statistic is checked three ways:
1. A hand-computed known answer (worked out in the comments below).
2. An independent implementation written here as the full double sum
   (1/n) * sum_t sum_s w(|t-s|) e_t e_s, which is algebraically equal to the
   Bartlett-weighted autocovariance form used by the package but shares no
   code with it.
3. A Monte Carlo calibration check: under a true null, a one-sided test at
   alpha=0.05 rejects about 5% of the time.
"""

from __future__ import annotations

import math
import random

import pytest

from alphagate import (
    INVALID_METRIC,
    Decision,
    ListSink,
    MetricResult,
    PairedComparator,
    gate,
    paired_test,
    spend_alpha,
)
from alphagate.comparators import (
    INFERIOR,
    INSUFFICIENT_SAMPLES,
    NON_INFERIOR,
    NOT_SIGNIFICANT,
    SAMPLES_MISMATCH,
    SIGNIFICANT_IMPROVEMENT,
)

from _helpers import AS_OF, challenger, champion, holdout


# --- reference implementation (independent of the package) -----------------


def _reference_dm(a, b, lags, null=0.0):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    e = [x - mean for x in d]
    lrv = 0.0
    for t in range(n):
        for s in range(n):
            j = abs(t - s)
            if j <= lags:
                lrv += (1.0 - j / (lags + 1.0)) * e[t] * e[s]
    lrv /= n
    se = math.sqrt(lrv / n)
    z = (mean - null) / se
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return mean, se, z, p


# --- paired_test: known answers ----------------------------------------------


def test_paired_test_hand_computed_known_answer_one_lag():
    # d = a - b = [1, 2, 3, 6]; mean 3; deviations -2, -1, 0, 3.
    # gamma_0 = (4 + 1 + 0 + 9) / 4 = 3.5
    # gamma_1 = ((-1)(-2) + (0)(-1) + (3)(0)) / 4 = 0.5
    # Bartlett weight for lag 1 with L=1: 1 - 1/2 = 0.5
    # long-run variance = 3.5 + 2 * 0.5 * 0.5 = 4.0
    # se = sqrt(4.0 / 4) = 1.0; t = 3.0; p = P(Z > 3) = 0.0013498980316301
    res = paired_test([1, 2, 3, 6], [0, 0, 0, 0], lags=1)
    assert res.mean_diff == pytest.approx(3.0, abs=1e-15)
    assert res.se == pytest.approx(1.0, abs=1e-15)
    assert res.t == pytest.approx(3.0, abs=1e-15)
    assert res.p == pytest.approx(0.0013498980316301, rel=1e-12)
    assert res.n == 4
    assert res.lags == 1


def test_paired_test_hand_computed_known_answer_zero_lags():
    # Same d, no HAC lags: variance 3.5, se = sqrt(3.5/4), t = 3 / se.
    res = paired_test([1, 2, 3, 6], [0, 0, 0, 0], lags=0)
    se = math.sqrt(3.5 / 4)
    assert res.se == pytest.approx(se, rel=1e-14)
    assert res.t == pytest.approx(3.0 / se, rel=1e-14)
    assert res.p == pytest.approx(0.5 * math.erfc((3.0 / se) / math.sqrt(2)), rel=1e-12)


def test_paired_test_null_shifts_the_statistic():
    res = paired_test([1, 2, 3, 6], [0, 0, 0, 0], lags=1, null=1.0)
    assert res.t == pytest.approx(2.0, abs=1e-15)  # (3 - 1) / 1
    assert res.mean_diff == pytest.approx(3.0)  # mean_diff is never shifted


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("lags", [0, 1, 3, 7])
def test_paired_test_matches_independent_double_sum(seed, lags):
    rng = random.Random(seed)
    n = 60
    # Autocorrelated differences, so the HAC lags actually matter.
    a, b, prev = [], [], 0.0
    for _ in range(n):
        prev = 0.5 * prev + rng.gauss(0.0, 1.0)
        b.append(rng.gauss(0.0, 1.0))
        a.append(b[-1] + 0.2 + prev)
    mean, se, z, p = _reference_dm(a, b, lags, null=0.1)
    res = paired_test(a, b, lags=lags, null=0.1)
    assert res.mean_diff == pytest.approx(mean, rel=1e-12)
    assert res.se == pytest.approx(se, rel=1e-10)
    assert res.t == pytest.approx(z, rel=1e-10)
    assert res.p == pytest.approx(p, rel=1e-9, abs=1e-15)


def test_paired_test_default_lags_is_newey_west_rule_of_thumb():
    # floor(4 * (n / 100) ** (2 / 9)): n=252 -> 4, n=100 -> 4, n=30 -> 3.
    for n, expected in ((252, 4), (100, 4), (30, 3)):
        res = paired_test([float(i % 3) for i in range(n)], [0.0] * n)
        assert res.lags == expected


def test_paired_test_lags_are_capped_below_n():
    res = paired_test([1.0, 2.0, 4.0], [0.0, 0.0, 0.0], lags=10)
    assert res.lags == 2


def test_paired_test_zero_variance():
    up = paired_test([1.0] * 5, [0.0] * 5, lags=1)
    assert up.se == 0.0 and up.p == 0.0 and up.t == math.inf
    flat = paired_test([1.0] * 5, [1.0] * 5, lags=1)
    assert flat.se == 0.0 and flat.p == 1.0
    down = paired_test([0.0] * 5, [1.0] * 5, lags=1)
    assert down.p == 1.0 and down.t == -math.inf


@pytest.mark.parametrize(
    "a,b,lags,exc",
    [
        ([1.0], [1.0], 0, ValueError),  # n < 2
        ([1.0, 2.0], [1.0], 0, ValueError),  # length mismatch
        ([1.0, float("nan")], [1.0, 2.0], 0, ValueError),
        ([1.0, 2.0], [1.0, float("inf")], 0, ValueError),
        ([1.0, 2.0], [1.0, 2.0], -1, ValueError),
        ([1.0, 2.0], [1.0, 2.0], 1.5, TypeError),
        ([1.0, 2.0], [1.0, 2.0], True, TypeError),
    ],
)
def test_paired_test_rejects_bad_input(a, b, lags, exc):
    with pytest.raises(exc):
        paired_test(a, b, lags=lags)


def test_paired_test_is_calibrated_under_the_null():
    """Under H0 (iid differences, mean 0) a one-sided 5% test should reject
    about 5% of the time. 4000 replications: the binomial sd of the
    rejection rate is ~0.0034, so [0.035, 0.065] is a > 4-sigma band."""
    rng = random.Random(12345)
    reps, n, rejections = 4000, 252, 0
    for _ in range(reps):
        a = [rng.gauss(0.0, 1.0) for _ in range(n)]
        b = [rng.gauss(0.0, 1.0) for _ in range(n)]
        if paired_test(a, b, lags=5).p < 0.05:
            rejections += 1
    assert 0.035 <= rejections / reps <= 0.065


# --- spend_alpha ----------------------------------------------------------------


def test_spend_alpha_values():
    assert spend_alpha(0.05, 1) == pytest.approx(0.025)
    assert spend_alpha(0.05, 2) == pytest.approx(0.05 / 6)
    assert spend_alpha(0.05, 3) == pytest.approx(0.05 / 12)


def test_spend_alpha_total_never_exceeds_budget():
    # sum_{k>=1} total / (k (k+1)) telescopes to total * (1 - 1/(K+1)) < total.
    for K in (1, 2, 10, 1000):
        s = sum(spend_alpha(0.05, k) for k in range(1, K + 1))
        assert s == pytest.approx(0.05 * (1 - 1 / (K + 1)), rel=1e-12)
        assert s < 0.05


@pytest.mark.parametrize("total,k,exc", [
    (0.05, 0, ValueError), (0.05, -1, ValueError), (0.0, 1, ValueError),
    (1.0, 1, ValueError), (float("nan"), 1, ValueError), (0.05, 1.0, TypeError),
    (0.05, True, TypeError),
])
def test_spend_alpha_rejects_bad_input(total, k, exc):
    with pytest.raises(exc):
        spend_alpha(total, k)


# --- PairedComparator -------------------------------------------------------------


def _mr(samples):
    samples = list(samples)
    return MetricResult(value=sum(samples) / len(samples), samples=samples)


def _noisy(n, mean, sd=0.01, seed=0):
    rng = random.Random(seed)
    return [mean + rng.gauss(0.0, sd) for _ in range(n)]


def test_constructor_validation():
    with pytest.raises(ValueError):
        PairedComparator(alpha=0.0)
    with pytest.raises(ValueError):
        PairedComparator(alpha=1.0)
    with pytest.raises(ValueError):
        PairedComparator(mode="better")
    with pytest.raises(ValueError):
        PairedComparator(margin=-0.1)
    with pytest.raises(ValueError):
        PairedComparator(margin=float("nan"))
    with pytest.raises(ValueError):
        PairedComparator(hac_lags=-1)
    with pytest.raises(ValueError):
        PairedComparator(min_samples=1)


def test_params_are_recorded_verbatim():
    c = PairedComparator(alpha=0.01, mode="non_inferiority", margin=0.002, hac_lags=5, min_samples=40)
    assert c.name == "paired_dm"
    assert dict(c.params()) == {
        "alpha": 0.01, "mode": "non_inferiority", "margin": 0.002,
        "hac_lags": 5, "min_samples": 40, "test": "diebold_mariano_newey_west_one_sided",
    }


def test_superiority_lower_is_better_clear_improvement_promotes():
    champ = _mr(_noisy(100, 0.70, seed=1))
    chal = _mr([x - 0.01 for x in champ.samples])  # every session 0.01 lower loss
    chal = _mr([x + e for x, e in zip(chal.samples, _noisy(100, 0.0, sd=0.002, seed=2))])
    v = PairedComparator(alpha=0.05, hac_lags=2).compare(champ, chal, higher_is_better=False)
    assert v.promote and v.reason_code == SIGNIFICANT_IMPROVEMENT
    assert v.stats["mean_diff"] > 0  # oriented as improvement
    assert v.stats["p"] < 0.05
    assert set(v.stats) >= {"mean_diff", "se", "t", "p", "n", "alpha", "margin", "lags"}
    assert v.stats["n"] == 100


def test_superiority_noise_only_rejects_not_significant():
    champ = _mr(_noisy(252, 0.69, seed=3))
    chal = _mr(_noisy(252, 0.69, seed=4))
    v = PairedComparator(alpha=0.05).compare(champ, chal, higher_is_better=False)
    # Pre-checked: this seed pair does not reach significance.
    assert not v.promote and v.reason_code == NOT_SIGNIFICANT


def test_superiority_direction_higher_is_better():
    base = _noisy(80, 0.5, seed=5)
    better = [x + 0.01 for x in base]
    worse = [x - 0.01 for x in base]
    c = PairedComparator(alpha=0.05, hac_lags=0)
    assert c.compare(_mr(base), _mr(better), higher_is_better=True).promote
    assert not c.compare(_mr(base), _mr(worse), higher_is_better=True).promote
    # Same numbers with lower-is-better flip the outcome.
    assert c.compare(_mr(base), _mr(worse), higher_is_better=False).promote


def test_superiority_margin_is_required_improvement():
    champ = _noisy(100, 0.70, seed=6)
    chal = [x - 0.01 for x in champ]  # exactly 0.01 better each session; zero variance
    assert PairedComparator(margin=0.0).compare(_mr(champ), _mr(chal), higher_is_better=False).promote
    v = PairedComparator(margin=0.02).compare(_mr(champ), _mr(chal), higher_is_better=False)
    assert not v.promote and v.reason_code == NOT_SIGNIFICANT


def test_non_inferiority_slightly_worse_within_margin_promotes():
    champ = _noisy(252, 0.69, seed=7)
    rng = random.Random(8)
    chal = [x + 0.0005 + rng.gauss(0, 0.001) for x in champ]  # 0.0005 worse, tight
    v = PairedComparator(mode="non_inferiority", margin=0.002, alpha=0.05).compare(
        _mr(champ), _mr(chal), higher_is_better=False
    )
    assert v.promote and v.reason_code == NON_INFERIOR
    assert v.stats["mean_diff"] < 0  # it IS worse, just not by more than the margin


def test_non_inferiority_clearly_worse_rejects_inferior():
    champ = _noisy(252, 0.69, seed=9)
    rng = random.Random(10)
    chal = [x + 0.01 + rng.gauss(0, 0.001) for x in champ]
    v = PairedComparator(mode="non_inferiority", margin=0.002).compare(
        _mr(champ), _mr(chal), higher_is_better=False
    )
    assert not v.promote and v.reason_code == INFERIOR


def test_non_inferiority_with_zero_margin_needs_evidence_of_no_worse():
    same = _noisy(100, 0.69, seed=11)
    v = PairedComparator(mode="non_inferiority", margin=0.0).compare(
        _mr(same), _mr(list(same)), higher_is_better=False
    )
    # Identical samples: zero variance, mean diff 0, null 0 -> p = 1, not shown.
    assert not v.promote and v.reason_code == INFERIOR


def test_insufficient_samples():
    c = PairedComparator(min_samples=30)
    v = c.compare(_mr(_noisy(29, 0.7)), _mr(_noisy(29, 0.6)), higher_is_better=False)
    assert not v.promote and v.reason_code == INSUFFICIENT_SAMPLES
    v = c.compare(MetricResult(value=0.7), MetricResult(value=0.6), higher_is_better=False)
    assert not v.promote and v.reason_code == INSUFFICIENT_SAMPLES


def test_samples_mismatch():
    v = PairedComparator().compare(
        _mr(_noisy(40, 0.7)), _mr(_noisy(41, 0.6)), higher_is_better=False
    )
    assert not v.promote and v.reason_code == SAMPLES_MISMATCH


def test_non_finite_samples_are_invalid():
    s = _noisy(40, 0.7)
    bad = list(s)
    bad[3] = float("nan")
    v = PairedComparator().compare(_mr(s), MetricResult(value=0.6, samples=bad), higher_is_better=False)
    assert not v.promote and v.reason_code == INVALID_METRIC


def test_samples_must_be_numbers():
    s = _noisy(40, 0.7)
    v = PairedComparator().compare(
        _mr(s), MetricResult(value=0.6, samples=["x"] * 40), higher_is_better=False
    )
    assert not v.promote and v.reason_code == INVALID_METRIC


# --- through gate(): records carry the stats, rejections are logged --------------


class SamplesMetric:
    name = "per_period_loss"
    higher_is_better = False

    def __call__(self, model, h):
        return _mr(model.samples)


class _M:
    def __init__(self, samples):
        self.samples = samples


def _cand(factory, samples, **kw):
    c = factory(0.0, **kw)
    return type(c)(model=_M(samples), id=c.id, trained_through=c.trained_through)


def test_gate_records_paired_stats_for_promotion_and_rejection():
    champ_s = _noisy(60, 0.70, seed=12)
    good = [x - 0.02 for x in champ_s]
    bad = [x + 0.02 for x in champ_s]
    sink = ListSink()
    comp = PairedComparator(alpha=0.01, hac_lags=3)
    for chal_s in (good, bad):
        gate(
            challenger=_cand(challenger, chal_s), champion=_cand(champion, champ_s),
            metric=SamplesMetric(), holdout=holdout(), sink=sink, comparator=comp,
            as_of=AS_OF, context={"trial_number": 3},
        )
    promoted, rejected = sink.records
    assert promoted.decision is Decision.PROMOTE
    assert promoted.reason_code == SIGNIFICANT_IMPROVEMENT
    assert rejected.decision is Decision.REJECT
    assert rejected.reason_code == NOT_SIGNIFICANT
    for rec in sink.records:
        assert rec.comparator_name == "paired_dm"
        assert set(rec.comparator_stats) >= {"mean_diff", "se", "t", "p", "n", "alpha", "margin"}
        assert rec.comparator_params["alpha"] == 0.01
        assert rec.context["trial_number"] == 3
        rec.to_json()  # strict JSON, including +/-inf stats
    assert rejected.explanation  # the reason it lost is human-readable


def test_gate_zero_variance_stats_serialise():
    champ_s = [0.7] * 40
    chal_s = [0.6] * 40
    sink = ListSink()
    rec = gate(
        challenger=_cand(challenger, chal_s), champion=_cand(champion, champ_s),
        metric=SamplesMetric(), holdout=holdout(), sink=sink,
        comparator=PairedComparator(), as_of=AS_OF,
    )
    assert rec.promoted
    assert '"t": "Infinity"' in rec.to_json()
