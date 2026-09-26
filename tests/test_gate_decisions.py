"""Decision semantics of gate() and MarginComparator."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from alphagate import (
    IMPROVED,
    INVALID_METRIC,
    NO_INCUMBENT,
    NOT_IMPROVED,
    Decision,
    ListSink,
    MarginComparator,
    MetricResult,
    NullSink,
    gate,
    metric,
)

from _helpers import AS_OF, SpyComparator, SpyMetric, challenger, champion, holdout


def run(chal_score, champ_score, *, higher_is_better=True, comparator=None, m=None, h=None):
    m = m or SpyMetric(higher_is_better=higher_is_better)
    kwargs = {}
    if comparator is not None:
        kwargs["comparator"] = comparator
    return gate(
        challenger=challenger(chal_score),
        champion=None if champ_score is None else champion(champ_score),
        metric=m,
        holdout=h or holdout(),
        sink=NullSink(),
        as_of=AS_OF,
        **kwargs,
    )


# --- basic direction ------------------------------------------------------

def test_better_challenger_promotes():
    rec = run(0.6, 0.5)
    assert rec.decision is Decision.PROMOTE
    assert rec.promoted is True
    assert rec.reason_code == IMPROVED


def test_equal_scores_reject_incumbent_wins_ties():
    rec = run(0.5, 0.5)
    assert rec.decision is Decision.REJECT
    assert rec.promoted is False
    assert rec.reason_code == NOT_IMPROVED


def test_worse_challenger_rejects():
    rec = run(0.4, 0.5)
    assert rec.decision is Decision.REJECT
    assert rec.reason_code == NOT_IMPROVED


def test_lower_is_better_flips_direction():
    # Lower is better (e.g. an error metric): a lower challenger score wins.
    assert run(0.4, 0.5, higher_is_better=False).promoted is True
    assert run(0.6, 0.5, higher_is_better=False).promoted is False
    assert run(0.5, 0.5, higher_is_better=False).promoted is False


def test_record_carries_both_scores_and_direction():
    rec = run(0.6, 0.5, higher_is_better=False)
    assert rec.challenger_score.value == 0.6
    assert rec.champion_score is not None and rec.champion_score.value == 0.5
    assert rec.higher_is_better is False
    assert rec.metric_name == "score"


# --- min_delta boundaries --------------------------------------------------
# Values chosen to be exactly representable in binary floating point so the
# boundary test is about the comparator's semantics, not float rounding.

@pytest.mark.parametrize("higher_is_better", [True, False])
def test_improvement_exactly_at_margin_rejects(higher_is_better):
    chal = 1.5 if higher_is_better else 0.5
    rec = run(chal, 1.0, higher_is_better=higher_is_better,
              comparator=MarginComparator(min_delta=0.5))
    assert rec.decision is Decision.REJECT
    assert rec.reason_code == NOT_IMPROVED


@pytest.mark.parametrize("higher_is_better", [True, False])
def test_improvement_strictly_over_margin_promotes(higher_is_better):
    chal = 1.625 if higher_is_better else 0.375
    rec = run(chal, 1.0, higher_is_better=higher_is_better,
              comparator=MarginComparator(min_delta=0.5))
    assert rec.decision is Decision.PROMOTE
    assert rec.reason_code == IMPROVED


def test_improvement_under_margin_rejects():
    rec = run(1.25, 1.0, comparator=MarginComparator(min_delta=0.5))
    assert rec.decision is Decision.REJECT


def test_margin_comparator_params_and_stats_recorded():
    rec = run(1.625, 1.0, comparator=MarginComparator(min_delta=0.5))
    assert rec.comparator_name == "margin"
    assert rec.comparator_params == {"min_delta": 0.5}
    assert rec.comparator_stats["improvement"] == 0.625
    assert rec.comparator_stats["min_delta"] == 0.5


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf])
def test_margin_comparator_rejects_invalid_min_delta(bad):
    with pytest.raises(ValueError):
        MarginComparator(min_delta=bad)


# --- no incumbent ----------------------------------------------------------

def test_no_champion_promotes_with_no_incumbent():
    spy = SpyComparator()
    rec = run(0.1, None, comparator=spy)
    assert rec.decision is Decision.PROMOTE
    assert rec.reason_code == NO_INCUMBENT
    assert rec.champion_id is None
    assert rec.champion_score is None
    assert spy.calls == []  # nothing to compare against


# --- invalid metric values -------------------------------------------------

@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("side", ["challenger", "champion"])
def test_non_finite_score_rejects_without_calling_comparator(bad, side):
    spy = SpyComparator()
    chal, champ = (bad, 0.5) if side == "challenger" else (0.5, bad)
    rec = run(chal, champ, comparator=spy)
    assert rec.decision is Decision.REJECT
    assert rec.reason_code == INVALID_METRIC
    assert side in rec.explanation
    assert spy.calls == []


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_challenger_with_no_champion_rejects(bad):
    # A broken score must never be promoted, even into an empty slot.
    rec = run(bad, None)
    assert rec.decision is Decision.REJECT
    assert rec.reason_code == INVALID_METRIC


def test_non_finite_score_not_hidden_by_higher_is_better_false():
    # -inf would "win" a naive lower-is-better comparison; it must not.
    rec = run(-math.inf, 0.5, higher_is_better=False)
    assert rec.reason_code == INVALID_METRIC
    assert rec.promoted is False


# --- metric invocation contract --------------------------------------------

def test_metric_called_exactly_twice_with_identical_holdout_object():
    m = SpyMetric()
    h = holdout()
    chal, champ = challenger(0.6), champion(0.5)
    gate(challenger=chal, champion=champ, metric=m, holdout=h, sink=NullSink(), as_of=AS_OF)
    assert len(m.calls) == 2
    assert all(call_holdout is h for _, call_holdout in m.calls)
    called_models = [model for model, _ in m.calls]
    assert any(model is chal.model for model in called_models)
    assert any(model is champ.model for model in called_models)


def test_metric_called_once_when_no_champion():
    m = SpyMetric()
    h = holdout()
    gate(challenger=challenger(0.6), champion=None, metric=m, holdout=h, sink=NullSink(), as_of=AS_OF)
    assert len(m.calls) == 1
    assert m.calls[0][1] is h


def test_metric_exception_propagates_and_nothing_is_logged():
    def boom(model, h):
        raise RuntimeError("metric failed")

    sink = ListSink()
    with pytest.raises(RuntimeError, match="metric failed"):
        gate(challenger=challenger(0.6), champion=champion(0.5),
             metric=metric("boom", boom), holdout=holdout(), sink=sink, as_of=AS_OF)
    assert sink.records == []


# --- metric() helper ---------------------------------------------------------

def test_metric_helper_wraps_bare_float():
    m = metric("acc", lambda model, h: model.score)
    assert m.name == "acc"
    assert m.higher_is_better is True
    res = m(champion(0.7).model, holdout())
    assert isinstance(res, MetricResult)
    assert res.value == 0.7


def test_metric_helper_passes_through_metric_result():
    mr = MetricResult(value=0.3, details={"n": 10.0})
    m = metric("loss", lambda model, h: mr, higher_is_better=False)
    assert m.higher_is_better is False
    assert m(object(), holdout()) is mr


def test_metric_helper_end_to_end_lower_is_better():
    m = metric("loss", lambda model, h: model.score, higher_is_better=False)
    rec = gate(challenger=challenger(0.2), champion=champion(0.3), metric=m,
               holdout=holdout(), sink=NullSink(), as_of=AS_OF)
    assert rec.promoted
    assert rec.metric_name == "loss"


def test_custom_comparator_returning_non_verdict_is_type_error():
    class Broken:
        name = "broken"

        def params(self):
            return {}

        def compare(self, champion, challenger, *, higher_is_better):
            return True

    with pytest.raises(TypeError):
        run(0.6, 0.5, comparator=Broken())


# --- property-based ----------------------------------------------------------

finite = st.floats(allow_nan=False, allow_infinity=False, width=64)
deltas = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)


@given(a=finite, b=finite, hib=st.booleans(), d=deltas)
def test_property_at_most_one_direction_promotes(a, b, hib, d):
    comp = MarginComparator(min_delta=d)
    ab = comp.compare(MetricResult(a), MetricResult(b), higher_is_better=hib).promote
    ba = comp.compare(MetricResult(b), MetricResult(a), higher_is_better=hib).promote
    assert not (ab and ba)


@given(a=finite, b=finite, d=deltas)
def test_property_negating_scores_mirrors_direction(a, b, d):
    comp = MarginComparator(min_delta=d)
    hi = comp.compare(MetricResult(a), MetricResult(b), higher_is_better=True).promote
    lo = comp.compare(MetricResult(-a), MetricResult(-b), higher_is_better=False).promote
    assert hi == lo


@given(a=finite, b=finite, hib=st.booleans(), d=deltas)
def test_property_identical_inputs_identical_decision(a, b, hib, d):
    def once():
        return run(a, b, higher_is_better=hib, comparator=MarginComparator(min_delta=d))

    r1, r2 = once(), once()
    assert (r1.decision, r1.reason_code) == (r2.decision, r2.reason_code)
    assert r1.record_id != r2.record_id  # distinct records, same decision
