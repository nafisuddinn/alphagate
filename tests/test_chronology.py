"""Chronology (lookahead) checks. Every violation must raise LookaheadError
BEFORE any scoring or logging happens."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from alphagate import AlphagateError, ListSink, LookaheadError, gate

from _helpers import (
    AS_OF,
    CHALLENGER_TT,
    HOLDOUT_END,
    HOLDOUT_START,
    SpyMetric,
    challenger,
    champion,
    daily,
    holdout,
)

NAIVE = datetime(2024, 1, 15)  # no tzinfo


def call(*, chal=None, champ="default", h=None, embargo=timedelta(0), as_of=AS_OF,
         allow_untimed=False, sink=None, m=None):
    return gate(
        challenger=chal or challenger(0.6),
        champion=champion(0.5) if champ == "default" else champ,
        metric=m or SpyMetric(),
        holdout=h or holdout(),
        sink=sink if sink is not None else ListSink(),
        embargo=embargo,
        as_of=as_of,
        allow_untimed=allow_untimed,
    )


def test_clean_timeline_passes_and_is_marked_checked():
    rec = call()
    assert rec.chronology_checked is True


def test_lookahead_error_is_an_alphagate_error():
    assert issubclass(LookaheadError, AlphagateError)


# --- training/holdout overlap ------------------------------------------------

def test_challenger_trained_through_equal_to_holdout_start_raises():
    with pytest.raises(LookaheadError):
        call(chal=challenger(0.6, trained_through=HOLDOUT_START))


def test_challenger_trained_into_holdout_raises():
    with pytest.raises(LookaheadError):
        call(chal=challenger(0.6, trained_through=HOLDOUT_START + timedelta(days=3)))


def test_champion_trained_through_equal_to_holdout_start_raises():
    with pytest.raises(LookaheadError, match="champion"):
        call(champ=champion(0.5, trained_through=HOLDOUT_START))


def test_embargo_gap_smaller_than_required_raises():
    # Challenger ends 1 day before holdout; a 5-day embargo is required.
    with pytest.raises(LookaheadError, match="embargo"):
        call(embargo=timedelta(days=5))


def test_embargo_gap_exactly_equal_raises():
    # Rule is trained_through + embargo < holdout.start (strict).
    gap = HOLDOUT_START - CHALLENGER_TT
    with pytest.raises(LookaheadError):
        call(embargo=gap)


def test_embargo_gap_satisfied_passes():
    gap = HOLDOUT_START - CHALLENGER_TT
    rec = call(embargo=gap - timedelta(seconds=1))
    assert rec.embargo_seconds == (gap - timedelta(seconds=1)).total_seconds()


def test_negative_embargo_is_value_error():
    with pytest.raises(ValueError):
        call(embargo=timedelta(days=-1))


# --- tz-awareness ---------------------------------------------------------

@pytest.mark.parametrize("where", ["challenger", "champion", "start", "end", "as_of", "timestamps"])
def test_naive_datetimes_raise(where):
    kw = {}
    if where == "challenger":
        kw["chal"] = challenger(0.6, trained_through=NAIVE)
    elif where == "champion":
        kw["champ"] = champion(0.5, trained_through=NAIVE)
    elif where == "start":
        kw["h"] = holdout(start=HOLDOUT_START.replace(tzinfo=None))
    elif where == "end":
        kw["h"] = holdout(end=HOLDOUT_END.replace(tzinfo=None))
    elif where == "as_of":
        kw["as_of"] = AS_OF.replace(tzinfo=None)
    elif where == "timestamps":
        ts = daily(HOLDOUT_START, 3)
        ts[1] = ts[1].replace(tzinfo=None)
        kw["h"] = holdout(timestamps=ts)
    with pytest.raises(LookaheadError, match="tz"):
        call(**kw)


def test_non_utc_aware_datetimes_are_compared_correctly():
    # 2024-02-01 00:30 at UTC+05:00 is 2024-01-31 19:30 UTC: before the
    # holdout start, so this is clean even though the wall-clock date matches.
    plus5 = timezone(timedelta(hours=5))
    tt = datetime(2024, 2, 1, 0, 30, tzinfo=plus5)
    rec = call(chal=challenger(0.6, trained_through=tt))
    assert rec.chronology_checked


# --- missing timing information -------------------------------------------

@pytest.mark.parametrize("where", ["challenger", "champion", "start", "end"])
def test_missing_timing_raises_unless_allow_untimed(where):
    kw = {}
    if where == "challenger":
        kw["chal"] = challenger(0.6, trained_through=None)
    elif where == "champion":
        kw["champ"] = champion(0.5, trained_through=None)
    elif where == "start":
        kw["h"] = holdout(start=None)
    else:
        kw["h"] = holdout(end=None)
    with pytest.raises(LookaheadError):
        call(**kw)


# --- holdout window vs as_of ------------------------------------------------

def test_holdout_end_after_as_of_raises():
    with pytest.raises(LookaheadError):
        call(as_of=HOLDOUT_END - timedelta(days=1))


def test_holdout_end_equal_to_as_of_passes():
    assert call(as_of=HOLDOUT_END).chronology_checked


def test_holdout_start_after_end_raises():
    with pytest.raises(LookaheadError):
        call(h=holdout(start=HOLDOUT_END, end=HOLDOUT_START))


def test_default_as_of_is_now_and_future_holdout_raises():
    future = datetime.now(timezone.utc) + timedelta(days=365)
    h = holdout(start=future, end=future + timedelta(days=30))
    with pytest.raises(LookaheadError):
        gate(challenger=challenger(0.6), champion=None, metric=SpyMetric(),
             holdout=h, sink=ListSink())


# --- per-sample timestamps --------------------------------------------------

def test_valid_timestamps_pass():
    ts = daily(HOLDOUT_START, 10)
    assert call(h=holdout(timestamps=ts, n_samples=10)).chronology_checked


def test_equal_consecutive_timestamps_allowed():
    ts = [HOLDOUT_START, HOLDOUT_START, HOLDOUT_START + timedelta(days=1)]
    assert call(h=holdout(timestamps=ts)).chronology_checked


def test_out_of_order_timestamps_raise():
    ts = daily(HOLDOUT_START, 5)
    ts[2], ts[3] = ts[3], ts[2]
    with pytest.raises(LookaheadError, match="non-decreasing"):
        call(h=holdout(timestamps=ts))


def test_timestamp_before_window_raises():
    ts = [HOLDOUT_START - timedelta(seconds=1)] + daily(HOLDOUT_START, 3)
    with pytest.raises(LookaheadError):
        call(h=holdout(timestamps=ts))


def test_timestamp_after_window_raises():
    ts = daily(HOLDOUT_START, 3) + [HOLDOUT_END + timedelta(seconds=1)]
    with pytest.raises(LookaheadError):
        call(h=holdout(timestamps=ts))


def test_timestamps_count_mismatch_with_n_samples_raises():
    with pytest.raises(LookaheadError, match="n_samples"):
        call(h=holdout(timestamps=daily(HOLDOUT_START, 5), n_samples=6))


# --- nothing happens after a violation ----------------------------------------

@pytest.mark.parametrize("bad_kwargs", [
    {"chal": challenger(0.6, trained_through=HOLDOUT_START)},
    {"champ": champion(0.5, trained_through=HOLDOUT_START)},
    {"embargo": timedelta(days=5)},
    {"as_of": HOLDOUT_END - timedelta(days=1)},
    {"h": holdout(timestamps=list(reversed(daily(HOLDOUT_START, 3))))},
])
def test_violation_scores_nothing_and_logs_nothing(bad_kwargs):
    sink, m = ListSink(), SpyMetric()
    with pytest.raises(LookaheadError):
        call(sink=sink, m=m, **bad_kwargs)
    assert m.calls == []
    assert sink.records == []


# --- explicit opt-out ------------------------------------------------------

@pytest.mark.parametrize("bad_kwargs", [
    {"chal": challenger(0.6, trained_through=HOLDOUT_START + timedelta(days=3))},
    {"chal": challenger(0.6, trained_through=NAIVE)},
    {"chal": challenger(0.6, trained_through=None), "champ": champion(0.5, trained_through=None),
     "h": holdout(start=None, end=None)},
    {"embargo": timedelta(days=5)},
    {"as_of": HOLDOUT_END - timedelta(days=1)},
    {"h": holdout(timestamps=list(reversed(daily(HOLDOUT_START, 3))), n_samples=99)},
])
def test_allow_untimed_bypasses_checks_and_marks_record(bad_kwargs):
    sink = ListSink()
    rec = call(allow_untimed=True, sink=sink, **bad_kwargs)
    assert rec.chronology_checked is False
    assert sink.records == [rec]
    # Records produced in untimed mode must still serialise.
    rec.to_json()
