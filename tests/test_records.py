"""GateRecord content, serialisation, and sink behaviour."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from alphagate import (
    INVALID_METRIC,
    SCHEMA_VERSION,
    AlphagateError,
    Decision,
    GateRecord,
    JsonlSink,
    ListSink,
    MetricResult,
    NullSink,
    SinkError,
    __version__,
    gate,
    metric,
)

from _helpers import AS_OF, HOLDOUT_END, HOLDOUT_START, SpyMetric, challenger, champion, holdout


def _strict_loads(s: str):
    """json.loads that refuses the non-standard NaN/Infinity tokens."""
    def reject(tok):
        raise ValueError(f"non-standard JSON constant {tok!r}")
    return json.loads(s, parse_constant=reject)


def run(chal_score=0.6, champ_score=0.5, *, sink, **kw):
    return gate(
        challenger=challenger(chal_score, metadata=kw.pop("chal_meta", {})),
        champion=None if champ_score is None else champion(champ_score),
        metric=kw.pop("m", SpyMetric()),
        holdout=kw.pop("h", holdout()),
        sink=sink,
        as_of=AS_OF,
        **kw,
    )


# --- exactly one record per call ------------------------------------------------

@pytest.mark.parametrize("chal,champ,expected", [
    (0.6, 0.5, Decision.PROMOTE),
    (0.4, 0.5, Decision.REJECT),
    (0.5, 0.5, Decision.REJECT),
    (0.6, None, Decision.PROMOTE),
    (math.nan, 0.5, Decision.REJECT),
])
def test_every_call_writes_exactly_one_record(chal, champ, expected):
    sink = ListSink()
    rec = run(chal, champ, sink=sink)
    assert rec.decision is expected
    assert len(sink.records) == 1
    assert sink.records[0] is rec


def test_rejected_challenger_is_logged_with_reason():
    sink = ListSink()
    run(0.4, 0.5, sink=sink)
    (rec,) = sink.records
    assert rec.decision is Decision.REJECT
    assert rec.challenger_id == "challenger-v2"
    assert rec.champion_id == "champion-v1"
    assert rec.explanation  # human-readable reason is non-empty


# --- record content -----------------------------------------------------------

def test_record_fields_populated():
    sink = ListSink()
    rec = run(sink=sink, embargo=timedelta(hours=12), context={"run": "nightly-42"},
              chal_meta={"git_sha": "deadbeef"},
              h=holdout(n_samples=20, fingerprint="sha256:ff"))
    assert isinstance(rec, GateRecord)
    assert rec.schema_version == SCHEMA_VERSION
    assert rec.alphagate_version == __version__
    assert len(rec.record_id) == 32 and int(rec.record_id, 16) >= 0  # uuid4 hex
    assert rec.decided_at.tzinfo is not None
    assert rec.as_of == AS_OF
    assert rec.holdout_start == HOLDOUT_START
    assert rec.holdout_end == HOLDOUT_END
    assert rec.holdout_fingerprint == "sha256:ff"
    assert rec.holdout_n_samples == 20
    assert rec.embargo_seconds == 12 * 3600
    assert rec.context == {"run": "nightly-42"}
    assert rec.challenger_metadata == {"git_sha": "deadbeef"}
    assert rec.champion_metadata == {}


def test_record_is_insulated_from_later_caller_mutation():
    meta = {"k": 1}
    ctx = {"c": 1}
    sink = ListSink()
    rec = run(sink=sink, chal_meta=meta, context=ctx)
    meta["k"] = 999
    ctx["c"] = 999
    assert rec.challenger_metadata == {"k": 1}
    assert rec.context == {"c": 1}


def test_record_is_insulated_from_later_metric_result_mutation():
    """Regression: MetricResult.samples/.details used to be stored by reference.

    A metric that returns mutable containers and keeps a handle on them must
    not be able to rewrite the record after gate() returns.
    """
    retained: dict[str, tuple[list, dict]] = {}

    def fn(model, h):
        samples = [model.score, model.score / 2]
        details = {"n": 2.0, "nested": {"x": 1.0}}
        retained[f"{model.score}"] = (samples, details)
        return MetricResult(value=model.score, samples=samples, details=details)

    m = metric("m", fn)
    sink = ListSink()
    rec = run(sink=sink, m=m)
    before = rec.to_dict()

    for samples, details in retained.values():
        samples[0] = 999.0
        samples.append(-1.0)
        details["n"] = 999.0
        details["injected"] = 1.0
        details["nested"]["x"] = 999.0

    assert len(retained) == 2  # both challenger and champion were exercised
    assert rec.challenger_score.samples == (0.6, 0.3)
    assert rec.challenger_score.details == {"n": 2.0, "nested": {"x": 1.0}}
    assert rec.champion_score is not None
    assert rec.champion_score.samples == (0.5, 0.25)
    assert rec.champion_score.details == {"n": 2.0, "nested": {"x": 1.0}}
    assert rec.to_dict() == before
    assert sink.records[0] is rec


def test_recorded_metric_result_containers_are_read_only():
    m = metric("m", lambda model, h: MetricResult(value=model.score, samples=[0.1, 0.2],
                                                  details={"n": 2.0}))
    rec = run(sink=ListSink(), m=m)
    assert isinstance(rec.challenger_score.samples, tuple)
    with pytest.raises(TypeError):
        rec.challenger_score.details["n"] = 0.0  # type: ignore[index]


def test_comparator_cannot_mutate_recorded_scores():
    """The comparator sees the same snapshot that is recorded, read-only."""
    from alphagate import MarginComparator

    class Meddler:
        name = "meddler"

        def params(self):
            return {}

        def compare(self, champion, challenger, *, higher_is_better):
            with pytest.raises(TypeError):
                challenger.details["n"] = 999.0
            return MarginComparator().compare(champion, challenger,
                                              higher_is_better=higher_is_better)

    m = metric("m", lambda model, h: MetricResult(value=model.score, samples=[0.1],
                                                  details={"n": 2.0}))
    rec = run(sink=ListSink(), m=m, comparator=Meddler())
    assert rec.challenger_score.details == {"n": 2.0}


def test_array_like_samples_are_snapshotted_to_plain_tuple():
    """Array-likes exposing tolist() (numpy-style) are copied out, not aliased."""

    class ArrayLike:  # duck-typed stand-in; alphagate must not depend on numpy
        def __init__(self, xs):
            self.buf = list(xs)

        def tolist(self):
            return list(self.buf)

        def __iter__(self):
            return iter(self.buf)

        def __len__(self):
            return len(self.buf)

    arr = ArrayLike([0.1, 0.2, 0.3])
    m = metric("m", lambda model, h: MetricResult(value=model.score, samples=arr))
    rec = run(sink=ListSink(), m=m)
    arr.buf[0] = 999.0
    assert rec.challenger_score.samples == (0.1, 0.2, 0.3)
    assert all(type(x) is float for x in rec.challenger_score.samples)


def test_record_ids_are_unique():
    sink = ListSink()
    for _ in range(5):
        run(sink=sink)
    assert len({r.record_id for r in sink.records}) == 5


# --- serialisation --------------------------------------------------------------

def test_to_json_round_trips_to_to_dict():
    sink = ListSink()
    rec = run(sink=sink, context={"nested": {"a": [1, 2.5, "x"]}, "when": AS_OF})
    assert _strict_loads(rec.to_json()) == rec.to_dict()


def test_to_dict_shapes():
    sink = ListSink()
    rec = run(sink=sink)
    d = rec.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["decision"] == "promote"
    assert d["holdout_start"] == "2024-02-01T00:00:00+00:00"
    assert d["challenger_score"] == {"value": 0.6, "samples": None, "details": {}}
    assert d["champion_score"]["value"] == 0.5


def test_datetimes_serialised_as_utc():
    plus5 = timezone(timedelta(hours=5))
    tt = datetime(2024, 1, 20, 5, 0, tzinfo=plus5)  # 2024-01-20T00:00Z
    sink = ListSink()
    rec = gate(challenger=challenger(0.6, trained_through=tt), champion=None,
               metric=SpyMetric(), holdout=holdout(), sink=sink, as_of=AS_OF)
    assert rec.to_dict()["challenger_trained_through"] == "2024-01-20T00:00:00+00:00"
    assert rec.to_dict()["decided_at"].endswith("+00:00")


def test_non_finite_scores_serialise_to_strict_json():
    sink = ListSink()
    rec = run(math.nan, math.inf, sink=sink)
    assert rec.reason_code == INVALID_METRIC
    d = _strict_loads(rec.to_json())
    assert d["challenger_score"]["value"] == "NaN"
    assert d["champion_score"]["value"] == "Infinity"


def test_samples_and_details_serialised():
    m = metric("m", lambda model, h: MetricResult(value=model.score, samples=(0.1, 0.2),
                                                  details={"n": 2.0}))
    sink = ListSink()
    rec = run(sink=sink, m=m)
    d = _strict_loads(rec.to_json())
    assert d["challenger_score"]["samples"] == [0.1, 0.2]
    assert d["challenger_score"]["details"] == {"n": 2.0}


def test_unserialisable_metadata_is_type_error():
    sink = ListSink()
    rec = run(sink=sink, chal_meta={"obj": object()})
    with pytest.raises(TypeError):
        rec.to_json()


# --- sinks ------------------------------------------------------------------

def test_jsonl_sink_appends_one_line_per_record(tmp_path):
    path = tmp_path / "gate.jsonl"
    sink = JsonlSink(path)
    r1 = run(0.6, 0.5, sink=sink)
    r2 = run(0.4, 0.5, sink=sink)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [_strict_loads(line)["record_id"] for line in lines] == [r1.record_id, r2.record_id]
    assert _strict_loads(lines[1])["decision"] == "reject"


def test_null_sink_discards():
    rec = run(sink=NullSink())
    assert rec.promoted


def test_failing_sink_raises_sink_error_and_returns_nothing():
    class Exploding:
        def write(self, record):
            raise OSError("disk full")

    result = "sentinel"
    with pytest.raises(SinkError) as exc:
        result = run(sink=Exploding())
    assert result == "sentinel"  # gate() never returned a record
    assert isinstance(exc.value.__cause__, OSError)
    assert issubclass(SinkError, AlphagateError)


def test_jsonl_sink_missing_directory_becomes_sink_error(tmp_path):
    with pytest.raises(SinkError):
        run(sink=JsonlSink(tmp_path / "missing-dir" / "gate.jsonl"))
