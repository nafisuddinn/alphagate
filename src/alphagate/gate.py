"""The promotion gate: one champion/challenger decision, fully audited.

Order of operations in :func:`gate` (each step only runs if the previous one
succeeded):

1. Validate arguments (``embargo`` type/sign).
2. Chronology check. Any way future information could reach a score raises
   :class:`LookaheadError` here, before any model is scored or anything is
   logged. Skipped only with ``allow_untimed=True``, which is recorded.
3. Score the challenger, then the champion (if any), exactly once each, with
   the *identical* holdout object. Metric exceptions propagate unchanged and
   nothing is logged. Each returned MetricResult is snapshotted (``samples``
   copied to a tuple, ``details`` deep-copied and made read-only) so later
   mutation of the metric's own containers cannot alter the record.
4. Non-finite scores (NaN, +/-inf) reject with ``INVALID_METRIC`` without
   consulting the comparator. A broken score is never promoted, not even
   into an empty champion slot.
5. No champion: promote with ``NO_INCUMBENT``.
6. Otherwise: ``comparator.compare(champion_score, challenger_score, ...)``.
7. Build one :class:`GateRecord` and write it to the sink. If the sink fails,
   raise :class:`SinkError` and return nothing: an unlogged decision must not
   be acted on.
"""

from __future__ import annotations

import copy
import math
import numbers
import uuid
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any, Mapping, Optional

from ._version import __version__
from .comparators import INVALID_METRIC, NO_INCUMBENT, Comparator, MarginComparator, Verdict
from .errors import LookaheadError, SinkError
from .records import SCHEMA_VERSION, Decision, GateRecord
from .sinks import RecordSink
from .types import Candidate, Holdout, Metric, MetricResult

__all__ = ["gate"]


def gate(
    *,
    challenger: Candidate[Any],
    champion: Optional[Candidate[Any]],
    metric: Metric[Any, Any],
    holdout: Holdout[Any],
    sink: RecordSink,
    comparator: Optional[Comparator] = None,
    embargo: timedelta = timedelta(0),
    as_of: Optional[datetime] = None,
    allow_untimed: bool = False,
    context: Optional[Mapping[str, Any]] = None,
) -> GateRecord:
    """Decide whether ``challenger`` should replace ``champion``, and log why.

    Args:
        challenger: The candidate seeking promotion.
        champion: The current incumbent, or ``None`` if the slot is empty.
        metric: Scores a model on the holdout. ``metric.higher_is_better``
            sets the direction for the comparator.
        holdout: Out-of-sample data. Must lie strictly after every
            candidate's ``trained_through`` (plus ``embargo``) and end no
            later than ``as_of``.
        sink: Receives exactly one :class:`GateRecord` per successful call,
            for promotions and rejections alike.
        comparator: Decision rule. Defaults to ``MarginComparator()``
            (strict improvement, incumbent wins ties).
        embargo: Required gap between each candidate's ``trained_through``
            and ``holdout.start``. The rule is strict:
            ``trained_through + embargo < holdout.start``. Use a non-zero
            embargo when observations near the boundary share information
            (e.g. overlapping label windows).
        as_of: Timezone-aware reference time for the evaluation. The holdout
            may not end after it. Defaults to the current UTC time.
        allow_untimed: Skip every chronology check. Use only when timing
            information genuinely does not exist; the record is marked
            ``chronology_checked=False`` so the gap stays visible downstream.
        context: Free-form, JSON-serialisable data copied into the record.

    Returns:
        The :class:`GateRecord` that was written to ``sink``.

    Raises:
        LookaheadError: Chronology violation (nothing scored, nothing logged).
        SinkError: The record could not be persisted (no record returned).
        ValueError: ``embargo`` is negative.
        TypeError: Bad argument types, a metric that does not return a
            :class:`MetricResult`, or a comparator that does not return a
            :class:`Verdict`.
    """
    # 1. Arguments ------------------------------------------------------------
    if not isinstance(embargo, timedelta):
        raise TypeError(f"embargo must be a datetime.timedelta, got {type(embargo).__name__}")
    if embargo < timedelta(0):
        raise ValueError(f"embargo must be non-negative, got {embargo!r}")
    if comparator is None:
        comparator = MarginComparator()
    if as_of is None:
        as_of = datetime.now(timezone.utc)
    higher_is_better = bool(metric.higher_is_better)

    # 2. Chronology (before any scoring or logging) ----------------------------
    if not allow_untimed:
        _check_chronology(challenger, champion, holdout, embargo, as_of)

    # 3. Score each candidate once, on the identical holdout object ------------
    challenger_score = _score(metric, challenger, holdout, "challenger")
    champion_score = None if champion is None else _score(metric, champion, holdout, "champion")

    # 4-6. Decide ---------------------------------------------------------------
    verdict = _decide(comparator, challenger_score, champion_score, metric.name, higher_is_better)

    # 7. Record and persist -----------------------------------------------------
    timestamps = holdout.timestamps
    n_samples = holdout.n_samples
    if n_samples is None and timestamps is not None:
        n_samples = len(timestamps)

    record = GateRecord(
        schema_version=SCHEMA_VERSION,
        record_id=uuid.uuid4().hex,
        decided_at=datetime.now(timezone.utc),
        as_of=as_of,
        alphagate_version=__version__,
        decision=Decision.PROMOTE if verdict.promote else Decision.REJECT,
        reason_code=verdict.reason_code,
        explanation=verdict.explanation,
        challenger_id=challenger.id,
        champion_id=None if champion is None else champion.id,
        challenger_trained_through=challenger.trained_through,
        champion_trained_through=None if champion is None else champion.trained_through,
        holdout_start=holdout.start,
        holdout_end=holdout.end,
        holdout_fingerprint=holdout.fingerprint,
        holdout_n_samples=n_samples,
        embargo_seconds=embargo.total_seconds(),
        chronology_checked=not allow_untimed,
        metric_name=metric.name,
        higher_is_better=higher_is_better,
        challenger_score=challenger_score,
        champion_score=champion_score,
        comparator_name=str(getattr(comparator, "name", type(comparator).__name__)),
        comparator_params=_frozen(comparator.params()),
        comparator_stats=_frozen(verdict.stats),
        challenger_metadata=_frozen(challenger.metadata),
        champion_metadata=_frozen({} if champion is None else champion.metadata),
        context=_frozen(context or {}),
    )

    try:
        sink.write(record)
    except SinkError:
        raise
    except Exception as exc:
        raise SinkError(
            f"failed to persist decision record {record.record_id} "
            f"({record.decision.value}, {record.reason_code}) to {sink!r}: {exc}"
        ) from exc
    return record


# --- chronology ---------------------------------------------------------------


def _is_aware(dt: datetime) -> bool:
    return dt.tzinfo is not None and dt.utcoffset() is not None


def _require_aware(dt: Any, label: str) -> None:
    if not isinstance(dt, datetime):
        raise LookaheadError(f"{label} must be a datetime, got {type(dt).__name__}")
    if not _is_aware(dt):
        raise LookaheadError(
            f"{label} ({dt.isoformat()}) is timezone-naive (no tzinfo); naive datetimes "
            "cannot be ordered unambiguously against the holdout window. Attach a tz "
            "(e.g. timezone.utc), or pass allow_untimed=True to skip chronology checks."
        )


def _check_chronology(
    challenger: Candidate[Any],
    champion: Optional[Candidate[Any]],
    holdout: Holdout[Any],
    embargo: timedelta,
    as_of: datetime,
) -> None:
    candidates = [("challenger", challenger)]
    if champion is not None:
        candidates.append(("champion", champion))

    # Missing timing information.
    for label, cand in candidates:
        if cand.trained_through is None:
            raise LookaheadError(
                f"{label} {cand.id!r} has trained_through=None, so leakage into the holdout "
                "cannot be ruled out. Set it, or pass allow_untimed=True."
            )
    if holdout.start is None or holdout.end is None:
        raise LookaheadError(
            "holdout.start and holdout.end are required for the chronology check "
            "(or pass allow_untimed=True)."
        )

    # tz-awareness, checked before any ordering comparison (comparing naive with
    # aware datetimes would otherwise fail with an unhelpful TypeError).
    for label, cand in candidates:
        _require_aware(cand.trained_through, f"{label} {cand.id!r} trained_through")
    _require_aware(holdout.start, "holdout.start")
    _require_aware(holdout.end, "holdout.end")
    _require_aware(as_of, "as_of")
    timestamps = None if holdout.timestamps is None else list(holdout.timestamps)
    if timestamps is not None:
        for i, ts in enumerate(timestamps):
            _require_aware(ts, f"holdout.timestamps[{i}]")

    # Window sanity.
    if holdout.start > holdout.end:
        raise LookaheadError(
            f"holdout.start {holdout.start.isoformat()} is after holdout.end {holdout.end.isoformat()}"
        )
    if holdout.end > as_of:
        raise LookaheadError(
            f"holdout.end {holdout.end.isoformat()} is after as_of {as_of.isoformat()}: the "
            "holdout contains observations from the future relative to the evaluation time"
        )

    # Training data must end strictly before the holdout, with the embargo gap.
    # Written as a difference (not trained_through + embargo) to avoid datetime
    # overflow near datetime.max.
    for label, cand in candidates:
        tt = cand.trained_through
        assert tt is not None  # checked above
        gap = holdout.start - tt
        if gap <= embargo:
            raise LookaheadError(
                f"{label} {cand.id!r} trained_through {tt.isoformat()} is not strictly more than "
                f"the embargo ({embargo}) before holdout.start {holdout.start.isoformat()} "
                f"(gap {gap}); its training data may overlap or leak into the holdout"
            )

    # Per-observation timestamps.
    if timestamps is not None:
        if holdout.n_samples is not None and holdout.n_samples != len(timestamps):
            raise LookaheadError(
                f"holdout.n_samples={holdout.n_samples} does not match "
                f"len(holdout.timestamps)={len(timestamps)}"
            )
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i - 1]:
                raise LookaheadError(
                    f"holdout.timestamps must be non-decreasing; index {i} "
                    f"({timestamps[i].isoformat()}) precedes index {i - 1} "
                    f"({timestamps[i - 1].isoformat()})"
                )
        for i, ts in enumerate(timestamps):
            if ts < holdout.start or ts > holdout.end:
                raise LookaheadError(
                    f"holdout.timestamps[{i}] {ts.isoformat()} lies outside the holdout window "
                    f"[{holdout.start.isoformat()}, {holdout.end.isoformat()}]"
                )


# --- scoring and decision --------------------------------------------------------


def _score(metric: Metric[Any, Any], cand: Candidate[Any], holdout: Holdout[Any], label: str) -> MetricResult:
    result = metric(cand.model, holdout)
    if not isinstance(result, MetricResult):
        raise TypeError(
            f"metric {metric.name!r} returned {type(result).__name__} for {label} {cand.id!r}; "
            "expected MetricResult (wrap plain functions with alphagate.metric())"
        )
    value = result.value
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"metric {metric.name!r} produced a non-numeric value {value!r} "
            f"({type(value).__name__}) for {label} {cand.id!r}"
        )
    # Snapshot immediately, so the comparator and the GateRecord both see the
    # same frozen values, and neither the metric (by retaining a reference to
    # its own containers) nor the comparator can alter the record afterwards.
    return _frozen_result(result)


def _decide(
    comparator: Comparator,
    challenger_score: MetricResult,
    champion_score: Optional[MetricResult],
    metric_name: str,
    higher_is_better: bool,
) -> Verdict:
    bad = [
        f"{label} score {float(score.value)!r}"
        for label, score in (("challenger", challenger_score), ("champion", champion_score))
        if score is not None and not math.isfinite(float(score.value))
    ]
    if bad:
        return Verdict(
            False,
            INVALID_METRIC,
            f"metric {metric_name!r} produced a non-finite value ({', '.join(bad)}); "
            "non-finite scores cannot be compared and are never promoted",
        )

    if champion_score is None:
        return Verdict(
            True,
            NO_INCUMBENT,
            f"no incumbent: challenger promoted with {metric_name!r} = "
            f"{float(challenger_score.value)!r} (no comparison was made)",
        )

    verdict = comparator.compare(champion_score, challenger_score, higher_is_better=higher_is_better)
    if not isinstance(verdict, Verdict):
        raise TypeError(
            f"comparator {getattr(comparator, 'name', comparator)!r} returned "
            f"{type(verdict).__name__}; expected Verdict"
        )
    return verdict


# --- helpers ---------------------------------------------------------------------


def _frozen(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read-only deep copy, so later mutation by the caller cannot alter the record.

    Falls back to a shallow copy for values that cannot be deep-copied; such
    values are not JSON-serialisable either, so serialising the record will
    raise ``TypeError`` rather than silently recording something wrong.
    """
    try:
        copied = copy.deepcopy(dict(mapping))
    except Exception:
        copied = dict(mapping)
    return MappingProxyType(copied)


def _frozen_result(result: MetricResult) -> MetricResult:
    """Return a new MetricResult whose containers are private, read-only copies.

    ``MetricResult`` is a frozen dataclass, but that only stops attribute
    reassignment: a metric that returns ``samples=[...]`` or ``details={...}``
    and keeps a reference could mutate those containers after :func:`gate`
    returns, silently rewriting an "immutable" decision record. So:

    * ``samples`` is copied into a ``tuple``. Array-likes exposing ``tolist()``
      (e.g. numpy arrays) are converted first, so the snapshot holds plain
      Python values rather than a view onto a caller-owned buffer.
    * ``details`` gets the same deep-copy-and-``MappingProxyType`` treatment
      as the other mapping fields (see :func:`_frozen`).

    Values that cannot be converted are deep-copied as-is (or kept, if that
    fails too); serialising the record will then raise ``TypeError`` rather
    than recording something wrong.
    """
    samples = result.samples
    if samples is not None:
        try:
            tolist = getattr(samples, "tolist", None)
            raw = tolist() if callable(tolist) else samples
            samples = tuple(copy.deepcopy(list(raw)))
        except Exception:
            try:
                samples = copy.deepcopy(samples)
            except Exception:
                pass
    details = _frozen(result.details if result.details is not None else {})
    return MetricResult(value=result.value, samples=samples, details=details)
