"""Core value types: candidates, holdout data, metric results, metrics.

alphagate treats models and holdout payloads as opaque. It never inspects
them; it only hands them to the caller-supplied metric.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Generic, Mapping, Protocol, Sequence, TypeVar, Union

__all__ = ["Candidate", "Holdout", "MetricResult", "Metric", "metric"]

M = TypeVar("M")  # opaque model type
D = TypeVar("D")  # opaque holdout payload type


@dataclass(frozen=True)
class Candidate(Generic[M]):
    """A model competing for (or holding) the champion slot.

    Attributes:
        model: The model object. Opaque to alphagate.
        id: Stable identifier used in decision records (e.g. a registry
            version or content hash).
        trained_through: Timezone-aware timestamp of the *latest* data point
            used for ANY part of producing this model: fitting, feature
            normalisation statistics, hyperparameter tuning, early stopping,
            threshold selection. If tuning looked at data up to time T, this
            must be >= T. Understating it defeats the lookahead check.
            ``None`` is only accepted with ``gate(..., allow_untimed=True)``.
        metadata: Free-form, JSON-serialisable information copied into the
            decision record.
    """

    model: M
    id: str
    trained_through: datetime | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Candidate.id must be a non-empty string")


@dataclass(frozen=True)
class Holdout(Generic[D]):
    """Out-of-sample evaluation data plus the timing facts needed to audit it.

    Attributes:
        data: The payload passed to the metric. Opaque to alphagate.
        start: Timezone-aware timestamp of the earliest observation (inclusive).
        end: Timezone-aware timestamp of the latest observation (inclusive).
        timestamps: Optional per-observation timestamps; if given, they are
            checked to be non-decreasing and inside ``[start, end]``.
        fingerprint: Optional content hash, recorded for reproducibility.
        n_samples: Optional observation count; checked against
            ``len(timestamps)`` when both are given.
    """

    data: D
    start: datetime | None
    end: datetime | None
    timestamps: Sequence[datetime] | None = None
    fingerprint: str | None = None
    n_samples: int | None = None

    def __post_init__(self) -> None:
        if self.n_samples is not None and self.n_samples < 0:
            raise ValueError("Holdout.n_samples must be non-negative")


@dataclass(frozen=True)
class MetricResult:
    """Outcome of scoring one model on one holdout.

    Attributes:
        value: The scalar the comparator decides on.
        samples: Optional per-observation (or per-period) values behind
            ``value``. Reserved for comparators that need a distribution,
            e.g. bootstrap or multiple-testing-adjusted tests.
        details: Optional named auxiliary numbers, recorded verbatim.
    """

    value: float
    samples: Sequence[float] | None = None
    details: Mapping[str, float] = field(default_factory=dict)


class Metric(Protocol[M, D]):
    """Scores a model on a holdout. Must be a pure function of its inputs."""

    name: str
    higher_is_better: bool

    def __call__(self, model: M, holdout: Holdout[D]) -> MetricResult: ...


class _FunctionMetric:
    """Adapter produced by :func:`metric`."""

    def __init__(self, name: str, fn: Callable[[Any, Holdout[Any]], Any], higher_is_better: bool) -> None:
        self.name = name
        self.higher_is_better = higher_is_better
        self._fn = fn

    def __call__(self, model: Any, holdout: Holdout[Any]) -> MetricResult:
        out = self._fn(model, holdout)
        if isinstance(out, MetricResult):
            return out
        if isinstance(out, bool) or not isinstance(out, numbers.Real):
            # bool is excluded: a metric returning True/False is almost always a bug.
            raise TypeError(
                f"metric {self.name!r} must return a float or MetricResult, got {type(out).__name__}"
            )
        return MetricResult(value=float(out))

    def __repr__(self) -> str:
        direction = "higher" if self.higher_is_better else "lower"
        return f"metric({self.name!r}, {direction}_is_better)"


def metric(
    name: str,
    fn: Callable[[M, Holdout[D]], Union[float, MetricResult]],
    *,
    higher_is_better: bool = True,
) -> Metric[M, D]:
    """Wrap a plain ``fn(model, holdout) -> float | MetricResult`` as a Metric.

    A bare number returned by ``fn`` is wrapped as ``MetricResult(value=...)``.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("metric name must be a non-empty string")
    if not callable(fn):
        raise TypeError("fn must be callable")
    return _FunctionMetric(name, fn, bool(higher_is_better))
