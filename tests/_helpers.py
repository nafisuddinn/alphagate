"""Shared test helpers: spies and a canonical, chronologically clean timeline.

Canonical timeline (all UTC):
    champion trained through   2023-12-31
    challenger trained through 2024-01-31
    holdout window             2024-02-01 .. 2024-03-01 (inclusive)
    as_of                      2024-06-01
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from alphagate import Candidate, Holdout, MarginComparator, MetricResult

UTC = timezone.utc

CHAMPION_TT = datetime(2023, 12, 31, tzinfo=UTC)
CHALLENGER_TT = datetime(2024, 1, 31, tzinfo=UTC)
HOLDOUT_START = datetime(2024, 2, 1, tzinfo=UTC)
HOLDOUT_END = datetime(2024, 3, 1, tzinfo=UTC)
AS_OF = datetime(2024, 6, 1, tzinfo=UTC)


@dataclass(frozen=True)
class FakeModel:
    """Opaque model stand-in. The spy metric just reads `.score`."""

    score: float


class SpyMetric:
    """Metric that records every call and returns the model's preset score."""

    def __init__(self, *, higher_is_better: bool = True, name: str = "score") -> None:
        self.name = name
        self.higher_is_better = higher_is_better
        self.calls: list[tuple[Any, Any]] = []

    def __call__(self, model: FakeModel, holdout: Holdout) -> MetricResult:
        self.calls.append((model, holdout))
        return MetricResult(value=model.score)


class SpyComparator:
    """Delegates to MarginComparator but records whether it was called."""

    def __init__(self, min_delta: float = 0.0) -> None:
        self._inner = MarginComparator(min_delta=min_delta)
        self.name = "spy"
        self.calls: list[tuple[MetricResult, MetricResult, bool]] = []

    def params(self):
        return self._inner.params()

    def compare(self, champion, challenger, *, higher_is_better):
        self.calls.append((champion, challenger, higher_is_better))
        return self._inner.compare(champion, challenger, higher_is_better=higher_is_better)


def challenger(score: float, *, trained_through: datetime | None = CHALLENGER_TT, **kw) -> Candidate:
    return Candidate(model=FakeModel(score), id=kw.pop("id", "challenger-v2"),
                     trained_through=trained_through, **kw)


def champion(score: float, *, trained_through: datetime | None = CHAMPION_TT, **kw) -> Candidate:
    return Candidate(model=FakeModel(score), id=kw.pop("id", "champion-v1"),
                     trained_through=trained_through, **kw)


def holdout(
    *,
    start: datetime | None = HOLDOUT_START,
    end: datetime | None = HOLDOUT_END,
    timestamps=None,
    n_samples: int | None = None,
    fingerprint: str | None = "sha256:abc123",
) -> Holdout:
    return Holdout(data=object(), start=start, end=end, timestamps=timestamps,
                   n_samples=n_samples, fingerprint=fingerprint)


def daily(start: datetime, n: int) -> list[datetime]:
    return [start + timedelta(days=i) for i in range(n)]
