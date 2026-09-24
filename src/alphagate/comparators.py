"""Comparators decide whether a challenger's score beats the champion's.

Comparators only ever see two :class:`MetricResult` objects and the metric's
direction. Chronology, scoring, and logging are the gate's job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from .types import MetricResult

__all__ = [
    "Verdict",
    "Comparator",
    "MarginComparator",
    "NO_INCUMBENT",
    "IMPROVED",
    "NOT_IMPROVED",
    "INVALID_METRIC",
]

# Core reason codes. ``Verdict.reason_code`` is an open string so that new
# comparators can introduce their own codes without changing this module.
NO_INCUMBENT = "no_incumbent"
IMPROVED = "improved"
NOT_IMPROVED = "not_improved"
INVALID_METRIC = "invalid_metric"


@dataclass(frozen=True)
class Verdict:
    promote: bool
    reason_code: str
    explanation: str
    stats: Mapping[str, float] = field(default_factory=dict)


class Comparator(Protocol):
    name: str

    def params(self) -> Mapping[str, Any]:
        """Configuration recorded verbatim in every decision record."""
        ...

    def compare(
        self, champion: MetricResult, challenger: MetricResult, *, higher_is_better: bool
    ) -> Verdict: ...


class MarginComparator:
    """Promote iff the challenger beats the champion by strictly more than ``min_delta``.

    ``improvement`` is ``challenger - champion`` when higher is better and
    ``champion - challenger`` when lower is better. The challenger is promoted
    only if ``improvement > min_delta``; an improvement exactly equal to
    ``min_delta`` (including a tie at ``min_delta == 0``) is rejected, so the
    incumbent wins ties.

    No tolerance is applied: the comparison is exact IEEE-754 arithmetic, so
    choose ``min_delta`` with the metric's noise level in mind rather than
    relying on boundary behaviour. A raw margin is not a significance test;
    comparators that account for sampling noise or the number of candidates
    tried can be plugged in via the :class:`Comparator` protocol.
    """

    name = "margin"

    def __init__(self, min_delta: float = 0.0) -> None:
        min_delta = float(min_delta)
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError(f"min_delta must be a finite, non-negative number, got {min_delta!r}")
        self.min_delta = min_delta

    def params(self) -> Mapping[str, Any]:
        return {"min_delta": self.min_delta}

    def compare(
        self, champion: MetricResult, challenger: MetricResult, *, higher_is_better: bool
    ) -> Verdict:
        champ, chal = float(champion.value), float(challenger.value)
        if not (math.isfinite(champ) and math.isfinite(chal)):
            # Defensive: gate() filters these before calling any comparator.
            return Verdict(False, INVALID_METRIC, "non-finite score passed to comparator")

        improvement = (chal - champ) if higher_is_better else (champ - chal)
        stats = {
            "champion": champ,
            "challenger": chal,
            "improvement": improvement,
            "min_delta": self.min_delta,
        }
        direction = "higher" if higher_is_better else "lower"
        if improvement > self.min_delta:
            return Verdict(
                True,
                IMPROVED,
                f"challenger {chal!r} beats champion {champ!r} by {improvement!r} "
                f"({direction} is better), which exceeds min_delta {self.min_delta!r}",
                stats,
            )
        return Verdict(
            False,
            NOT_IMPROVED,
            f"challenger {chal!r} vs champion {champ!r}: improvement {improvement!r} "
            f"({direction} is better) does not exceed min_delta {self.min_delta!r}; "
            "incumbent retained",
            stats,
        )

    def __repr__(self) -> str:
        return f"MarginComparator(min_delta={self.min_delta!r})"
