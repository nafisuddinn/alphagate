"""alphagate: domain-agnostic champion/challenger promotion gating.

Score a challenger and the current champion on the same chronologically clean
holdout, decide whether to promote, and log every decision (promotions and
rejections alike) as an immutable, serialisable record.
"""

from ._version import __version__
from .comparators import (
    IMPROVED,
    INVALID_METRIC,
    NO_INCUMBENT,
    NOT_IMPROVED,
    Comparator,
    MarginComparator,
    Verdict,
)
from .errors import AlphagateError, LookaheadError, SinkError
from .gate import gate
from .records import SCHEMA_VERSION, Decision, GateRecord
from .sinks import JsonlSink, ListSink, NullSink, RecordSink
from .types import Candidate, Holdout, Metric, MetricResult, metric

__all__ = [
    "__version__",
    # core types
    "Candidate",
    "Holdout",
    "MetricResult",
    "Metric",
    "metric",
    # comparison
    "Verdict",
    "Comparator",
    "MarginComparator",
    "NO_INCUMBENT",
    "IMPROVED",
    "NOT_IMPROVED",
    "INVALID_METRIC",
    # records and sinks
    "GateRecord",
    "Decision",
    "SCHEMA_VERSION",
    "RecordSink",
    "JsonlSink",
    "ListSink",
    "NullSink",
    # entry point
    "gate",
    # errors
    "AlphagateError",
    "LookaheadError",
    "SinkError",
]
