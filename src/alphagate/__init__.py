"""alphagate: domain-agnostic champion/challenger promotion gating.

Score a challenger and the current champion on the same chronologically clean
holdout, decide whether to promote, and log every decision (promotions and
rejections alike) as an immutable, serialisable record.
"""

from ._version import __version__
from .comparators import (
    IMPROVED,
    INFERIOR,
    INSUFFICIENT_SAMPLES,
    INVALID_METRIC,
    NO_INCUMBENT,
    NON_INFERIOR,
    NOT_IMPROVED,
    NOT_SIGNIFICANT,
    SAMPLES_MISMATCH,
    SIGNIFICANT_IMPROVEMENT,
    Comparator,
    MarginComparator,
    PairedComparator,
    PairedTest,
    Verdict,
    paired_test,
    spend_alpha,
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
    "PairedComparator",
    "PairedTest",
    "paired_test",
    "spend_alpha",
    "NO_INCUMBENT",
    "IMPROVED",
    "NOT_IMPROVED",
    "INVALID_METRIC",
    "SIGNIFICANT_IMPROVEMENT",
    "NOT_SIGNIFICANT",
    "NON_INFERIOR",
    "INFERIOR",
    "INSUFFICIENT_SAMPLES",
    "SAMPLES_MISMATCH",
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
