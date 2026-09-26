"""The decision record: one immutable, serialisable entry per gate() call.

Every call to :func:`alphagate.gate` produces exactly one :class:`GateRecord`,
whether the challenger is promoted or rejected. Rejections are logged with the
same detail as promotions so that "why did this candidate lose?" is always
answerable after the fact.
"""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from .types import MetricResult

__all__ = ["SCHEMA_VERSION", "Decision", "GateRecord"]

# Bump when a field is renamed/removed or its meaning changes. Adding a field
# is backwards compatible for readers that ignore unknown keys.
SCHEMA_VERSION = 1


class Decision(str, Enum):
    PROMOTE = "promote"
    REJECT = "reject"


@dataclass(frozen=True)
class GateRecord:
    schema_version: int
    record_id: str
    decided_at: datetime
    as_of: datetime
    alphagate_version: str
    decision: Decision
    reason_code: str
    explanation: str
    challenger_id: str
    champion_id: str | None
    challenger_trained_through: datetime | None
    champion_trained_through: datetime | None
    holdout_start: datetime | None
    holdout_end: datetime | None
    holdout_fingerprint: str | None
    holdout_n_samples: int | None
    embargo_seconds: float
    chronology_checked: bool
    metric_name: str
    higher_is_better: bool
    challenger_score: MetricResult
    champion_score: MetricResult | None
    comparator_name: str
    comparator_params: Mapping[str, Any]
    comparator_stats: Mapping[str, float]
    challenger_metadata: Mapping[str, Any]
    champion_metadata: Mapping[str, Any]
    context: Mapping[str, Any]

    @property
    def promoted(self) -> bool:
        return self.decision is Decision.PROMOTE

    def to_dict(self) -> dict[str, Any]:
        """Return a strictly JSON-safe dict.

        * Timezone-aware datetimes become ISO-8601 strings in UTC. Naive
          datetimes (only possible with ``allow_untimed=True``) are emitted
          without an offset rather than being silently assumed to be UTC.
        * Non-finite floats become the strings ``"NaN"``, ``"Infinity"``,
          ``"-Infinity"`` so the output is valid strict JSON.
        * Anything not representable raises ``TypeError``.
        """
        return {f.name: _jsonable(getattr(self, f.name), f.name) for f in fields(self)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=False)


def _jsonable(obj: Any, path: str) -> Any:
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, Enum):
        return _jsonable(obj.value, path)
    if isinstance(obj, datetime):
        if obj.tzinfo is None or obj.utcoffset() is None:
            return obj.isoformat()
        return obj.astimezone(timezone.utc).isoformat()
    if isinstance(obj, MetricResult):
        return {
            "value": _jsonable(obj.value, f"{path}.value"),
            "samples": None if obj.samples is None else _jsonable(obj.samples, f"{path}.samples"),
            "details": _jsonable(obj.details, f"{path}.details"),
        }
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        x = float(obj)
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "Infinity" if x > 0 else "-Infinity"
        return x
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"{path}: mapping keys must be str, got {type(k).__name__}")
            out[k] = _jsonable(v, f"{path}.{k}")
        return out
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, f"{path}[{i}]") for i, v in enumerate(obj)]
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):  # array-likes, without depending on any array library
        return _jsonable(tolist(), path)
    raise TypeError(f"{path}: value of type {type(obj).__name__} is not JSON-serialisable")
