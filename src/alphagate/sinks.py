"""Record sinks: where :class:`GateRecord` objects go once a decision is made.

A sink has one job: durably accept a record or raise. :func:`alphagate.gate`
wraps any exception raised by ``write()`` in :class:`SinkError` and does not
return the decision, so a sink must never swallow its own failures.
"""

from __future__ import annotations

import os
from typing import List, Protocol, Union, runtime_checkable

from .records import GateRecord

__all__ = ["RecordSink", "JsonlSink", "ListSink", "NullSink"]


@runtime_checkable
class RecordSink(Protocol):
    def write(self, record: GateRecord) -> None:
        """Persist ``record`` or raise. Must not fail silently."""
        ...


class JsonlSink:
    """Append each record as one line of strict JSON to a file.

    The record is serialised *before* the file is opened, so a record that
    cannot be serialised raises without touching the file. Each call opens the
    file in append mode, writes the complete line in a single call, and closes
    it, so no handle is left open between decisions. The parent directory must
    already exist; a missing directory raises (and surfaces as
    :class:`SinkError` from :func:`alphagate.gate`).
    """

    def __init__(self, path: Union[str, "os.PathLike[str]"]) -> None:
        self.path = os.fspath(path)

    def write(self, record: GateRecord) -> None:
        line = record.to_json() + "\n"
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()

    def __repr__(self) -> str:
        return f"JsonlSink({self.path!r})"


class ListSink:
    """Keep records in memory, in call order, on ``.records``. Intended for tests."""

    def __init__(self) -> None:
        self.records: List[GateRecord] = []

    def write(self, record: GateRecord) -> None:
        self.records.append(record)

    def __repr__(self) -> str:
        return f"ListSink(<{len(self.records)} records>)"


class NullSink:
    """Discard every record.

    Only use this where the decision genuinely does not need an audit trail
    (e.g. unit tests); it defeats the point of logging rejected candidates.
    """

    def write(self, record: GateRecord) -> None:
        return None

    def __repr__(self) -> str:
        return "NullSink()"
