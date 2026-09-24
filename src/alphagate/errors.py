"""Exception hierarchy for alphagate."""

from __future__ import annotations

__all__ = ["AlphagateError", "LookaheadError", "SinkError"]


class AlphagateError(Exception):
    """Base class for every error alphagate raises deliberately."""


class LookaheadError(AlphagateError):
    """The evaluation setup could let future information leak into a score.

    Raised by :func:`alphagate.gate` *before* any candidate is scored or any
    record is written, e.g. when a candidate's training data reaches into the
    holdout window, a timestamp is timezone-naive, or the holdout extends past
    the evaluation reference time.
    """


class SinkError(AlphagateError):
    """The decision record could not be persisted.

    When this is raised, :func:`alphagate.gate` does not return a decision:
    an unlogged decision must not be acted on. The original exception is
    available as ``__cause__``.
    """
