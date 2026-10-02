"""Comparators decide whether a challenger's score beats the champion's.

Comparators only ever see two :class:`MetricResult` objects and the metric's
direction. Chronology, scoring, and logging are the gate's job.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence

from .types import MetricResult

__all__ = [
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
]

# Core reason codes. ``Verdict.reason_code`` is an open string so that new
# comparators can introduce their own codes without changing this module.
NO_INCUMBENT = "no_incumbent"
IMPROVED = "improved"
NOT_IMPROVED = "not_improved"
INVALID_METRIC = "invalid_metric"

# PairedComparator reason codes.
SIGNIFICANT_IMPROVEMENT = "significant_improvement"
NOT_SIGNIFICANT = "not_significant"
NON_INFERIOR = "non_inferior"
INFERIOR = "inferior"
INSUFFICIENT_SAMPLES = "insufficient_samples"
SAMPLES_MISMATCH = "samples_mismatch"


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


# --- paired significance test -------------------------------------------------------


@dataclass(frozen=True)
class PairedTest:
    """Result of :func:`paired_test`.

    Attributes:
        mean_diff: mean of ``a[i] - b[i]`` (never shifted by ``null``).
        se: Newey-West (HAC) standard error of ``mean_diff``.
        t: ``(mean_diff - null) / se``.
        p: one-sided p-value for H1: ``E[a - b] > null`` under the asymptotic
            standard normal distribution of ``t``.
        n: number of pairs.
        lags: number of autocovariance lags actually used.
    """

    mean_diff: float
    se: float
    t: float
    p: float
    n: int
    lags: int


def _default_lags(n: int) -> int:
    # Newey & West (1994) rule of thumb for the Bartlett kernel.
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _as_float_list(values: Sequence[Any], label: str) -> list:
    out = []
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            raise ValueError(f"{label}[{i}] is not a real number: {v!r}")
        x = float(v)
        if not math.isfinite(x):
            raise ValueError(f"{label}[{i}] is not finite: {x!r}")
        out.append(x)
    return out


def paired_test(
    a: Sequence[float],
    b: Sequence[float],
    lags: Optional[int] = None,
    *,
    null: float = 0.0,
) -> PairedTest:
    """One-sided Diebold-Mariano test on paired per-period values.

    Tests H0: ``E[a_i - b_i] <= null`` against H1: ``E[a_i - b_i] > null``
    using the differences ``d_i = a_i - b_i``, in the order given.

    Reference: Diebold, F. X. & Mariano, R. S. (1995), "Comparing Predictive
    Accuracy", Journal of Business & Economic Statistics 13(3), 253-263. The
    variance of the mean difference uses the Newey & West (1987) Bartlett
    kernel estimator:

        gamma_j = (1/n) * sum_{i=j}^{n-1} (d_i - dbar)(d_{i-j} - dbar)
        lrv     = gamma_0 + 2 * sum_{j=1}^{L} (1 - j/(L+1)) * gamma_j
        se      = sqrt(lrv / n)
        t       = (dbar - null) / se,   p = P(Z > t) = erfc(t / sqrt 2) / 2

    The p-value uses the standard normal limit, so with few pairs it is
    somewhat optimistic; callers should keep ``n`` reasonably large.

    Args:
        a, b: Equal-length sequences of finite numbers (at least 2 pairs).
            Pairs are matched by position; the caller is responsible for
            putting both in the same order.
        lags: Number of autocovariance lags ``L``. ``None`` uses
            ``floor(4 * (n/100) ** (2/9))``. Capped at ``n - 1``.
        null: The H0 boundary for the mean difference.

    If the estimated variance is exactly zero the result is deterministic:
    ``p = 0`` when ``dbar > null``, otherwise ``p = 1``.
    """
    if lags is not None and (isinstance(lags, bool) or not isinstance(lags, numbers.Integral)):
        raise TypeError(f"lags must be an int or None, got {type(lags).__name__}")
    if lags is not None and lags < 0:
        raise ValueError(f"lags must be non-negative, got {lags!r}")
    null = float(null)
    if not math.isfinite(null):
        raise ValueError(f"null must be finite, got {null!r}")
    xa = _as_float_list(a, "a")
    xb = _as_float_list(b, "b")
    if len(xa) != len(xb):
        raise ValueError(f"a and b must have the same length, got {len(xa)} and {len(xb)}")
    n = len(xa)
    if n < 2:
        raise ValueError(f"need at least 2 pairs, got {n}")

    d = [x - y for x, y in zip(xa, xb)]
    dbar = math.fsum(d) / n
    e = [x - dbar for x in d]
    L = _default_lags(n) if lags is None else int(lags)
    L = min(L, n - 1)

    lrv = math.fsum(x * x for x in e) / n
    for j in range(1, L + 1):
        gamma_j = math.fsum(e[i] * e[i - j] for i in range(j, n)) / n
        lrv += 2.0 * (1.0 - j / (L + 1.0)) * gamma_j
    # The Bartlett-weighted estimator is non-negative in exact arithmetic.
    lrv = max(lrv, 0.0)
    se = math.sqrt(lrv / n)

    num = dbar - null
    if se == 0.0:
        if num > 0:
            t, p = math.inf, 0.0
        elif num < 0:
            t, p = -math.inf, 1.0
        else:
            t, p = math.nan, 1.0
    else:
        t = num / se
        p = 0.5 * math.erfc(t / math.sqrt(2.0))
    return PairedTest(mean_diff=dbar, se=se, t=t, p=p, n=n, lags=L)


def spend_alpha(total: float, k: int) -> float:
    """Significance level for the ``k``-th of an open-ended series of tests.

    ``alpha_k = total / (k * (k + 1))``. Because
    ``sum_{k=1}^{K} 1/(k(k+1)) = 1 - 1/(K+1) < 1``, the levels of all tests
    ever run sum to less than ``total`` no matter how many are eventually run,
    so the probability of at least one false positive across the whole series
    stays below ``total`` (a union bound; no independence assumption).

    The guarantee only holds if ``k`` honestly counts every test in the
    series, including abandoned ones. This function cannot check that.
    """
    if isinstance(k, bool) or not isinstance(k, numbers.Integral):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")
    total = float(total)
    if not (0.0 < total < 1.0):
        raise ValueError(f"total must be in (0, 1), got {total!r}")
    return total / (k * (k + 1))


_MODES = ("superiority", "non_inferiority")


class PairedComparator:
    """Promote on a one-sided paired significance test, not on point scores.

    Uses :attr:`MetricResult.samples`, which must hold per-observation (or
    per-period) values for both models, in the same order. The test is the
    Diebold-Mariano test (see :func:`paired_test`) on the per-sample
    *improvement* of the challenger over the champion: ``challenger -
    champion`` when higher is better, ``champion - challenger`` when lower is
    better. It tests the mean of the samples, which is not necessarily the
    same number as ``MetricResult.value``.

    Modes:
        ``superiority``: promote iff H0 "mean improvement <= margin" is
            rejected at ``alpha`` (reason ``significant_improvement``,
            otherwise ``not_significant``).
        ``non_inferiority``: promote iff H0 "mean improvement <= -margin"
            (i.e. the challenger is worse by at least ``margin``) is rejected
            at ``alpha`` (reason ``non_inferior``, otherwise ``inferior``,
            which means "non-inferiority was not demonstrated").

    Rejection is strict: promote only if ``p < alpha``. Missing samples or
    fewer than ``min_samples`` pairs reject with ``insufficient_samples``;
    different lengths reject with ``samples_mismatch``; a non-finite or
    non-numeric sample rejects with ``invalid_metric``.

    This comparator corrects for sampling noise in one comparison. It does
    not know how many comparisons the caller has run; pass a level from
    :func:`spend_alpha` if this is one of a series.
    """

    name = "paired_dm"

    def __init__(
        self,
        alpha: float = 0.05,
        mode: str = "superiority",
        margin: float = 0.0,
        hac_lags: Optional[int] = None,
        min_samples: int = 30,
    ) -> None:
        alpha = float(alpha)
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        margin = float(margin)
        if not math.isfinite(margin) or margin < 0:
            raise ValueError(f"margin must be a finite, non-negative number, got {margin!r}")
        if hac_lags is not None:
            if isinstance(hac_lags, bool) or not isinstance(hac_lags, numbers.Integral):
                raise ValueError(f"hac_lags must be a non-negative int or None, got {hac_lags!r}")
            if hac_lags < 0:
                raise ValueError(f"hac_lags must be non-negative, got {hac_lags!r}")
        if isinstance(min_samples, bool) or not isinstance(min_samples, numbers.Integral) or min_samples < 2:
            raise ValueError(f"min_samples must be an int >= 2, got {min_samples!r}")
        self.alpha = alpha
        self.mode = mode
        self.margin = margin
        self.hac_lags = None if hac_lags is None else int(hac_lags)
        self.min_samples = int(min_samples)

    def params(self) -> Mapping[str, Any]:
        return {
            "alpha": self.alpha,
            "mode": self.mode,
            "margin": self.margin,
            "hac_lags": self.hac_lags,
            "min_samples": self.min_samples,
            "test": "diebold_mariano_newey_west_one_sided",
        }

    def compare(
        self, champion: MetricResult, challenger: MetricResult, *, higher_is_better: bool
    ) -> Verdict:
        base = {"alpha": self.alpha, "margin": self.margin}
        ch_s, ca_s = challenger.samples, champion.samples
        if ch_s is None or ca_s is None:
            missing = [lbl for lbl, s in (("challenger", ch_s), ("champion", ca_s)) if s is None]
            return Verdict(
                False, INSUFFICIENT_SAMPLES,
                f"paired test needs per-sample values; samples missing for {', '.join(missing)}",
                base,
            )
        ch_s, ca_s = list(ch_s), list(ca_s)
        if len(ch_s) != len(ca_s):
            return Verdict(
                False, SAMPLES_MISMATCH,
                f"challenger has {len(ch_s)} samples but champion has {len(ca_s)}; a paired "
                "test needs both scored on the same observations in the same order",
                {**base, "n_challenger": len(ch_s), "n_champion": len(ca_s)},
            )
        try:
            ch_f = _as_float_list(ch_s, "challenger.samples")
            ca_f = _as_float_list(ca_s, "champion.samples")
        except ValueError as exc:
            return Verdict(False, INVALID_METRIC, f"invalid sample: {exc}", base)
        n = len(ch_f)
        if n < self.min_samples:
            return Verdict(
                False, INSUFFICIENT_SAMPLES,
                f"only {n} paired samples; at least {self.min_samples} are required",
                {**base, "n": n},
            )

        # Orient so that a positive difference always means "challenger better".
        a, b = (ch_f, ca_f) if higher_is_better else (ca_f, ch_f)
        null = self.margin if self.mode == "superiority" else -self.margin
        res = paired_test(a, b, self.hac_lags, null=null)
        stats = {
            **base,
            "mean_diff": res.mean_diff,
            "se": res.se,
            "t": res.t,
            "p": res.p,
            "n": res.n,
            "lags": res.lags,
            "null": null,
            "champion": float(champion.value),
            "challenger": float(challenger.value),
        }
        passed = res.p < self.alpha
        direction = "higher" if higher_is_better else "lower"
        summary = (
            f"mean improvement {res.mean_diff!r} ({direction} is better) over {n} paired "
            f"samples, HAC se {res.se!r} ({res.lags} lags), t {res.t!r}, one-sided p {res.p!r}"
        )
        if self.mode == "superiority":
            if passed:
                return Verdict(
                    True, SIGNIFICANT_IMPROVEMENT,
                    f"{summary} < alpha {self.alpha!r}: improvement greater than margin "
                    f"{self.margin!r} is statistically significant",
                    stats,
                )
            return Verdict(
                False, NOT_SIGNIFICANT,
                f"{summary} >= alpha {self.alpha!r}: cannot conclude the challenger improves "
                f"on the champion by more than margin {self.margin!r}; incumbent retained",
                stats,
            )
        if passed:
            return Verdict(
                True, NON_INFERIOR,
                f"{summary} < alpha {self.alpha!r}: the challenger is not worse than the "
                f"champion by margin {self.margin!r} or more",
                stats,
            )
        return Verdict(
            False, INFERIOR,
            f"{summary} >= alpha {self.alpha!r}: non-inferiority not demonstrated (cannot rule "
            f"out the challenger being worse by margin {self.margin!r} or more); incumbent retained",
            stats,
        )

    def __repr__(self) -> str:
        return (
            f"PairedComparator(alpha={self.alpha!r}, mode={self.mode!r}, margin={self.margin!r}, "
            f"hac_lags={self.hac_lags!r}, min_samples={self.min_samples!r})"
        )
