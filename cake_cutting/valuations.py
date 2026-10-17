"""Valuations used by the tests and the demo (never read by the protocol).

The protocol only sees valuations through :class:`cake_cutting.oracle.RobertsonWebbOracle`.
This module provides nonnegative, additive, nonatomic valuations with a
piecewise-constant density, rational breakpoints and rational weights, so that
every Eval answer and every Cut answer is an exact rational number.

(Piecewise-*linear* densities are deliberately not provided: a Cut query would
then have to return the root of a quadratic, which is irrational in general and
cannot be represented exactly.)
"""

from __future__ import annotations

import bisect
import random
from fractions import Fraction
from typing import List, Sequence

from .pieces import OrderedPiece, ZERO, ONE


class PiecewiseConstantValuation:
    """Additive valuation with density ``d_k`` on ``[x_{k-1}, x_k)``.

    ``breakpoints`` must start at 0 and end at 1 and be strictly increasing;
    ``densities`` are nonnegative rationals (one per segment).
    """

    def __init__(self, breakpoints: Sequence, densities: Sequence) -> None:
        xs = [Fraction(x) for x in breakpoints]
        ds = [Fraction(d) for d in densities]
        if len(xs) != len(ds) + 1:
            raise ValueError("need one density per segment")
        if xs[0] != 0 or xs[-1] != 1:
            raise ValueError("breakpoints must start at 0 and end at 1")
        for u, v in zip(xs, xs[1:]):
            if not u < v:
                raise ValueError("breakpoints must be strictly increasing")
        for d in ds:
            if d < 0:
                raise ValueError("densities must be nonnegative")
        self.breakpoints = xs
        self.densities = ds
        cum = [ZERO]
        for (u, v), d in zip(zip(xs, xs[1:]), ds):
            cum.append(cum[-1] + d * (v - u))
        self._cum = cum
        self._float_bps = [float(x) for x in xs]

    def _segment(self, t: Fraction) -> int:
        """Index ``k`` with ``x_k <= t < x_{k+1}`` (``0 < t < 1``): a float bisection
        followed by exact adjustment, which avoids most exact comparisons of the
        huge rationals produced by the protocol."""
        xs = self.breakpoints
        k = bisect.bisect_right(self._float_bps, float(t)) - 1
        k = max(0, min(k, len(self.densities) - 1))
        while k + 1 < len(self.densities) and t >= xs[k + 1]:
            k += 1
        while k > 0 and t < xs[k]:
            k -= 1
        return k

    # -- evaluation -------------------------------------------------------
    @property
    def total(self) -> Fraction:
        return self._cum[-1]

    def F(self, t: Fraction) -> Fraction:
        """Cumulative value ``V([0, t])``."""
        if t <= 0:
            return ZERO
        if t >= 1:
            return self._cum[-1]
        k = self._segment(t)  # segment index with x_k <= t < x_{k+1}
        return self._cum[k] + self.densities[k] * (t - self.breakpoints[k])

    def value(self, x: Fraction, y: Fraction) -> Fraction:
        """``V([x, y])`` for ``0 <= x <= y <= 1``."""
        if not (0 <= x <= y <= 1):
            raise ValueError(f"invalid interval [{x}, {y}]")
        return self.F(y) - self.F(x)

    def value_of(self, piece: OrderedPiece) -> Fraction:
        return sum((self.value(c.a, c.b) for c in piece.components), ZERO)

    # -- cut queries ------------------------------------------------------
    def cut(self, x: Fraction, a: Fraction, rule: str = "left") -> Fraction:
        """Return ``y >= x`` with ``V([x, y]) = a`` (requires ``0 <= a <= V([x, 1])``).

        With ``rule == "left"`` the smallest such ``y`` is returned, with
        ``rule == "right"`` the largest.  Both are legitimate Robertson–Webb
        answers; the protocol must work with either.
        """
        if not (0 <= x <= 1):
            raise ValueError(f"cut start {x} outside the cake")
        target = self.F(x) + a
        if a < 0 or target > self._cum[-1]:
            raise ValueError(f"cut value {a} not attainable from {x}")
        y = self._leftmost(x, target)
        if rule == "left":
            return y
        if rule == "right":
            return self._extend_right(y, target)
        raise ValueError(f"unknown cut rule {rule!r}")

    def _leftmost(self, x: Fraction, target: Fraction) -> Fraction:
        if self.F(x) == target:
            return x
        k = self._segment(x) if 0 < x < 1 else (0 if x <= 0 else len(self.densities) - 1)
        while k < len(self.densities):
            left = max(x, self.breakpoints[k])
            right = self.breakpoints[k + 1]
            f_right = self._cum[k + 1]
            if f_right >= target:
                f_left = self.F(left)
                if f_left == target:
                    return left
                d = self.densities[k]
                assert d > 0
                return left + (target - f_left) / d
            k += 1
        raise AssertionError("unreachable: target below total")

    def _extend_right(self, y: Fraction, target: Fraction) -> Fraction:
        # move right over zero-density stretches starting at y
        if y >= 1:
            return ONE
        k = self._segment(y) if y > 0 else 0
        while k < len(self.densities) and y < 1:
            if self.densities[k] == 0:
                y = self.breakpoints[k + 1]
                k += 1
            else:
                break
        return min(y, ONE)


# ---------------------------------------------------------------------------
# Generators for test instances (seeded).
# ---------------------------------------------------------------------------

def _grid_breakpoints(rng: random.Random, segments: int, grid: int) -> List[Fraction]:
    """``segments`` random segments with breakpoints on the grid ``k/grid``."""
    if segments < 1 or grid < segments:
        raise ValueError("need grid >= segments >= 1")
    inner = sorted(rng.sample(range(1, grid), segments - 1))
    return [ZERO] + [Fraction(k, grid) for k in inner] + [ONE]


def random_valuation(rng: random.Random, segments: int = 6, grid: int = 60,
                     max_weight: int = 9, zero_prob: float = 0.15) -> PiecewiseConstantValuation:
    """Generic random piecewise-constant valuation with small integer weights."""
    xs = _grid_breakpoints(rng, segments, grid)
    ds = []
    for _ in range(segments):
        if rng.random() < zero_prob:
            ds.append(0)
        else:
            ds.append(rng.randint(1, max_weight))
    if sum(ds) == 0:
        ds[rng.randrange(segments)] = 1
    return PiecewiseConstantValuation(xs, ds)


def perturbed(base: PiecewiseConstantValuation, rng: random.Random, scale: Fraction,
              grid: int = 60) -> PiecewiseConstantValuation:
    """``base`` plus a tiny random bump (density ``scale``) on one grid cell."""
    cell = rng.randrange(grid)
    lo, hi = Fraction(cell, grid), Fraction(cell + 1, grid)
    xs = sorted(set(base.breakpoints) | {lo, hi})
    ds = []
    for u, v in zip(xs, xs[1:]):
        d = base.densities[bisect.bisect_right(base.breakpoints, u) - 1]
        if lo <= u and v <= hi:
            d = d + scale
        ds.append(d)
    return PiecewiseConstantValuation(xs, ds)


def concentrated(rng: random.Random, grid: int = 1000, width: int = 1) -> PiecewiseConstantValuation:
    """All value inside one tiny interval of width ``width/grid``."""
    start = rng.randrange(0, grid - width + 1)
    lo, hi = Fraction(start, grid), Fraction(start + width, grid)
    xs = sorted({ZERO, lo, hi, ONE})
    ds = [1 if (u == lo) else 0 for u, v in zip(xs, xs[1:])]
    return PiecewiseConstantValuation(xs, ds)


def uniform() -> PiecewiseConstantValuation:
    return PiecewiseConstantValuation([0, 1], [1])


def blocky(rng: random.Random, segments: int = 4, grid: int = 8, weights=(0, 1, 2)) -> PiecewiseConstantValuation:
    """Few segments, very few distinct weights: makes many Core calls tie."""
    xs = _grid_breakpoints(rng, segments, grid)
    ds = [rng.choice(weights) for _ in range(segments)]
    if sum(ds) == 0:
        ds[0] = 1
    return PiecewiseConstantValuation(xs, ds)
