"""Seeded valuation families and exact certification helpers shared by the tests."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Dict

from cake_cutting import bounds
from cake_cutting import valuations as V
from cake_cutting.oracle import RobertsonWebbOracle
from cake_cutting.pieces import OrderedPiece

# name -> (description, generator(rng, i, vals_so_far, n))
_HOT = [Fraction(0), Fraction(2, 5), Fraction(3, 5), Fraction(1)]


def _random(rng, i, vals, n):
    return V.random_valuation(rng)


def _many_segments(rng, i, vals, n):
    return V.random_valuation(rng, segments=40, grid=400, max_weight=5)


def _identical(rng, i, vals, n):
    return vals[0] if i > 0 else V.random_valuation(rng)


def _pairs(rng, i, vals, n):
    return vals[i - 1] if i % 2 == 1 else V.random_valuation(rng, segments=4, grid=12, max_weight=3)


def _types3(rng, i, vals, n):
    return vals[i % 3] if i >= 3 else V.random_valuation(rng, segments=5, grid=30, max_weight=3)


def _perturbed(rng, i, vals, n):
    return V.perturbed(vals[0], rng, Fraction(1, 10 ** 6)) if i > 0 else V.random_valuation(rng, segments=5, grid=30, max_weight=3)


def _zero_regions(rng, i, vals, n):
    return V.random_valuation(rng, segments=30, grid=300, max_weight=3, zero_prob=0.5)


def _blocky(rng, i, vals, n):
    return V.blocky(rng, segments=6, grid=12)


def _uniform(rng, i, vals, n):
    return V.uniform()


def _concentrated(rng, i, vals, n):
    return V.concentrated(rng, grid=50, width=rng.randint(1, 5))


def _concentrated_pairs(rng, i, vals, n):
    return vals[i - 1] if i % 2 == 1 else V.concentrated(rng, grid=50, width=rng.randint(1, 5))


def _concentrated_perturbed(rng, i, vals, n):
    if i % 2 == 0:
        return V.concentrated(rng, grid=50, width=rng.randint(1, 5))
    return V.perturbed(vals[i - 1], rng, Fraction(1, 10 ** 6), grid=50)


def _hotspot(rng, i, vals, n):
    return V.PiecewiseConstantValuation(_HOT, [rng.randint(0, 2), rng.randint(20, 40), rng.randint(0, 2)])


def _nearly_identical_pairs(rng, i, vals, n):
    """Pairs: a valuation concentrated on a few cells, and a copy with a bump of relative
    size 10^-200 on one sub-cell *inside* the support.  Differences this small survive
    pre-processing as positive *small* bonuses, so Stage 3 has to cut non-empty additions
    (Lemma 4.14 (i)); exact arithmetic makes this regime testable."""
    if i % 2 == 0:
        width = rng.randint(2, 5)
        start = rng.randrange(0, 50 - width + 1)
        lo, hi = Fraction(start, 50), Fraction(start + width, 50)
        xs = sorted({Fraction(0), lo, hi, Fraction(1)})
        ds = [rng.randint(1, 3) if u == lo else 0 for u, v in zip(xs, xs[1:])]
        return V.PiecewiseConstantValuation(xs, ds)
    base = vals[i - 1]
    support = [k for k, d in enumerate(base.densities) if d > 0][0]
    lo, hi = base.breakpoints[support], base.breakpoints[support + 1]
    cell = rng.randrange(int(lo * 500), int(hi * 500))
    blo, bhi = Fraction(cell, 500), Fraction(cell + 1, 500)
    xs = sorted(set(base.breakpoints) | {blo, bhi})
    ds = []
    for u, v in zip(xs, xs[1:]):
        d = base.densities[max(k for k, x in enumerate(base.breakpoints) if x <= u)]
        if blo <= u and v <= bhi:
            d = d + d / Fraction(10) ** 200
        ds.append(d)
    return V.PiecewiseConstantValuation(xs, ds)


FAMILIES = {
    "random": _random,
    "many_segments": _many_segments,
    "identical": _identical,
    "pairs": _pairs,
    "types3": _types3,
    "perturbed": _perturbed,
    "zero_regions": _zero_regions,
    "blocky": _blocky,
    "uniform": _uniform,
    "concentrated": _concentrated,
    "concentrated_pairs": _concentrated_pairs,
    "concentrated_perturbed": _concentrated_perturbed,
    "hotspot": _hotspot,
    "nearly_identical_pairs": _nearly_identical_pairs,
}

#: families whose residue tends to survive pre-processing (Stages 2-3 are exercised)
DEEP_FAMILIES = ("concentrated", "concentrated_pairs", "concentrated_perturbed", "nearly_identical_pairs")


def make_valuations(family: str, n: int, seed: int) -> Dict[int, V.PiecewiseConstantValuation]:
    rng = random.Random(seed * 101 + n)
    gen = FAMILIES[family]
    vals: Dict[int, V.PiecewiseConstantValuation] = {}
    for i in range(n):
        vals[i] = gen(rng, i, vals, n)
    return vals


def certify(oracle: RobertsonWebbOracle, allocation, agents, cake=None) -> None:
    """Exact certificate: complete, envy-free (true valuations), well-formed, within the bound."""
    cake = OrderedPiece.cake() if cake is None else cake
    assert allocation.is_complete(cake), "allocation is not a partition of the cake"
    for i in agents:
        own = oracle.true_value(i, allocation[i])
        for j in agents:
            other = oracle.true_value(i, allocation[j])
            assert own >= other, f"agent {i} envies agent {j}: {own} < {other}"
    n = len(agents)
    assert oracle.queries <= bounds.query_bound(n), "query count exceeds the proven bound"
    if oracle.log is not None:
        for entry in oracle.log:
            if entry[0] == "eval":
                _, agent, x, y, v = entry
                assert 0 <= x <= y <= 1 and v >= 0
            else:
                _, agent, x, a, y = entry
                assert 0 <= x <= y <= 1 and a >= 0
