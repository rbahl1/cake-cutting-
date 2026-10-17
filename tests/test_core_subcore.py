"""Core and SubCore guarantees (Propositions 4.3-4.5, Appendix D) on many instances,
including the Bechler–Bitikofer counterexample to the original SubCore."""

import random
from fractions import Fraction

import pytest

from cake_cutting import bounds
from cake_cutting import valuations as V
from cake_cutting.bookkeeping import ValueTable
from cake_cutting.oracle import RobertsonWebbOracle
from cake_cutting.pieces import OrderedPiece, pairwise_disjoint, union_measure
from cake_cutting import equilibrium
from cake_cutting.subcore import core, subcore

from .families import FAMILIES, make_valuations

F = Fraction


def bechler_bitikofer():
    """Five agents, 13 equal cells: the instance on which AM's original SubCore fails."""
    xs = [F(k, 13) for k in range(14)]
    rows = [(1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 6, 6), (4, 4, 2, 1, 1, 8, 0, 0, 0, 0, 0, 8, 0),
            (1, 1, 1, 1, 1, 1, 1, 1, 3, 0, 0, 4, 0), (1, 1, 1, 1, 2, 2, 4, 2, 4, 2, 4, 4, 0),
            (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 3, 0)]
    return {i: V.PiecewiseConstantValuation(xs, r) for i, r in enumerate(rows)}


def check_core(oracle, table, agents, R, cutter, alloc, R2, trims, queries_used):
    """Proposition 4.3 (i)-(ii) and the query count of Proposition 4.4 (explicit form)."""
    n = len(agents)
    pieces = [alloc[a] for a in agents]
    assert pairwise_disjoint(pieces + [R2])
    assert union_measure(pieces + [R2]) == R.normalized_length()
    for i in agents:
        vi = oracle.true_value(i, alloc[i])
        for j in agents:
            assert vi >= oracle.true_value(i, alloc[j]), f"envy {i} -> {j}"
    vc = oracle.true_value(cutter, R)
    whole = [a for a in agents if oracle.true_value(cutter, alloc[a]) == vc / n]
    assert cutter in whole and len(whole) >= 2, "the cutter and another agent must receive whole pieces"
    assert oracle.true_value(cutter, R2) <= F(n - 2, n) * vc
    assert trims <= bounds.subcore_cuts(n - 1, n)
    assert queries_used <= bounds.core_queries(n)


def run_sequence(vals, calls, cut_rule, cutter_of):
    oracle = RobertsonWebbOracle(vals, cut_rule=cut_rule)
    agents = sorted(vals)
    table = ValueTable(oracle, agents)
    table.register_cake()
    R = OrderedPiece.cake()
    for t in range(calls):
        cutter = cutter_of(t, len(agents))
        q0 = oracle.queries
        alloc, R2, trims = core(table, agents, R, cutter)
        check_core(oracle, table, agents, R, cutter, alloc, R2, trims, oracle.queries - q0)
        R = R2
        if R.is_empty:
            break
    return oracle


@pytest.mark.parametrize("cut_rule", ["left", "right"])
@pytest.mark.parametrize("cutter", range(5))
def test_bechler_bitikofer_instance(cut_rule, cutter):
    run_sequence(bechler_bitikofer(), 40, cut_rule, lambda t, n: cutter)


@pytest.mark.parametrize("cut_rule", ["left", "right"])
@pytest.mark.parametrize("family", ["random", "blocky", "pairs", "zero_regions", "uniform",
                                    "concentrated", "types3", "perturbed", "hotspot"])
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_core_guarantees_on_families(n, family, cut_rule):
    for seed in range(3):
        vals = make_valuations(family, n, seed)
        run_sequence(vals, 25, cut_rule, lambda t, n: (t // 3) % n)


def test_prescribed_sequence_reduces_residue_value():
    """Proposition 4.3 (iii): ceil(log_{n/(n-2)} beta) calls reduce V_c(R) by the factor beta."""
    n = 5
    for seed in range(4):
        vals = make_valuations("concentrated", n, seed)
        oracle = RobertsonWebbOracle(vals)
        agents = sorted(vals)
        table = ValueTable(oracle, agents)
        table.register_cake()
        R = OrderedPiece.cake()
        cutter = 1
        beta = F(1000)
        start = table.value(cutter, R)
        for _ in range(bounds.ceil_log(n, beta)):
            if table.value(cutter, R) == 0:
                break
            _, R, _ = core(table, agents, R, cutter)
        assert table.value(cutter, R) <= start / beta
        assert oracle.true_value(cutter, R) == table.value(cutter, R)


def test_direct_subcore_call_proposition_4_5():
    """m agents, m ordered pieces W_h with V_i(W_h) >= q_i and one empty piece placed last:
    every agent gets a suffix of a distinct W_h worth >= q_i, the empty piece stays unallocated."""
    rng = random.Random(7)
    for trial in range(40):
        m = rng.randint(1, 5)
        n_total = m + rng.randint(0, 2)
        vals = {i: V.random_valuation(rng, segments=6, grid=40) for i in range(n_total)}
        oracle = RobertsonWebbOracle(vals, cut_rule=rng.choice(["left", "right"]))
        agents = list(range(n_total))
        table = ValueTable(oracle, agents)
        table.register_cake()
        # m disjoint ordered pieces from random cut points (values known via split_at)
        cuts = sorted({F(rng.randint(1, 39), 40) for _ in range(m - 1)})
        comps = []
        cur = OrderedPiece.cake().components[0]
        for y in cuts:
            if cur.a < y < cur.b:
                left, cur = table.split_at(cur, y)
                comps.append(left)
        comps.append(cur)
        while len(comps) < m:  # pathological duplicates: split the last component again
            c = comps.pop()
            left, right = table.split_at(c, (c.a + c.b) / 2)
            comps.extend([left, right])
        order = list(range(m))
        rng.shuffle(order)
        W = [OrderedPiece([comps[k]]) for k in order]
        E = list(range(m))
        q = {i: min(table.value(i, w) for w in W) * F(rng.randint(0, 10), 10) for i in E}
        pieces = W + [OrderedPiece.empty()]
        q0 = oracle.queries
        result, trims = subcore(table, pieces, E, q)
        assert trims <= bounds.subcore_cuts(m, m)
        assert oracle.queries - q0 <= bounds.subcore_queries(m, n_total)
        used = set()
        for i in E:
            idx, suffix = result[i]
            assert idx < m, "the empty piece was allocated"
            assert idx not in used
            used.add(idx)
            assert oracle.true_value(i, suffix) >= q[i]
            for j in E:
                assert oracle.true_value(i, suffix) >= oracle.true_value(i, result[j][1])
        assert len(used) == m


# ------------------------------------------------- the equilibrium fallback ----
def _fallback_instance(rng, zero):
    """``r`` agents with piecewise-constant valuations on a grid (many exact ties) and
    ``k`` nonempty ordered pieces made of grid cells (single intervals or shuffled
    unions); a Core-type call (``k > r``, minima 0) or a stage-3-type call (``r``
    pieces, one empty piece, minima up to the smallest whole value)."""
    r = rng.choice([1, 2, 3, 4, 4, 5])
    k = r + 1 if rng.random() < 0.8 else r + 2
    cells = rng.choice([1, 2, 3])
    grid = k * cells
    xs = [F(t, grid) for t in range(grid + 1)]
    lo = 0 if zero else 1
    vals = {}
    for i in range(r):
        if vals and rng.random() < 0.35:
            vals[i] = vals[rng.randrange(i)]
            continue
        ds = [rng.choice([lo, 1, 2, 3]) for _ in range(grid)]
        if sum(ds) == 0:
            ds[0] = 1
        vals[i] = V.PiecewiseConstantValuation(xs, ds)
    oracle = RobertsonWebbOracle(vals, cut_rule=rng.choice(["left", "right"]))
    agents = list(range(r))
    table = ValueTable(oracle, agents)
    table.register_cake()
    rest = OrderedPiece.cake().components[0]
    ivs = []
    for t in range(1, grid):
        left, rest = table.split_at(rest, F(t, grid))
        ivs.append(left)
    ivs.append(rest)
    if rng.random() < 0.5:
        groups = [ivs[p * cells:(p + 1) * cells] for p in range(k)]
    else:
        order = list(range(grid))
        rng.shuffle(order)
        groups = [[ivs[c] for c in order[p::k]] for p in range(k)]
    pieces = [OrderedPiece.concat([OrderedPiece([iv]) for iv in g]) for g in groups]
    direct = rng.random() < 0.35
    if direct:
        pieces = pieces[:r] + [OrderedPiece.empty()]
        q = {i: min(table.value(i, p) for p in pieces[:r]) * F(rng.randint(0, 4), 4) for i in agents}
    else:
        q = {i: F(0) for i in agents}
    rng.shuffle(agents)
    return oracle, table, pieces, agents, q, direct


def test_equilibrium_fallback_positive_density():
    """Theorem D (CORRECTNESS.md §8): with positive densities the fallback always returns
    suffixes of distinct pieces with the minima, envy-free, no agent preferring a free
    piece, a whole piece in Core-type calls, no empty piece in stage-3-type calls, and at
    most E(m, k) cuts."""
    rng = random.Random(2026)
    for trial in range(300):
        oracle, table, pieces, agents, q, direct = _fallback_instance(rng, zero=False)
        r = len(agents)
        k = sum(1 for p in pieces if len(p) > 0)
        c0 = oracle.cuts
        res, calls = equilibrium.allocate(table, pieces, agents, q)
        assert res is not None, trial
        assert calls <= bounds.E(r, k) and oracle.cuts - c0 <= bounds.E(r, k)
        held = [idx for idx, _ in res.values()]
        assert len(set(held)) == r
        for i in agents:
            vi = oracle.true_value(i, res[i][1])
            assert vi >= q[i]
            for j in agents:
                assert vi >= oracle.true_value(i, res[j][1])
            for u, p in enumerate(pieces):
                if u not in held:
                    assert vi >= oracle.true_value(i, p)
        if direct:
            assert all(idx < r for idx in held), "the empty piece was allocated"
        else:
            assert any(len(s) > 0 and len(s) == len(pieces[idx]) and s.first == pieces[idx].first
                       for idx, s in res.values()), "no whole piece allocated"


def test_equilibrium_fallback_zero_density_is_checked():
    """With zero-density stretches the fallback is not proved; whatever it returns
    still has SubCore's properties (it returns None otherwise)."""
    rng = random.Random(77)
    returned = 0
    for trial in range(150):
        oracle, table, pieces, agents, q, direct = _fallback_instance(rng, zero=True)
        res, _ = equilibrium.allocate(table, pieces, agents, q)
        if res is None:
            continue
        returned += 1
        held = [idx for idx, _ in res.values()]
        for i in agents:
            vi = oracle.true_value(i, res[i][1])
            assert vi >= q[i]
            assert all(vi >= oracle.true_value(i, res[j][1]) for j in agents)
            assert all(vi >= oracle.true_value(i, p) for u, p in enumerate(pieces) if u not in held)
    assert returned >= 140


def test_cut_count_of_the_fallback():
    """E(m, k) = (m-1) k C(m+k-3, m-2) is the number of (buyer, good) pairs over the high
    states (I - i0 - A, J - B), |A| = |B|, and e -> E(e, e) is superadditive."""
    from math import comb
    for m in range(1, 9):
        for k in range(1, 10):
            direct = sum(comb(m - 1, t) * comb(k, t) * (m - 1 - t) * (k - t) for t in range(m))
            assert bounds.E(m, k) == direct
    for a in range(0, 12):
        for b in range(0, 12):
            assert bounds.E(a, a) + bounds.E(b, b) <= bounds.E(a + b, a + b)
