"""Lemma 4.6 / Algorithm 4: cutting several pieces at once."""

import random
from fractions import Fraction

import pytest

from cake_cutting import bounds
from cake_cutting import valuations as V
from cake_cutting.bookkeeping import ValueTable
from cake_cutting.oracle import RobertsonWebbOracle
from cake_cutting.pieces import OrderedPiece, pairwise_disjoint, union_measure
from cake_cutting.simultaneous import SimultaneousCutError, simultaneous_cut

F = Fraction


def residue_with_known_values(table, rng, parts):
    cuts = sorted({F(rng.randint(1, 99), 100) for _ in range(parts)})
    comps = []
    cur = OrderedPiece.cake().components[0]
    for y in cuts:
        if cur.a < y < cur.b:
            left, cur = table.split_at(cur, y)
            comps.append(left)
    comps.append(cur)
    rng.shuffle(comps)
    return OrderedPiece(comps[: max(1, len(comps) - 1)])


def check_output(oracle, agents, R, requests, delta, pieces, leftover):
    total = {i: oracle.true_value(i, R) for i in agents}
    for k, lam in requests.items():
        for i in agents:
            assert abs(oracle.true_value(i, pieces[k]) - lam * total[i]) <= delta * total[i]
    allp = list(pieces.values()) + [leftover]
    assert pairwise_disjoint(allp)
    assert union_measure(allp) == R.normalized_length()
    # every component of every output piece lies inside R
    assert union_measure(allp + [R]) == R.normalized_length()


@pytest.mark.parametrize("seed", range(12))
def test_rounding_meets_the_lemma_exactly(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 6)
    vals = {i: rng.choice([V.random_valuation(rng, segments=8, grid=80, zero_prob=0.3),
                           V.concentrated(rng, grid=100, width=rng.randint(1, 3)),
                           V.blocky(rng)]) for i in range(n)}
    oracle = RobertsonWebbOracle(vals, cut_rule=rng.choice(["left", "right"]))
    agents = list(range(n))
    table = ValueTable(oracle, agents)
    table.register_cake()
    R = residue_with_known_values(table, rng, rng.randint(1, 6))
    m = rng.randint(1, 8)
    delta = F(1, rng.choice([50, 1000, 10 ** 5]))
    lam = [F(rng.randint(0, 40), 200) for _ in range(m)]
    while sum(lam) > 1:
        lam = [x / 2 for x in lam]
    requests = {("a", j): lam[j] for j in range(m)}
    requests[("tiny",)] = delta / 2  # may be returned empty
    budget = bounds.lemma46_queries(n, len(requests), delta)
    q0 = oracle.queries
    pieces, leftover, cutter = simultaneous_cut(table, agents, R, requests, delta, budget=budget)
    assert cutter.phase_used == "rounding"
    assert oracle.queries - q0 <= budget
    check_output(oracle, agents, R, requests, delta, pieces, leftover)
    # output values are known in the table (no further queries needed by the caller)
    for p in pieces.values():
        for i in agents:
            assert table.value(i, p) == oracle.true_value(i, p)


def test_rounding_with_nobody_valuing_the_residue():
    vals = {0: V.PiecewiseConstantValuation([0, F(1, 2), 1], [1, 0]),
            1: V.PiecewiseConstantValuation([0, F(1, 2), 1], [2, 0])}
    oracle = RobertsonWebbOracle(vals)
    table = ValueTable(oracle, [0, 1])
    table.register_cake()
    left, right = table.split_at(OrderedPiece.cake().components[0], F(1, 2))
    R = OrderedPiece([right])
    pieces, leftover, cutter = simultaneous_cut(table, [0, 1], R, {"p": F(1, 3)}, F(1, 10))
    check_output(oracle, [0, 1], R, {"p": F(1, 3)}, F(1, 10), pieces, leftover)


@pytest.mark.parametrize("seed", range(6))
def test_paper_procedure_on_tiny_inputs(seed):
    """Procedure ``paper`` (the literal procedure of the proof) with a coarse gamma: value-based cells,
    greedy / exhaustive labeling, exact verification."""
    rng = random.Random(100 + seed)
    n = rng.randint(1, 3)
    vals = {i: V.random_valuation(rng, segments=4, grid=20) for i in range(n)}
    oracle = RobertsonWebbOracle(vals)
    agents = list(range(n))
    table = ValueTable(oracle, agents)
    table.register_cake()
    R = residue_with_known_values(table, rng, rng.randint(1, 3))
    requests = {"a": F(1, 4), "b": F(1, 3)}
    delta = F(1, 4)
    pieces, leftover, cutter = simultaneous_cut(table, agents, R, requests, delta,
                                                force_fallback=True, gamma=F(1, 12))
    assert cutter.phase_used == "paper"
    check_output(oracle, agents, R, requests, delta, pieces, leftover)
    assert oracle.cuts <= n * 12


def test_paper_procedure_refuses_infeasible_resolution():
    vals = {0: V.uniform(), 1: V.random_valuation(random.Random(1))}
    oracle = RobertsonWebbOracle(vals)
    table = ValueTable(oracle, [0, 1])
    table.register_cake()
    with pytest.raises(SimultaneousCutError):
        simultaneous_cut(table, [0, 1], OrderedPiece.cake(), {"a": F(1, 2)}, bounds.stage3_delta(4),
                         force_fallback=True)


def test_rejects_bad_requests():
    vals = {0: V.uniform()}
    oracle = RobertsonWebbOracle(vals)
    table = ValueTable(oracle, [0])
    table.register_cake()
    with pytest.raises(ValueError):
        simultaneous_cut(table, [0], OrderedPiece.cake(), {"a": F(3, 4), "b": F(1, 2)}, F(1, 10))
    with pytest.raises(ValueError):
        simultaneous_cut(table, [0], OrderedPiece.cake(), {"a": F(1, 4)}, F(1))
