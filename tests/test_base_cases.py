"""Base cases of Main (Section 4.3): n = 1, 2, 3 on ordered subcakes."""

import random
from fractions import Fraction

import pytest

from cake_cutting import bounds
from cake_cutting import valuations as V
from cake_cutting.base_cases import base_case
from cake_cutting.bookkeeping import ValueTable
from cake_cutting.oracle import RobertsonWebbOracle
from cake_cutting.pieces import OrderedPiece, pairwise_disjoint, union_measure

from .families import FAMILIES, make_valuations

F = Fraction


def random_subcake(table, rng, parts):
    """A subcake with ``parts`` components, in a shuffled order, with known values."""
    cuts = sorted({F(rng.randint(1, 59), 60) for _ in range(parts - 1)})
    comps = []
    cur = OrderedPiece.cake().components[0]
    for y in cuts:
        if cur.a < y < cur.b:
            left, cur = table.split_at(cur, y)
            comps.append(left)
    comps.append(cur)
    rng.shuffle(comps)
    keep = comps[: max(1, len(comps) - 1)]  # drop one component so the subcake is proper
    return OrderedPiece(keep)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("family", list(FAMILIES))
def test_base_cases_are_envy_free_complete_and_bounded(n, family):
    for seed in range(6):
        vals = make_valuations(family, n, seed)
        for cut_rule in ("left", "right"):
            oracle = RobertsonWebbOracle(vals, cut_rule=cut_rule)
            agents = sorted(vals)
            table = ValueTable(oracle, agents)
            table.register_cake()
            rng = random.Random(seed)
            cake = random_subcake(table, rng, rng.randint(1, 4))
            q0 = oracle.queries
            alloc = base_case(table, agents, cake)
            assert oracle.queries - q0 <= bounds.BASE_CASE_QUERIES[n]
            pieces = [alloc[a] for a in agents]
            assert pairwise_disjoint(pieces)
            assert union_measure(pieces) == cake.normalized_length()
            for i in agents:
                for j in agents:
                    assert oracle.true_value(i, alloc[i]) >= oracle.true_value(i, alloc[j])


def test_selfridge_conway_handles_zero_valued_divider():
    vals = {0: V.uniform(), 1: V.PiecewiseConstantValuation([0, F(1, 3), 1], [3, 0]),
            2: V.PiecewiseConstantValuation([0, F(1, 3), F(2, 3), 1], [0, 3, 0])}
    oracle = RobertsonWebbOracle(vals)
    table = ValueTable(oracle, [0, 1, 2])
    table.register_cake()
    alloc = base_case(table, [0, 1, 2], OrderedPiece.cake())
    assert union_measure(list(alloc.values())) == 1
    for i in alloc:
        for j in alloc:
            assert oracle.true_value(i, alloc[i]) >= oracle.true_value(i, alloc[j])
    assert oracle.queries <= 3 + 15
