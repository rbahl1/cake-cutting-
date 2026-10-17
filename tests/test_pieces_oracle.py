"""Unit tests: pieces, valuations, the Robertson–Webb oracle and the value table."""

import random
from fractions import Fraction

import pytest

from cake_cutting import valuations as V
from cake_cutting.bookkeeping import ValueTable
from cake_cutting.oracle import QueryBudgetExceeded, QueryError, RobertsonWebbOracle
from cake_cutting.pieces import (Interval, OrderedPiece, difference, overlap_length,
                                 pairwise_disjoint, union_measure)

F = Fraction


# ---------------------------------------------------------------- pieces ----
def test_interval_basics():
    c = Interval(F(1, 4), F(3, 4))
    assert c.length == F(1, 2)
    left, right = c.split(F(1, 2))
    assert (left.a, left.b, right.a, right.b) == (F(1, 4), F(1, 2), F(1, 2), F(3, 4))
    with pytest.raises(ValueError):
        Interval(F(1, 2), F(1, 2))
    with pytest.raises(ValueError):
        c.split(F(3, 4))


def test_ordered_piece_order_is_kept_and_slices_follow_it():
    # the example of Appendix C.1: [P, A, D] keeps P's components first
    P = OrderedPiece([Interval(0, F(1, 10)), Interval(F(7, 10), F(8, 10))])
    A = OrderedPiece.interval(F(2, 10), F(3, 10))
    D = OrderedPiece([Interval(F(4, 10), F(5, 10)), Interval(F(9, 10), 1)])
    W = OrderedPiece.concat([P, A, D])
    assert [c.a for c in W.components] == [0, F(7, 10), F(2, 10), F(4, 10), F(9, 10)]
    assert W.prefix_at(2) == P and W.suffix_at(2) == A + D
    assert W.normalized_length() == W.length == F(5, 10)
    assert W.normalized()[0] == Interval(0, F(1, 10))


def test_measure_helpers():
    p = OrderedPiece([Interval(0, F(1, 2))])
    q = OrderedPiece([Interval(F(1, 4), F(3, 4))])
    r = OrderedPiece([Interval(F(1, 2), 1)])
    assert overlap_length(p, q) == F(1, 4)
    assert pairwise_disjoint([p, r]) and not pairwise_disjoint([p, q])
    assert union_measure([p, q, r]) == 1
    d = difference(OrderedPiece.cake(), [q])
    assert d.components == (Interval(0, F(1, 4)), Interval(F(3, 4), 1))
    assert difference(p, [p]).is_empty


# ----------------------------------------------------------- valuations ----
def test_piecewise_constant_values_and_cuts():
    v = V.PiecewiseConstantValuation([0, F(1, 2), 1], [2, 0])
    assert v.total == 1 and v.value(0, F(1, 4)) == F(1, 2) and v.value(F(1, 2), 1) == 0
    assert v.cut(0, F(1, 2)) == F(1, 4)
    # a cut for the whole value: "left" stops at the start of the zero stretch, "right" at its end
    assert v.cut(0, 1, "left") == F(1, 2)
    assert v.cut(0, 1, "right") == 1
    assert v.cut(F(3, 4), 0, "left") == F(3, 4)
    with pytest.raises(ValueError):
        v.cut(0, 2)
    with pytest.raises(ValueError):
        V.PiecewiseConstantValuation([0, 1], [-1])
    with pytest.raises(ValueError):
        V.PiecewiseConstantValuation([0, F(1, 2), F(1, 2), 1], [1, 1, 1])


@pytest.mark.parametrize("seed", range(20))
def test_generators_produce_valid_valuations(seed):
    rng = random.Random(seed)
    for v in (V.random_valuation(rng), V.blocky(rng), V.concentrated(rng), V.uniform(),
              V.perturbed(V.random_valuation(rng), rng, F(1, 10 ** 6)),
              V.random_valuation(rng, zero_prob=0.9)):
        assert v.total > 0
        for _ in range(5):
            a = F(rng.randint(0, 100), 100)
            x = F(rng.randint(0, 100), 100)
            amount = a * v.value(x, 1)
            for rule in ("left", "right"):
                y = v.cut(x, amount, rule)
                assert x <= y <= 1 and v.value(x, y) == amount


# --------------------------------------------------------------- oracle ----
def test_oracle_counts_and_validates():
    vals = {0: V.uniform(), 1: V.PiecewiseConstantValuation([0, F(1, 2), 1], [0, 2])}
    oracle = RobertsonWebbOracle(vals, log=True)
    assert oracle.eval(0, F(1, 4), F(3, 4)) == F(1, 2)
    assert oracle.cut(1, 0, F(1, 2)) == F(3, 4)
    assert (oracle.evals, oracle.cuts, oracle.queries) == (1, 1, 2)
    assert oracle.evals_by_agent == {0: 1, 1: 0} and oracle.cuts_by_agent == {0: 0, 1: 1}
    assert len(oracle.log) == 2
    with pytest.raises(QueryError):
        oracle.eval(0, F(3, 4), F(1, 4))
    with pytest.raises(QueryError):
        oracle.eval(2, 0, 1)
    with pytest.raises(QueryError):
        oracle.cut(0, F(1, 2), 1)  # more than the remaining value
    with pytest.raises(QueryError):
        oracle.cut(0, 0, -1)
    assert oracle.queries == 2  # ill-formed queries are not charged
    budgeted = RobertsonWebbOracle(vals, budget=1)
    budgeted.eval(0, 0, 1)
    with pytest.raises(QueryBudgetExceeded):
        budgeted.eval(0, 0, 1)


# ---------------------------------------------------------- value table ----
def test_prefix_of_value_follows_appendix_c():
    vals = {0: V.uniform(), 1: V.PiecewiseConstantValuation([0, F(1, 2), 1], [2, 0]), 2: V.uniform()}
    oracle = RobertsonWebbOracle(vals)
    table = ValueTable(oracle, [0, 1, 2])
    table.register_cake()
    assert oracle.queries == 3
    piece = OrderedPiece.cake()
    # a prefix of value 1/4 for agent 0 costs one Cut and one Eval per other agent (Prop. C.1)
    prefix, suffix, pos = table.prefix_of_value(piece, 0, F(1, 4))
    assert oracle.queries == 3 + 3 and prefix.components == (Interval(0, F(1, 4)),)
    assert table.value(1, prefix) == F(1, 2) and table.value(2, suffix) == F(3, 4)
    # a boundary that gives the value is used without any query (C.2)
    two = OrderedPiece.concat([prefix, suffix])
    q = oracle.queries
    p2, s2, pos2 = table.prefix_of_value(two, 2, F(1, 4))
    assert oracle.queries == q and p2 == prefix and pos2 == (1, F(1, 4))
    # value zero: the empty prefix and the full suffix; agent 1 values [1/2, 1) at zero so the
    # first boundary giving the full value is used for her suffix of value zero
    p0, s0, _ = table.prefix_of_value(two, 1, 0)
    assert p0.is_empty and s0 == two
    # a suffix of value zero of a piece whose last component is worth zero: the trim is
    # placed at the first boundary giving the full value, i.e. after the last valued component
    table.split_at(suffix.components[0], F(1, 2))
    three = OrderedPiece([Interval(0, F(1, 4)), Interval(F(1, 4), F(1, 2)), Interval(F(1, 2), 1)])
    trimmed_off, kept, pos = table.suffix_of_value(three, 1, 0)
    assert kept.components == (Interval(F(1, 2), 1),) and pos == (2, F(1, 2))
    assert table.value(1, kept) == 0 and table.value(1, trimmed_off) == 1
    assert oracle.queries == q + 3  # the split_at cost one Eval per agent, the suffix nothing


def test_split_at_and_record_union():
    vals = {0: V.uniform(), 1: V.PiecewiseConstantValuation([0, F(1, 2), 1], [2, 0])}
    oracle = RobertsonWebbOracle(vals)
    table = ValueTable(oracle, [0, 1])
    table.register_cake()
    c = Interval(0, 1)
    left, right = table.split_at(c, F(1, 3))
    assert oracle.evals == 2 + 2 and table.value_of_interval(1, left) == F(2, 3)
    whole = table.record_union(left, right)
    assert whole == c and oracle.evals == 4
    # splitting the same interval again at the same point makes no new queries
    table.split_at(c, F(1, 3))
    assert oracle.evals == 4
    with pytest.raises(KeyError):
        table.value_of_interval(0, Interval(F(1, 3), F(1, 2)))
