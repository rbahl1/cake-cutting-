"""The persistent AVL sequences (:mod:`cake_cutting.seq`) and the suffix views of
ordered pieces: sequence semantics, the AVL invariant, the node-creation bounds
used by the operation bound, and agreement of views with materialized pieces."""

import random
from fractions import Fraction

import pytest

from cake_cutting import seq
from cake_cutting import valuations as V
from cake_cutting.bookkeeping import ValueTable
from cake_cutting.oracle import RobertsonWebbOracle
from cake_cutting.pieces import Interval, OrderedPiece

F = Fraction


def check_avl(t):
    """Height, size, first/last and balance of every node; returns (height, size)."""
    if t is None:
        return 0, 0
    hl, sl = check_avl(t.left)
    hr, sr = check_avl(t.right)
    assert abs(hl - hr) <= 1, "AVL invariant violated"
    assert t.h == max(hl, hr) + 1 and t.size == sl + sr + 1
    assert t.first is (t.left.first if t.left else t.iv)
    assert t.last is (t.right.last if t.right else t.iv)
    return t.h, t.size


def items(k, base=0):
    return [(base + j, (F(base + j), F(1))) for j in range(k)]


class Counting:
    """Counts Node creations during a block."""

    def __enter__(self):
        self.count = 0
        orig = seq.Node.__init__
        counter = self

        def init(node, *a, **kw):
            counter.count += 1
            orig(node, *a, **kw)

        self.orig = orig
        seq.Node.__init__ = init
        return self

    def __exit__(self, *exc):
        seq.Node.__init__ = self.orig


def test_fibonacci_height_bound():
    for size in [1, 2, 3, 4, 7, 12, 20, 33, 54, 100, 1000]:
        assert seq.max_height(size) >= 1
        # the minimal AVL tree of height h has F(h+2) - 1 nodes
        h = seq.max_height(size)
        assert seq.min_size(h) <= size < seq.min_size(h + 1)


@pytest.mark.parametrize("seed", range(20))
def test_random_operations_keep_semantics_and_balance(seed):
    rng = random.Random(seed)
    model = [list(range(rng.randint(0, 40)))]
    trees = [seq.build([(x, (F(x), F(1))) for x in model[0]])]
    nxt = 1000
    for _ in range(300):
        op = rng.choice(["split", "concat", "join", "build"])
        if op == "build":
            k = rng.randint(0, 50)
            lst = list(range(nxt, nxt + k))
            nxt += k
            model.append(lst)
            trees.append(seq.build([(x, (F(x), F(1))) for x in lst]))
        elif op == "split":
            j = rng.randrange(len(model))
            k = rng.randint(0, len(model[j]))
            a, b = seq.split(trees[j], k)
            ha = seq.height(trees[j])
            model += [model[j][:k], model[j][k:]]
            trees += [a, b]
            assert seq.height(a) <= ha and seq.height(b) <= ha
        elif op == "concat":
            i, j = rng.randrange(len(model)), rng.randrange(len(model))
            if set(model[i]) & set(model[j]):
                continue
            model.append(model[i] + model[j])
            trees.append(seq.concat(trees[i], trees[j]))
        else:
            i, j = rng.randrange(len(model)), rng.randrange(len(model))
            if set(model[i]) & set(model[j]):
                continue
            x = nxt
            nxt += 1
            hi, hj = seq.height(trees[i]), seq.height(trees[j])
            t = seq.join(trees[i], x, (F(x), F(1)), trees[j])
            assert max(hi, hj) <= t.h <= max(hi, hj) + 1
            model.append(model[i] + [x] + model[j])
            trees.append(t)
        t, lst = trees[-1], model[-1]
        h, s = check_avl(t)
        assert s == len(lst) and seq.to_list(t) == lst
        assert h <= seq.max_height(max(s, 1)) if s else h == 0
        if lst:
            k = rng.randrange(len(lst))
            assert seq.nth(t, k)[0] == lst[k]
            assert seq.prefix_sum(t, 0, k) == sum(F(x) for x in lst[:k])
            assert seq.prefix_sum(t, 1, k) == k
            amount = F(rng.randint(0, len(lst)))
            kk, pk = seq.find_ge(t, 1, amount)
            assert kk == max(0, int(amount)) and pk == kk
            assert t.sum(0) == sum(F(x) for x in lst)
    # persistence: old trees are unchanged
    for t, lst in zip(trees, model):
        assert seq.to_list(t) == lst


@pytest.mark.parametrize("seed", range(10))
def test_node_creation_bounds(seed):
    """The bounds of the operation-count proof (PROOF.md, Lemmas 2.1-2.2): join creates
    <= 3|h(L)-h(R)| + 1 nodes, split <= 10 h(T), split3 <= 10 h(T), concat
    <= 7 h(L) + 3 max(h(L), h(R)) + 1."""
    rng = random.Random(seed)
    for _ in range(200):
        a, b = rng.randint(0, 300), rng.randint(0, 300)
        L = seq.build(items(a))
        R = seq.build(items(b, 10_000))
        # unbalance the inputs a little by joining random prefixes
        if a > 3 and rng.random() < 0.5:
            x, y = seq.split(L, rng.randint(0, a))
            L = seq.concat(y, x)
        with Counting() as c:
            seq.join(L, -1, (F(-1), F(1)), R)
        assert c.count <= 3 * abs(seq.height(L) - seq.height(R)) + 1
        k = rng.randint(0, a)
        with Counting() as c:
            seq.split(L, k)
        assert c.count <= 10 * seq.height(L)
        if a:
            with Counting() as c:
                left, (iv, _), right = seq.split3(L, min(k, a - 1))
            assert c.count <= 10 * seq.height(L)
            assert seq.to_list(left) + [iv] + seq.to_list(right) == seq.to_list(L)
        with Counting() as c:
            seq.concat(L, R)
        assert c.count <= 7 * seq.height(L) + 3 * max(seq.height(L), seq.height(R)) + 1


def _table(seed, n=4):
    rng = random.Random(seed)
    vals = {i: V.random_valuation(rng, segments=6, grid=60, zero_prob=0.3) for i in range(n)}
    oracle = RobertsonWebbOracle(vals, cut_rule=rng.choice(["left", "right"]))
    table = ValueTable(oracle, list(range(n)))
    table.register_cake()
    comps = []
    cur = OrderedPiece.cake().components[0]
    for y in sorted({F(rng.randint(1, 199), 200) for _ in range(rng.randint(3, 20))}):
        if cur.a < y < cur.b:
            left, cur = table.split_at(cur, y)
            comps.append(left)
    comps.append(cur)
    rng.shuffle(comps)
    return rng, oracle, table, comps


@pytest.mark.parametrize("seed", range(15))
def test_suffix_views_agree_with_materialized_pieces(seed):
    rng, oracle, table, comps = _table(seed)
    piece = table.make_piece(comps)
    tuple_piece = OrderedPiece(comps)
    for _ in range(30):
        agent = rng.randrange(4)
        total = table.value(agent, piece)
        if total == 0:
            continue
        amount = total * F(rng.randint(0, 20), 20)
        pos = table.suffix_position_of_value(piece, agent, amount)
        view = table.suffix_from(piece, pos)
        k0, y0 = pos
        if k0 < len(comps):
            assert view.components[1:] == tuple(comps[k0 + 1:])
            assert view.components[0] == Interval(y0, comps[k0].b)
        else:
            assert view.is_empty
        for i in range(4):
            assert table.value(i, view) == oracle.true_value(i, view)
        assert len(view) == len(view.components)
        if len(view):
            assert view.first == view.components[0] and view.last == view.components[-1]
            for j in range(len(view)):
                assert table.nth_component(view, j) == view.components[j]
        # a view of a view
        if len(view) and table.value(agent, view) > 0:
            amount2 = table.value(agent, view) * F(rng.randint(0, 10), 10)
            pos2 = table.suffix_position_of_value(view, agent, amount2)
            v2 = table.suffix_from(view, pos2)
            for i in range(4):
                assert table.value(i, v2) == oracle.true_value(i, v2)
            assert table.value(agent, v2) == amount2
            # the same position computed on the materialized copy
            copy = table.make_piece(view.components)
            assert table.suffix_position_of_value(copy, agent, amount2) == pos2
            for k in range(len(view)):
                q = (k, table.nth_component(view, k).a)
                assert table.normalize_left(view, agent, q) == table.normalize_left(copy, agent, q)
        # turning a view into a tree keeps components and values
        before = view.components
        vals = [table.value(i, view) for i in range(4)]
        table.full_tree(view)
        assert view.components == before and [table.value(i, view) for i in range(4)] == vals
        check_avl(view._tree)
