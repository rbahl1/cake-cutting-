"""Explicit bound on the operations executed by the protocol (proof in ``PROOF.md``).

Cost model: :mod:`cake_cutting.opcount`.  One operation is one executed bytecode
instruction of this package; arithmetic, comparisons, hashing and construction of
numbers, dictionary and set operations, and oracle queries cost one each; a built-in
whose work grows with its data costs one more per element it touches.

Every function ``c_*`` returns an upper bound on the operations of one call of the
program function it is named after, *including* everything that call executes,
except the parts that the docstring says are charged elsewhere.  The bounds are
assembled from

* ``own(key, t_1, ..., t_L)``: the instructions of the function itself, where
  ``t_j`` bounds how often the instructions of its ``j``-th loop (in source order)
  execute, i.e. the iterations plus the entries of that loop (``PROOF.md`` §1);
* the costs of the functions it calls, times the number of times each call site
  can execute;
* the data-dependent cost of the built-ins it uses (named in the comments).

Parameters: ``n`` = agents of the value table (the length of every value vector,
the number of agents of the top-level call), ``m`` = active agents (the agents of
the current ``Main`` call), ``h`` = an upper bound on the height of every tree
involved.

One cost is *prepaid* (``PROOF.md`` §3): a suffix view computes its value for an
agent at most once (one prefix sum, then cached), and that computation is charged
to the view's creation (:func:`c_suffix_from`).  Values are only computed for
active agents, so ``m`` evaluations per view suffice.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from typing import Dict

from . import bounds as B
from ._opprofiles import PROFILES
from .polybound import VARS, Poly, dec, is_poly, pmax, sum_below


def _loop_cost(loop, tj) -> int:
    _, _, cycle, entry = loop
    if isinstance(tj, tuple):  # (iterations, entries)
        it, en = tj
        if not is_poly(it, en) and (it < 0 or en < 0):
            raise ValueError("negative execution bound")
        return cycle * (it + en) + entry * en
    if not is_poly(tj) and tj < 0:   # one bound on iterations + entries (hence on entries)
        raise ValueError("negative execution bound")
    return (cycle + entry) * tj


def own(key: str, *t) -> int:
    """Instructions of one call of ``key`` itself: those outside every loop once; for loop
    ``j``, ``t[j]`` is ``(iterations, entries)`` (its cycle runs at most
    ``iterations + entries`` times, its other instructions ``entries`` times) or a
    single number bounding both."""
    prof = PROFILES[key]
    loops = prof["loops"]
    if len(t) != len(loops):
        raise ValueError(f"{key}: {len(loops)} loops, {len(t)} execution bounds")
    return prof["outside"] + sum(_loop_cost(lp, tj) for lp, tj in zip(loops, t))


def own_calls(key: str, calls: int, *t) -> int:
    """Instructions of ``calls`` calls of ``key`` together, ``t[j]`` bounding the executions
    of loop ``j`` summed over all of them."""
    return calls * PROFILES[key]["outside"] + own(key, *t) - PROFILES[key]["outside"]


def sub(parent: str, name: str, k: int = 0) -> str:
    """Key of the ``k``-th (in source order) nested code object called ``name`` of
    ``parent`` (``genexpr``, ``lambda`` or an inner function's name)."""
    prefix = f"{parent}.<{name}@"
    keys = [key for key in PROFILES if key.startswith(prefix) and ".<" not in key[len(prefix):]]
    keys.sort(key=lambda key: int(key[len(prefix):-1]))
    return keys[k]


def nested(key: str) -> int:
    """All instructions of a lambda or generator expression: a bound on one call of the
    lambda, or on the work of the generator per element of its iterable (and once
    more for its exhaustion)."""
    return PROFILES[key]["total"]


def clog2(x: int) -> int:
    """``ceil(log2(x))`` for an integer ``x >= 1`` (for a polynomial: ``x``)."""
    if is_poly(x):
        return x
    return (x - 1).bit_length() if x > 1 else 0


@lru_cache(maxsize=None)
def tail_moment(k: int, U0: int) -> Fraction:
    """``sum_{u > U0} u^k / 2^u`` exactly: ``sum_{u>=1} u^k x^u = sum_j j! S(k, j)
    x^j / (1 - x)^(j+1)`` at ``x = 1/2`` (``S`` = Stirling numbers of the second kind,
    ``k >= 1``; the sum is 1 for ``k = 0``), minus the first ``U0`` terms."""
    if k == 0:
        total = Fraction(1)
    else:
        S = [[0] * (k + 1) for _ in range(k + 1)]
        S[0][0] = 1
        for a in range(1, k + 1):
            for b in range(1, a + 1):
                S[a][b] = b * S[a - 1][b] + S[a - 1][b - 1]
        fact = 1
        total = Fraction(0)
        for j in range(0, k + 1):
            if j:
                fact *= j
            total += fact * S[k][j] * Fraction(1, 2 ** j) * 2 ** (j + 1)
    return total - sum(Fraction(u ** k, 2 ** u) for u in range(1, U0 + 1))


def c_sorted(L: int, cmp: int = 1) -> int:
    """Built-in ``sorted`` (or ``list.sort``) of ``L`` elements: at most ``L ceil(log2 L)``
    comparisons, each costing ``cmp``, plus one per element moved in and out."""
    return L * clog2(L) * cmp + 2 * L + 1


def digits(x: int) -> int:
    """Decimal digits of an integer ``x >= 1`` (for a polynomial: ``x``)."""
    if is_poly(x):
        return x
    return len(str(abs(x)))


# ============================================================================
# seq.py -- persistent AVL sequences (PROOF.md §2)
# ============================================================================

HEIGHT = own("seq.height")
SIZE = own("seq.size")
JOINSTEP = max(own("seq._join_right"), own("seq._join_left")) + 3 * HEIGHT
ROT = max(own("seq._rotate_left"), own("seq._rotate_right"))


def c_node(n: int) -> int:
    """Creating one node: ``Node.__init__`` with one of its three comprehensions over the
    ``n`` agents (the one for two children is the most expensive)."""
    return own("seq.Node.__init__", 0, 0, (n, 1))


def c_join(D: int, n: int) -> int:
    """``join(L, x, R)`` with ``D = |h(L) - h(R)|``: two ``height`` calls, at most ``D - 1``
    invocations of ``_join_right``/``_join_left`` (three ``height`` calls each), at most
    ``D`` rotations and at most ``3D + 1`` new nodes (Lemma 2.1)."""
    return own("seq.join") + 2 * HEIGHT + dec(D) * JOINSTEP + D * ROT + (3 * D + 1) * c_node(n)


def _joins(count: int, dsum: int, n: int) -> int:
    """``count`` joins whose height differences sum to at most ``dsum``."""
    return count * (own("seq.join") + 2 * HEIGHT + c_node(n)) + dsum * (JOINSTEP + ROT + 3 * c_node(n))


def c_split(h: int, n: int) -> int:
    """``split(T, k)`` with ``h(T) <= h``: at most ``h + 1`` invocations (one ``size`` call
    each) and at most ``h`` joins with height differences summing to at most ``3h``
    (Lemma 2.2)."""
    return (h + 1) * (own("seq.split") + SIZE) + _joins(h, 3 * h, n)


def c_split3(h: int, n: int) -> int:
    """``split3(T, k)`` with ``h(T) <= h``: at most ``h`` invocations and ``h - 1`` joins with
    height differences summing to at most ``3h`` (Lemma 2.2)."""
    return h * (own("seq.split3") + SIZE) + _joins(dec(h), 3 * h, n)


def c_split_last(h: int, n: int) -> int:
    """``_split_last(T)`` with ``h(T) <= h``: at most ``h`` invocations and ``h - 1`` joins with
    height differences summing to at most ``2h`` (Lemma 2.2)."""
    return h * own("seq._split_last") + _joins(dec(h), 2 * h, n)


def c_concat(h: int, n: int) -> int:
    """``seq.concat(L, R)``, all heights at most ``h``: ``_split_last`` and one join."""
    return own("seq.concat") + c_split_last(h, n) + c_join(h, n)


def c_nth(h: int) -> int:
    """``nth``: at most ``h`` iterations, one ``size`` call each."""
    return own("seq.nth", (h, 1)) + h * SIZE


def c_find_ge(h: int) -> int:
    """``find_ge``: at most ``h`` iterations, one ``size`` call each."""
    return own("seq.find_ge", (h, 1)) + h * SIZE


def c_prefix_sum(h: int) -> int:
    """``prefix_sum``: at most ``h`` iterations, one ``size`` call each."""
    return own("seq.prefix_sum", (h, 1)) + h * SIZE


def c_build(L: int, n: int) -> int:
    """``build`` of ``L`` items: ``2L + 1`` invocations of ``rec`` and ``L`` nodes."""
    return own("seq.build") + (2 * L + 1) * own(sub("seq.build", "rec")) + L * c_node(n)


# ============================================================================
# pieces.py
# ============================================================================

IVINIT = own("pieces.Interval.__init__")
IVSPLIT = own("pieces.Interval.split") + 2 * IVINIT
IVEQ = own("pieces.Interval.__eq__")
LEN = own("pieces.OrderedPiece.__len__")
FIRST = own("pieces.OrderedPiece.first")
LAST = own("pieces.OrderedPiece.last")
ISEMPTY = own("pieces.OrderedPiece.is_empty") + LEN
FROMTREE = own("pieces.OrderedPiece._from_tree")
VIEW = own("pieces.OrderedPiece._view")


def c_op_init(L: int) -> int:
    """``OrderedPiece(components)`` with ``L`` components (``tuple`` copies them)."""
    return own("pieces.OrderedPiece.__init__", (L, 1)) + L


EMPTY = own("pieces.OrderedPiece.empty") + c_op_init(0)
CAKE = own("pieces.OrderedPiece.cake") + own("pieces.OrderedPiece.interval") + IVINIT + c_op_init(1)


def c_vals_of(n: int) -> int:
    """``ValueTable.vals_of``: one dictionary lookup and a tuple of ``n`` entries."""
    return own("bookkeeping.ValueTable.vals_of") + n


def c_full_tree(h: int, n: int) -> int:
    """``ValueTable.full_tree`` of a suffix view of a tree of height ``<= h``: one split, or
    one ``split3``, a value vector and one join (the view becomes a tree in place)."""
    return (own("bookkeeping.ValueTable.full_tree")
            + pmax(c_split(h, n), c_split3(h, n) + c_vals_of(n) + c_join(h, n)))


FULL_TREE_EARLY = own("bookkeeping.ValueTable.full_tree")    # a piece that is a tree already


def c_view_join(hv: int, h_cat: int, n: int) -> int:
    """Appending a suffix view (base tree of height ``<= hv``) in ``OrderedPiece.concat``:
    one ``split3`` of its tree, possibly the head's value vector, one join."""
    return c_split3(hv, n) + c_vals_of(n) + c_join(h_cat, n)


def c_tree_join(h_cat: int, n: int) -> int:
    """Appending a tree in ``OrderedPiece.concat``: ``seq.concat`` (``_split_last`` and one
    join when the accumulated tree is not empty)."""
    return own("seq.concat") + c_split_last(h_cat, n) + c_join(h_cat, n)


def c_concat_frame(Kin: int, n: int) -> int:
    """``OrderedPiece.concat`` of ``Kin`` pieces without the appending work: the list of
    nonempty pieces, the ``all`` test, the loop, the wrap (or ``empty``)."""
    K = Kin
    return (own("pieces.OrderedPiece.concat", (Kin, 1), (K, 1), 0) + Kin * LEN
            + (K + 1) * nested(sub("pieces.OrderedPiece.concat", "genexpr")) + K + FROMTREE + EMPTY)


def c_piece_concat(Kin: int, h_cat: int, n: int, views=()) -> int:
    """``OrderedPiece.concat`` of ``Kin`` tree-backed pieces of the table (Lemma 4.1).
    ``views`` lists pairs ``(base height, number of pieces)`` for the pieces that may be
    suffix views (or trees); each of them costs the larger of the two appending paths,
    every other piece is a tree.  All results have height ``<= h_cat``.  An iterable
    given as a generator is charged to the caller."""
    nv = sum((c for _, c in views), 0)
    tree = c_tree_join(h_cat, n)
    view_cost = sum((c * pmax(c_view_join(hv, h_cat, n), tree) for hv, c in views), 0)
    return c_concat_frame(Kin, n) + view_cost + (Kin - nv) * tree


def c_concat_onto_tree(hv: int, h_cat: int, n: int) -> int:
    """``K[a] + X[a]`` in ``shrink`` when ``K[a]`` is a tree: the first piece costs only
    ``seq.concat(None, .)``; ``X[a]`` is a view (base ``<= hv``) or a tree.  The case that
    ``K[a]`` is still a view is charged separately (:meth:`MainCost.total`)."""
    return (c_concat_frame(2, n) + own("seq.concat")
            + pmax(c_view_join(hv, h_cat, n), c_tree_join(h_cat, n)))


def c_prefix_at(h: int, n: int) -> int:
    """``prefix_at`` of a tree (Lemma 4.1)."""
    return own("pieces.OrderedPiece.prefix_at") + FULL_TREE_EARLY + c_split(h, n) + FROMTREE


# ============================================================================
# bookkeeping.py -- the value table
# ============================================================================

VOI = own("bookkeeping.ValueTable.value_of_interval")
BASE = own("bookkeeping.ValueTable._base")           # tree-backed pieces only (Lemma 4.1)
WRAP = own("bookkeeping.ValueTable._wrap") + FROMTREE
SET = own("bookkeeping.ValueTable._set")             # value vectors charged to their creator
TREE_OF = own("bookkeeping.ValueTable.tree_of", 0)   # a tree: returned as it is


def c_comp(n: int, h: int) -> int:
    """``_comp``: ``_base`` and one of: ``vals_of``, the stored first component, ``nth``."""
    return own("bookkeeping.ValueTable._comp") + BASE + pmax(c_vals_of(n), c_nth(h))


def c_nth_component(n: int, h: int) -> int:
    return own("bookkeeping.ValueTable.nth_component") + c_comp(n, h)


def c_value(h: int) -> int:
    """``value``: ``len``, ``_base`` and the root's stored sum; the first evaluation of a view
    for an agent (one prefix sum) is prepaid by the view (:func:`c_view_prepaid`)."""
    return own("bookkeeping.ValueTable.value") + LEN + BASE


def c_view_prepaid(m: int, h: int) -> int:
    """First evaluations of one suffix view for at most ``m`` agents (Lemma 3.1)."""
    return m * c_prefix_sum(h)


LEARN = own("bookkeeping.ValueTable._learn_split") + VOI + 1 + 2 * SET


def c_split_by_cut(m: int, n: int) -> int:
    """``split_by_cut``: one Cut, ``Interval.split``, two ``_set`` that may create the two
    new value vectors (``n`` elements each) and ``_learn_split`` (at most one Eval) for
    the other ``m - 1`` active agents."""
    return (own("bookkeeping.ValueTable.split_by_cut", (m, 1)) + VOI + 1 + IVSPLIT
            + 2 * (SET + n) + (m - 1) * LEARN)


def c_complement_prefix(m: int, n: int) -> int:
    return own("bookkeeping.ValueTable.complement_prefix", (m, 1)) + IVINIT + n + m * (2 * VOI + SET)


LEFTMOST = own("bookkeeping.ValueTable._leftmost_start") + VOI


def c_locate(n: int, h: int) -> int:
    """``_locate``: tree path ``find_ge`` + ``nth``; view path ``_comp`` + ``prefix_sum`` +
    ``find_ge`` + ``nth``."""
    return (own("bookkeeping.ValueTable._locate") + BASE + c_comp(n, h) + c_prefix_sum(h)
            + c_find_ge(h) + c_nth(h))


def c_position_of_value(m: int, n: int, h: int) -> int:
    return (own("bookkeeping.ValueTable.position_of_value") + c_value(h) + LEN + c_locate(n, h)
            + c_nth_component(n, h) + c_split_by_cut(m, n))


def c_suffix_position_of_value(m: int, n: int, h: int) -> int:
    return own("bookkeeping.ValueTable.suffix_position_of_value") + c_value(h) + c_position_of_value(m, n, h)


def c_normalize_left(n: int, h: int) -> int:
    """``normalize_left`` (all paths): ``len``, ``last``, three ``nth_component``, one
    ``value_of_interval`` of a new ``Interval``, two ``_leftmost_start``, ``_base``,
    ``_comp``, two prefix sums and one ``find_ge``."""
    return (own("bookkeeping.ValueTable.normalize_left") + LEN + LAST + 3 * c_nth_component(n, h)
            + VOI + IVINIT + 2 * LEFTMOST + BASE + c_comp(n, h) + 2 * c_prefix_sum(h) + c_find_ge(h))


def c_suffix_from(m: int, n: int, h: int) -> int:
    """``suffix_from``: ``len``, ``_base``, one ``nth``, a new ``Interval``, a view and its
    prepaid evaluations; no node is created."""
    return (own("bookkeeping.ValueTable.suffix_from") + LEN + BASE + c_nth(h) + IVINIT + VIEW
            + c_view_prepaid(m, h) + EMPTY)


def c_prefix_to(n: int, h: int) -> int:
    """``prefix_to`` of a tree (Lemma 4.1): ``len``, ``tree_of``, ``nth``, one split, a new
    ``Interval``, its value vector, one join, a wrap."""
    return (own("bookkeeping.ValueTable.prefix_to") + LEN + TREE_OF + c_nth(h)
            + c_split(h, n) + IVINIT + c_vals_of(n) + c_join(h, n) + WRAP)


def c_prefix_of_value(m: int, n: int, h: int) -> int:
    """``prefix_of_value`` of a tree (Lemma 4.1): ``value``, ``len``, ``tree_of``, ``_locate``,
    then either a split and two wraps (boundary) or ``split3``, ``split_by_cut``, two
    value vectors, two joins and two wraps (inside)."""
    boundary = c_split(h, n) + 2 * WRAP
    inside = c_split3(h, n) + c_split_by_cut(m, n) + 2 * c_vals_of(n) + 2 * c_join(h, n) + 2 * WRAP
    return (own("bookkeeping.ValueTable.prefix_of_value") + c_value(h) + LEN + TREE_OF
            + c_locate(n, h) + EMPTY + pmax(boundary, inside))


def c_suffix_of_value(m: int, n: int, h: int) -> int:
    return own("bookkeeping.ValueTable.suffix_of_value") + c_value(h) + c_prefix_of_value(m, n, h)


def c_register_cake(n: int) -> int:
    """``register_cake``: ``n`` Evals and ``_set`` calls, one new value vector."""
    return own("bookkeeping.ValueTable.register_cake", (n, 1)) + IVINIT + n + n * (1 + SET)


def c_set_active(m: int) -> int:
    return own("bookkeeping.ValueTable.set_active", (m, 1)) + m


def c_value_table_init(n: int) -> int:
    return own("bookkeeping.ValueTable.__init__", (n, 1)) + 2 * n


def c_cake_tree(n: int) -> int:
    """The one tree built from a tuple-backed piece: the cake's, on its first evaluation
    (``_base`` -> ``tree_of`` -> ``build`` of one item; Lemma 4.1)."""
    return (own("bookkeeping.ValueTable.tree_of", (1, 1)) + own("pieces.OrderedPiece.components")
            + c_vals_of(n) + c_build(1, n))


# ============================================================================
# equilibrium.py -- the SubCore fallback (PROOF.md §5.3)
# ============================================================================

def eq_counts(w: int, k: int) -> Dict[str, int]:
    """Counts of one ``allocate`` call with ``w`` agents and ``k`` nonempty pieces
    (Lemma 5.3), with ``a = w - 1`` buyers in the top state:

    * ``HS`` / ``LS``: computed high / low states, ``C(a+k, a)`` / ``C(a+k, a+1)``;
    * ``HC`` / ``LC``: calls of ``high`` / ``low`` (memo hits included),
      ``1 + a C(a+k-1, a)`` / ``k C(a+k-1, a)``;
    * ``VIEWS``: suffixes made by ``high`` (= ``LC``); ``LX``: ``utility`` calls made by
      ``low`` (= ``HC - 1``);
    * ``HXY``: iterations of the inner loop of ``high`` and calls of
      ``suffix_position_of_value``, ``a k C(a+k-2, a-1)`` (= ``bounds.E(w, k)``);
    * ``LXY``: values read by those ``utility`` calls, ``a k C(a+k-2, a)``."""
    if w <= 0:
        return dict(HS=0, LS=0, HC=0, LC=0, VIEWS=0, LX=0, HXY=0, LXY=0)
    a = w - 1
    c1 = B.binom(a + k - 1, a)
    return dict(HS=B.binom(a + k, a), LS=B.binom(a + k, a + 1), HC=1 + a * c1, LC=k * c1,
                VIEWS=k * c1, LX=a * c1, HXY=a * k * B.binom(a + k - 2, a - 1),
                LXY=a * k * B.binom(a + k - 2, a))


def eq_counts_core_sym(m) -> Dict[str, Poly]:
    """:func:`eq_counts` for the Core call of ``Main(m)`` (``w = m - 1``, ``k = m``) bounded
    by polynomials in ``m`` and ``G = C(2m, m)``, valid for ``m >= 4``: the counts are
    ``C(2m-2, m-2) <= G_{m-1}``, ``C(2m-2, m-1) = G_{m-1}``, ``C(2m-3, m-2) = G_{m-1}/2``,
    ``C(2m-4, m-3) <= C(2m-4, m-2) = G_{m-2}``, and ``G_{m-1} <= (2/7) G``,
    ``G_{m-2} <= (3/35) G`` (``G_j / G_{j-1} = 2(2j-1)/j``, at least 7/2 for ``j >= 4``
    and at least 10/3 for ``j >= 3``)."""
    G = Poly.var("G")
    g1 = Fraction(2, 7) * G
    g2 = Fraction(3, 35) * G
    a = m - 2
    return dict(HS=g1, LS=g1, HC=1 + a * g1 / 2, LC=m * g1 / 2, VIEWS=m * g1 / 2,
                LX=a * g1 / 2, HXY=a * m * g2, LXY=a * m * g2)


def eq_counts_class_sym(e) -> Dict[str, Poly]:
    """:func:`eq_counts` for a stage-3 class of ``e`` agents (``w = k = e``) bounded by
    polynomials in ``e`` and ``G``, where ``G`` here stands for ``G_e = C(2e, e)``:
    ``C(2e-1, e-1) = C(2e-1, e) = G_e/2``, ``C(2e-2, e-1) = G_{e-1} <= G_e/2``,
    ``C(2e-3, e-2) = C(2e-3, e-1) = G_{e-1}/2 <= G_e/4``."""
    Ge = Poly.var("G")
    a = e - 1
    return dict(HS=Ge / 2, LS=Ge / 2, HC=1 + a * Ge / 2, LC=e * Ge / 2, VIEWS=e * Ge / 2,
                LX=a * Ge / 2, HXY=a * e * Ge / 4, LXY=a * e * Ge / 4)


# ============================================================================
# subcore.py (PROOF.md §5)
# ============================================================================

class Ctx:
    """The parameters of one SubCore family: table size ``n``, active agents ``m`` and
    the tree height bound ``h`` (all its pieces are views of trees of height ``<= h``)."""

    def __init__(self, n: int, m: int, h: int) -> None:
        self.n, self.m, self.h = n, m, h
        self.VAL = own("subcore.SubCoreCall.val") + c_value(h)
        self.NTHC = c_nth_component(n, h)
        self.SFROM = c_suffix_from(m, n, h)
        self.SSP = own("subcore.suffix_start_position") + 2 * LEN + self.NTHC + FIRST
        self.HSUF = own("subcore.SubCoreCall.hint_suffix")
        # place_trim without its trim: two candidates (hint path)
        self.PTB = (own("subcore.SubCoreCall.place_trim", 3) + self.HSUF
                    + 2 * (ISEMPTY + LEN + 2 * LAST + self.SSP + FIRST + self.VAL + 2))
        # the trim itself: suffix_position_of_value + normalize_left
        self.TRIM = c_suffix_position_of_value(m, n, h) + c_normalize_left(n, h)
        self.IS_SUFFIX = own("subcore._is_suffix") + ISEMPTY + 4 * LEN + 2 * LAST + self.NTHC + FIRST
        self.SSPV = c_suffix_position_of_value(m, n, h)
        self._memo: Dict = {}

    # -- the equilibrium fallback (PROOF.md §5.3) ----------------------------
    def augment(self, w: int, k: int) -> int:
        """``_augment`` in a graph with ``w`` buyers and ``k`` goods: each buyer is dequeued at
        most once (``w`` iterations of the search), each adjacency list scanned at most once
        (at most ``wk`` edges), the path has at most ``w`` buyers; two lists of ``k`` and
        ``w`` entries, the ``len`` calls and the appends."""
        return (own("equilibrium._augment", (w, 1), (w * k, w), (w, 1))
                + k + w + 2 + (w + 1) + w)

    def cover_matching(self, w: int, k: int) -> int:
        """``_cover_matching`` with ``w`` buyers, ``k`` goods (at most ``wk`` edges): phase 1 at
        most ``w`` augmentations; phase 2 at most ``k`` searches, each dequeuing at most ``k``
        goods, scanning at most ``wk`` edges and flipping a path of at most ``k`` goods."""
        t = ((k, 1),               # L170 radj
             (w, 1),               # L171 buyers
             (w * k, w),           # L172 their edges
             (w, 1),               # L174 phase 1
             (k, 1),               # L177 phase 2
             (k * k, k),           # L187 search queue
             (k * w * k, k * k),   # L190 edges of the dequeued goods
             (k * k, k))           # L207 flip
        return (own("equilibrium._cover_matching", *t) + w + k + w * k
                + w * self.augment(w, k)
                + k * (w + k + (k + 1) + k))      # lists, len(queue), appends

    def verify_eq(self, w: int, P: int) -> int:
        """``equilibrium._verify`` with ``w`` agents and ``P`` pieces (no queries)."""
        return (own("equilibrium._verify", (w, 1), (w, 1), (w * w, w), (w * P, w))
                + (w + w * w + w * P) * self.VAL)

    def fallback(self, w: int, k: int, P: int, cnt: Dict = None) -> int:
        """``equilibrium.allocate`` with ``w`` agents, ``P`` pieces of which ``k`` are nonempty
        (trees of height ``<= h``); ``cnt`` are the counts of Lemma 5.3
        (:func:`eq_counts` by default)."""
        if cnt is None:
            cnt = eq_counts(w, k)
        VAL = c_value(self.h)
        HS, LS, HC, LC = cnt["HS"], cnt["LS"], cnt["HC"], cnt["LC"]
        VIEWS, LX, HXY, LXY = cnt["VIEWS"], cnt["LX"], cnt["HXY"], cnt["LXY"]
        init = (own("equilibrium._Market.__init__", (P, 1), (k, 1), (k, 1),
                    (w * k + w, w + 1), (w * k + w, w + 1))
                + w + P * LEN + k * FIRST + w * k * VAL)
        high = (own_calls("equilibrium._Market.high", HC, (k * HS, HS), (HXY, VIEWS))
                + HXY * self.SSPV + VIEWS * self.SFROM)
        low = own_calls("equilibrium._Market.low", LC, (w * LS, LS))
        util = (own_calls("equilibrium._Market.utility", LX + w, (LXY + w * k, LX + w))
                + (LXY + w * k) * VAL)
        tail = (own("equilibrium.allocate", (w, 1), (w, 1), (w * k, w), (w, 1), (k, 1), (w, 1), (w, 1))
                + w                                   # list(agents)
                + w * k * VAL + w                     # tight edges, adj.append
                + self.cover_matching(w, k)
                + 2 * w                               # used.add
                + (P + 1) * nested(sub("equilibrium.allocate", "genexpr"))
                + c_sorted(P) + P * (nested(sub("equilibrium.allocate", "lambda")) + LEN)
                + w * (EMPTY + 2)                     # unmatched agents, len(free)
                + self.verify_eq(w, P))
        return init + high + low + util + tail

    def hint_index(self, P: int) -> int:
        return own("subcore.SubCoreCall.hint_index", P + 1) + ISEMPTY + LAST + P * (LEN + LAST)

    def fav(self, P: int) -> int:
        """``favorite_unallocated``: ``max`` over a generator of ``P`` values, a list of at
        most ``P`` more values, ``hint_index`` and one ``in`` test on that list."""
        return (own("subcore.SubCoreCall.favorite_unallocated", P + 1)
                + (P + 1) * nested(sub("subcore.SubCoreCall.favorite_unallocated", "genexpr")) + P
                + 2 * P * self.VAL + self.hint_index(P) + P)

    def rightmost(self, g: int) -> int:
        """``rightmost`` over ``g`` agents (two 2-tuple comparisons per agent)."""
        return own("subcore.SubCoreCall.rightmost", g + 1) + 4 * g

    def break_tie(self, g: int, P: int) -> int:
        return (own("subcore.SubCoreCall.break_tie", g + 1, g + 1) + 2 * LEN + self.NTHC
                + g * (self.HSUF + ISEMPTY + LEN + 2 * FIRST + 2) + g * self.hint_index(P)
                + g + g * nested(sub("subcore.SubCoreCall.break_tie", "lambda")) + g)

    def verify(self, w: int, P: int) -> int:
        return (own("subcore.SubCoreCall.verify", w + 1, w * w + w, w * P + w)
                + w * (self.IS_SUFFIX + self.VAL) + w * w * self.VAL + w * P * self.VAL)

    def init(self, w: int, P: int) -> int:
        """``SubCoreCall.__init__``: two comprehensions over the agents, ``list``/``dict`` copies."""
        return own("subcore.SubCoreCall.__init__", w + 1, w + 1) + P + 2 * w

    def contest(self, s: int, w: int, P: int) -> int:
        """``contest(step)`` with ``s = step`` contested pieces, in a call with ``w`` agents and
        ``P`` pieces, *without* its trims and without the recursive calls (which are
        charged to ``TRIM`` and to :meth:`R`)."""
        g = s + 1                      # agents of the group
        U = P - s                      # uncontested pieces (>= 1)
        it = s - 1                     # iterations of the while loop
        t = (P + 1,                    # L257 uncontested
             g + 1,                    # L261 beta
             g + 1, g * s + g,         # L266/L267 trims
             s + 1,                    # L273 winners
             s + 1,                    # L284 Ye-Bai branch
             s,                        # L288 while
             it * (g + 1),             # L289 nonwinners
             it * (s + 1),             # L291 margins
             it * (w + 1),             # L294 sub_agents
             it * s,                   # L307 sub_hints
             it * (s + 1),             # L324 margins list
             it * s,                   # L325 benchmarks
             it * s,                   # L328 tentative
             it * s,                   # L329 taken
             it * (s + 1),             # L331
             it * (s + 1),             # L346 beta update
             g + 1,                    # L350 losers
             U + 1,                    # L354 cands
             g + 1)                    # L363 holder
        cost = own("subcore.SubCoreCall.contest", *t)
        cost += 2 * g                                                     # two slices of the agents
        cost += (s + 1) * nested(sub("subcore.SubCoreCall.contest", "genexpr")) + c_sorted(s)
        cost += g * ((U + 1) * nested(sub("subcore.SubCoreCall.contest", "genexpr", 1)) + U * self.VAL + U)
        cost += g * s * (self.VAL + self.PTB)
        cost += s * (self.rightmost(g) + self.break_tie(g, P))
        cost += s * self.SFROM                                            # Ye-Bai branch
        per_iter = (s * (self.rightmost(g) + self.SFROM)                  # margins
                    + it * (self.SFROM + (s + 1) * nested(sub("subcore.SubCoreCall.contest", "genexpr", 2)) + s)
                    + s * self.rightmost(g) + self.break_tie(g, P) + self.SFROM   # L331
                    + s * self.VAL)                                       # beta update
        cost += it * per_iter
        cost += (U + 1) * nested(sub("subcore.SubCoreCall.contest", "genexpr", 3)) + U * self.VAL + U
        cost += U * self.VAL + self.hint_index(P) + U
        return cost

    def A(self, w: int, P: int) -> int:
        if is_poly(w, P):
            return self._A(w, P)
        key = ("A", w, P)
        if key not in self._memo:
            self._memo[key] = self._A(w, P)
        return self._memo[key]

    def _A(self, w: int, P: int) -> int:
        """One ``SubCoreCall`` with ``w`` agents and ``P`` pieces: ``__init__``, ``run``, the
        favorite of every agent, the contests of steps ``1..w-1`` (step 0 never contests:
        nothing is allocated yet) and ``verify``; without trims and recursive calls."""
        return (self.init(w, P) + own("subcore.SubCoreCall.run", w + 1) + w * self.fav(P)
                + sum_below(lambda s: self.contest(s, w, P), "s", w) + self.verify(w, P))

    def T(self, w: int) -> int:
        """``T(w)``: the recursive calls below a call with ``w`` agents -- for every step ``s``
        of it, calls with distinct sizes ``j`` in ``{1, ..., s - 1}`` and ``s`` pieces
        (Lemma 5.1): ``sum_{s<w} sum_{j<s} R(j, s)``."""
        Tm = self._memo.setdefault("T", [0, 0])            # T(0) = T(1) = 0
        while len(Tm) <= w:
            v = len(Tm) - 1                                 # T(v + 1) = T(v) + sum_{j<v} R(j, v)
            Tm.append(Tm[v] + sum((self.A(j, v) + Tm[j] for j in range(1, v)), 0))
        return Tm[w]

    def R(self, w: int, P: int) -> int:
        """The SubCore call tree rooted at a call with ``w`` agents and ``P`` pieces, without
        trims: ``A(w, P) + T(w)``."""
        return self.A(w, P) + self.T(w)

    def zinf(self, U0: int = 40):
        """``Z = sum_{u>=1} alpha(u) / 2^u`` with ``alpha(u) = sum_{j<u} A(j, u)``, so that
        ``T(w) <= 2^(w-1) Z`` for every ``w`` (Lemma 5.2): the terms ``u <= U0`` exactly,
        the others from the polynomial bound of ``alpha`` (:func:`sum_below`)."""
        if "zinf" not in self._memo:
            head = sum((Fraction(1, 2 ** u) * sum_below(lambda j: self.A(j, u), "j", u)
                        for u in range(1, U0 + 1)), 0)
            usym = Poly.var("u")
            alpha = Poly.lift(sum_below(lambda j: self.A(j, usym), "j", usym)).positive()
            tail = 0
            for k, coef in alpha.by_power("u").items():
                tail = tail + coef * tail_moment(k, U0)
            z = head + tail
            if isinstance(z, Poly) and set(z.terms) <= {(0,) * len(VARS)}:
                z = z.terms.get((0,) * len(VARS), Fraction(0))    # a number
            self._memo["zinf"] = z
        return self._memo["zinf"]

    def subcore(self, w: int, P: int, pow2=None, k=None, cnt=None) -> int:
        """``subcore(...)``: the call tree plus at most ``S_w`` trims (Appendix D), and the
        equilibrium fallback on ``k <= P`` nonempty pieces (``k = P`` by default).  For a
        polynomial ``w``, ``pow2`` is a polynomial bound on ``2^w``, then
        ``T(w) <= (pow2 / 2) Z`` and ``S_w <= 2 pow2``, and ``cnt`` gives the counts of the
        fallback as polynomials (:func:`eq_counts_core_sym`)."""
        if k is None:
            k = P
        fb = self.fallback(w, k, P, cnt)
        if pow2 is None:
            return own("subcore.subcore") + self.R(w, P) + B.S(w) * self.TRIM + fb
        return (own("subcore.subcore") + self.A(w, P) + Fraction(1, 2) * pow2 * self.zinf()
                + 2 * pow2 * self.TRIM + fb)

    def class_subcores(self, m, pow2=None):
        """The SubCore calls of stage 3: one per profile class, the class sizes ``e`` forming
        a partition of ``m`` into at least two parts.  Exactly: the maximum over all such
        partitions (dynamic programming).  For a polynomial ``m`` (``pow2`` bounding
        ``2^m``): at most ``m`` parts, ``A(e, e + 1) <= A(m, m + 1)``, and
        ``sum 2^e <= 2^(m-1) + 2`` over the parts (Lemma 7.3)."""
        if pow2 is None:
            return max_class_split(m, lambda e: self.subcore(e, e + 1, k=e))
        # fallback: F(e) <= F0(e) + G_e F1(e) with G_e = C(2e, e); over the classes,
        # sum F0(e) <= m F0(m) and sum G_e F1(e) <= F1(m) (m/(m-1)) G_{m-1} <= (8/21) G F1(m)
        # (G_e / e is nondecreasing; Lemma 7.3)
        fb = Poly.lift(self.fallback(m, m, m + 1, eq_counts_class_sym(m))).positive()
        parts = fb.by_power("G")
        if set(parts) - {0, 1}:
            raise ValueError("the fallback bound is not linear in G")
        fb_total = m * parts.get(0, Poly()) + Fraction(8, 21) * Poly.var("G") * parts.get(1, Poly())
        return (m * (own("subcore.subcore") + self.A(m, m + 1))
                + Fraction(1, 2) * (Fraction(1, 2) * pow2 + 2) * self.zinf()
                + 2 * (Fraction(1, 2) * pow2 + 2) * self.TRIM
                + fb_total)


def c_unallocated_part(m: int, n: int, h: int, h_view=None) -> int:
    """``unallocated_part`` of a tree of height ``<= h``; if the piece may be a suffix view
    (stage 3: ``W_h`` equal to one snapshot piece), ``h_view`` bounds its base tree and
    the conversion of the view into a tree (``full_tree``) is added (Lemma 4.1)."""
    conv = 0 if h_view is None else c_full_tree(h_view, n)
    return (own("subcore.unallocated_part") + ISEMPTY + 2 * LEN + c_nth_component(n, h) + FIRST + conv
            + pmax(c_prefix_at(h, n), c_complement_prefix(m, n) + c_prefix_to(n, h)))


def c_equal_division(m: int, n: int, h: int) -> int:
    return (own("subcore.equal_division", m + 1, m + 1) + c_value(h) + m + (m - 1) * EMPTY
            + (m - 1) * c_prefix_of_value(m, n, h))


def c_core(m: int, n: int, h: int) -> int:
    """``core`` with ``m`` agents on a residue whose derived trees have height ``<= h``."""
    ctx = Ctx(n, m, h)
    t = (m + 1,          # L455 others
         m,              # L456 benchmarks
         m,              # L459 allocation
         m + 1,          # L462 free
         m + 1,          # L467 leftovers
         m * m)          # L470 holder (m entries, m - 1 iterations each)
    return (own("subcore.core", *t) + 2 * m            # list(agents), `cutter not in agents`
            + c_equal_division(m, n, h)
            + (ctx.subcore(m - 1, m, pow2=Fraction(1, 2) * Poly.var("E"), cnt=eq_counts_core_sym(m))
               if is_poly(m) else ctx.subcore(m - 1, m))
            + (m - 1) * c_unallocated_part(m, n, h)
            + c_piece_concat(m - 1, h, n)
            + 2 * c_value(h))


# ============================================================================
# allocation.py, bounds.py helpers used at run time
# ============================================================================

ALLOC_INIT = own("allocation.Allocation.__init__")
GETITEM = own("allocation.Allocation.__getitem__") + EMPTY


def c_alloc_init(k: int) -> int:
    return ALLOC_INIT + k


def c_oplus(a: int, b: int, h_cat: int, n: int, m: int, views=(), pair=None) -> int:
    """``X.oplus(Y)`` with ``|X| <= a`` and ``|Y| <= b`` agents: a dictionary copy, one
    two-piece concatenation per agent of ``Y`` (``pair``, by default
    :func:`c_piece_concat` of two pieces with the given ``views``), a new allocation."""
    if pair is None:
        pair = c_piece_concat(2, h_cat, n, views)
    return (own("allocation.Allocation.oplus", b + 1) + a
            + b * (own("pieces.OrderedPiece.__add__") + pair)
            + c_alloc_init(a + b))


def c_give(a: int, h_cat: int, n: int, m: int, views=()) -> int:
    return own("allocation.Allocation.give") + c_alloc_init(1) + c_oplus(a, 1, h_cat, n, m, views)


LN2 = Fraction(6932, 10000)          # > ln 2


def D_value(kind: str, m):
    """``D_mid`` / ``D_shortcut`` / ``D_final`` of ``Main(m)``.  For a polynomial ``m``: the
    bound ``ceil(log_{m/(m-2)} x) <= (m/2) ln x + 1`` (``ln(m/(m-2)) >= 2/m``) with
    ``x <= 4m^4``, ``m^4`` and ``4m^7`` respectively, in terms of ``B = log2 m``."""
    if not is_poly(m):
        return {"mid": B.D_mid, "shortcut": B.D_shortcut, "final": B.D_final}[kind](m)
    Bv = Poly.var("B")
    if kind == "mid":
        return LN2 * m * (1 + 2 * Bv) + 1
    if kind == "shortcut":
        return 2 * LN2 * m * Bv + 1
    return LN2 * m * (1 + Fraction(7, 2) * Bv) + 1


def digits_agent(n):
    """Decimal digits of an agent ``< n`` (agents are ``0..n-1``); for a polynomial ``n``,
    ``L + 1`` with ``L = log2 n``."""
    if is_poly(n):
        return Poly.var("L") + 1
    return digits(max(1, n - 1))


def c_ceil_log(D: int) -> int:
    """``bounds.ceil_log`` returning ``D``: ``D`` multiplications."""
    return own("bounds.ceil_log", D + 1)


def c_D(kind: str, m: int) -> int:
    """``bounds.D_mid`` / ``D_shortcut`` / ``D_final`` (with ``Q_of`` and ``ceil_log``)."""
    return own(f"bounds.D_{kind}") + own("bounds.Q_of") + c_ceil_log(D_value(kind, m))


def c_core_queries(m) -> int:
    """``bounds.core_queries(m)``: ``subcore_cuts``, ``S`` and ``E`` with one ``binom`` of at most
    ``m`` loop iterations."""
    return (own("bounds.core_queries") + own("bounds.subcore_cuts") + own("bounds.S") + own("bounds.E")
            + own("bounds.binom", (m + 1, 1)))


# ============================================================================
# simultaneous.py -- Lemma 4.6, procedure "rounding" (PROOF.md §6)
# ============================================================================

def c_kernel_vector(r: int) -> int:
    """``kernel_vector`` for ``r`` rows and ``r + 1`` columns."""
    t = (r + 1,                  # L92 copy rows
         r + 2,                  # L96 columns
         (r + 1) * (r + 1),      # L100 pivot search
         r * (r + 1) + r,        # L108 scale the pivot row
         r * r + r,              # L109 other rows
         r * r * (r + 1) + r * r,  # L113 eliminate
         r + 1)                  # L120 back substitution
    return (own("simultaneous.kernel_vector", *t) + r * (r + 1)   # list(row) copies
            + r + (r + 2) * nested(sub("simultaneous.kernel_vector", "genexpr")) + (r + 1))


def lemma46_counts(m: int) -> Dict[str, int]:
    """Counts of one Lemma 4.6 call of ``Main(m)``: requests ``K``, nonempty labels at
    most ``K``, leaves ``L = K + 1``, depth ``D``, halvings ``H`` (``bounds.lemma46_cuts``),
    rounds per split ``Hc = r ceil(log2(1/gamma))`` with ``r <= m``."""
    K = B.max_requests(m)
    if is_poly(m):
        # D = bit_length(K) <= 3B + 1 (K <= m^3); 1/delta = 16 m (Q+1) m Q^2 <= 16 m^8, so
        # m D / (2 delta) <= 8 m^9 (3B + 1) and ceil(log2) <= 4 + 9B + log2(3B + 1) + 1
        # <= 5 + 10B for B >= 2
        Bv = Poly.var("B")
        D = 3 * Bv + 1
        cl = 5 + 10 * Bv
        return {"K": K, "L": K + 1, "D": D, "H": K * m * m * cl, "Hc": m * cl, "r": m}
    delta = B.stage3_delta(m)
    D = B.lemma46_depth(K)
    H = B.lemma46_cuts(m, K, delta)
    Hc = m * B.ceil_log2(Fraction(m * D) / (2 * delta))
    return {"K": K, "L": K + 1, "D": D, "H": H, "Hc": Hc, "r": m}


def c_lemma46(m: int, n: int, h: int) -> int:
    """``simultaneous_cut`` in ``Main(m)``: ``SimultaneousCutter.__init__``, ``run`` with the
    rounding procedure and ``verify``.  ``h`` bounds the heights of the residue's tree
    and of all cells and pieces (Lemma 4.2)."""
    c = lemma46_counts(m)
    K, L, D, H, Hc, r = c["K"], c["L"], c["D"], c["H"], c["Hc"], c["r"]
    Q = m * (m - 1)
    # repr of a request key ("a", (c, t), k) or ("p", h), agents c, k < n and t <= Q^2:
    # at most RL = 13 + 2 digits(agent) + digits(Q^2) characters
    if is_poly(m, n):
        RL = 13 + 2 * digits_agent(n) + 2 * Poly.var("B") + 1
    else:
        RL = 13 + 2 * digits_agent(n) + digits(Q * Q)
    VAL = c_value(h)
    init = (own("simultaneous.SimultaneousCutter.__init__", K + 1, K + 1, m + 1, m + 1) + m
            + K * RL + c_sorted(K, RL) + K + m * VAL)
    # the label tree: L leaves, L - 1 internal nodes, depth D; the label lists of the
    # internal nodes of one level are disjoint, so their lengths sum to at most D * L
    assign = (own_calls("simultaneous.SimultaneousCutter._assign", 2 * L - 1)
              + D * L                                                   # the two slices
              + (D * L + L) * (nested(sub("simultaneous.SimultaneousCutter._assign", "genexpr"))
                               + nested(sub("simultaneous.SimultaneousCutter._assign", "genexpr", 1)))
              + D * L)                                                  # the two sums
    # the L - 1 splits together (Lemma 6.1)
    splits_n = L - 1
    cells_in = D * (1 + H)          # cells entering the splits
    reduces = cells_in + H          # each reduce step removes a cell from F
    rounds = splits_n * Hc          # rounds of halvings
    fine = (own("simultaneous.SimultaneousCutter._fine")
            + (r + 1) * nested(sub("simultaneous.SimultaneousCutter._fine", "genexpr")) + r * VAL + r)
    frac = own("simultaneous._Frac.__init__")
    halve = (own("simultaneous.SimultaneousCutter._halve", r + 1) + r * VAL + VAL
             + c_prefix_of_value(m, n, h) + 2 * frac)
    reduce_ = (own("simultaneous.SimultaneousCutter._reduce", r + 1, r * (r + 1) + r, r + 2, r + 2)
               + (r + 1) + r * (r + 1) * VAL + c_kernel_vector(r) + (r + 1))
    t = (cells_in + splits_n,                  # L262 cells into F
         rounds + splits_n,                    # L266 rounds (+ the final test)
         (rounds + splits_n) * (r + 1),        # L267 coarse cells
         rounds * (r + 1),                     # L270 fine cells
         H + rounds,                           # L271 halvings
         reduces + rounds,                     # L274 reduce steps after a round
         splits_n * (r + 1))                   # L276 rounding of F
    splits = (own_calls("simultaneous.SimultaneousCutter._split", splits_n, *t)
              + cells_in * frac
              + ((rounds + splits_n) * r + rounds * r) * fine
              + H * (halve + 2) + rounds * 2 * r      # kept.extend, F[:] = kept
              + reduces * reduce_
              + cells_in)                             # list(cells) when p is 0 or 1
    # the pieces: at most 1 + H cells reach the leaves
    leaves = 1 + H
    collect = (own_calls("simultaneous.SimultaneousCutter._collect", K + 1, leaves + K + 1)
               + leaves * nested(sub("simultaneous.SimultaneousCutter._collect", "lambda"))
               + c_sorted(leaves, 3)
               + own_calls("pieces.OrderedPiece.concat", K + 1, leaves + K + 1, leaves + K + 1, 0)
               + leaves * LEN + (leaves + K + 1) * nested(sub("pieces.OrderedPiece.concat", "genexpr")) + leaves
               + leaves * (own("seq.concat") + c_split_last(h, n) + c_join(h, n))
               + (K + 1) * (FROMTREE + EMPTY))
    rounding = (own("simultaneous.SimultaneousCutter._rounding", m + 1, K + 1, K + 1, K + 1, m + 1, K + 1)
                + K + (K + 1) * nested(sub("simultaneous.SimultaneousCutter._rounding", "genexpr")) + K
                + EMPTY + FIRST + assign + splits + collect)
    verify = own("simultaneous.SimultaneousCutter.verify", K + 1, K * m + K) + K * m * VAL
    run = (own("simultaneous.SimultaneousCutter.run") + rounding
           + own("simultaneous.SimultaneousCutter.used_queries") + verify)
    return own("simultaneous.simultaneous_cut") + init + run


# ============================================================================
# base_cases.py
# ============================================================================

def c_equal_pieces(k: int, m: int, n: int, h: int) -> int:
    return own("base_cases._equal_pieces", k) + c_value(h) + (k - 1) * c_prefix_of_value(m, n, h)


def c_argmax(L: int) -> int:
    return own("base_cases._argmax") + L + (L + 1) * nested(sub("base_cases._argmax", "genexpr")) + L


def c_base_case(m: int, n: int, h: int) -> int:
    """``base_case`` with ``m <= 3`` agents (all paths), including its envy check."""
    VAL = c_value(h)
    cost = own("base_cases.base_case", m + 1, m * m + m) + c_sorted(m) + m * VAL + m * m * VAL
    if m == 2:
        cost += (own("base_cases.cut_and_choose", 3) + c_sorted(2) + c_equal_pieces(2, m, n, h)
                 + 2 * VAL + c_argmax(2))
    elif m == 3:
        pick = 3 * (nested(sub("base_cases.selfridge_conway", "lambda")) + VAL + 2) + 3 + 3
        cost += (own("base_cases.selfridge_conway", 4, 4) + c_sorted(3) + c_equal_pieces(3, m, n, h)
                 + 3 * VAL + c_argmax(3) + 4 * nested(sub("base_cases.selfridge_conway", "genexpr")) + 3
                 + EMPTY + c_suffix_of_value(m, n, h) + 3          # list(range(3))
                 + 2 * pick + 3                                    # two choices, `in remaining`
                 + ISEMPTY + c_equal_pieces(3, m, n, h) + 3
                 + 3 * (pick + own("pieces.OrderedPiece.__add__") + c_piece_concat(2, h, n)))
    return cost


# ============================================================================
# main.py (PROOF.md §7)
# ============================================================================

class MainCost:
    """Bounds for one ``Main`` call with ``m >= 4`` agents in a table of ``n`` agents;
    ``hr`` bounds the heights of residue-derived trees and ``ha`` those of all trees
    (Lemma 4.2)."""

    def __init__(self, m: int, n: int, hr: int, ha: int) -> None:
        self.m, self.n, self.hr, self.ha = m, n, hr, ha
        self.Q = m * (m - 1)
        self.V = own("main.MainProtocol.V")
        self.VR = self.V + c_value(hr)       # values of residue-derived pieces
        self.VA = self.V + c_value(ha)       # values of other pieces
        self.core_call = own("main.MainProtocol._core") + c_core(m, n, hr) + c_core_queries(m)
        self.D_mid = D_value("mid", m)
        self.D_short = D_value("shortcut", m)
        self.D_final = D_value("final", m)

    def shrink(self, D: int) -> int:
        """``shrink`` with at most ``D`` Core calls; ``K`` has at most ``m`` agents, ``X`` has ``m``
        pieces (views of residue trees, or the cutter's tree).  ``K[a]`` is taken to be a
        tree; the at most one concatenation per agent and sequence ``H``, ``Y``, ``X`` in
        which ``K[a]`` is still a view is charged in :meth:`total` (Lemma 7.1)."""
        m, n = self.m, self.n
        it = (self.VR + self.core_call + c_alloc_init(m)
              + c_oplus(m, m, self.ha, n, m, pair=c_concat_onto_tree(self.hr, self.ha, n)))
        return own("main.MainProtocol.shrink", D + 1) + D * it + self.VR

    def record(self) -> int:
        m = self.m
        return own("main.Snapshot.record", m + 1, m * m + m) + m * m * c_value(self.hr)

    def identity(self, snaps: int) -> int:
        """``identity`` of at most ``snaps`` snapshots of ``m`` pieces each (views of residue
        trees); an iterable given as a generator is charged to the caller."""
        m, n = self.m, self.n
        return (own("main.identity", snaps + 1, snaps * m + snaps, m + 1) + snaps * m
                + m * c_piece_concat(snaps, self.ha, n, views=[(self.hr, snaps)]) + c_alloc_init(m))

    def preprocess(self) -> int:
        m, Q = self.m, self.Q
        S = m * Q * Q
        steps = m * Q ** 3                    # prescribed sequences (Lemma 4.7, checked at run time)
        nvals = m * Q * Q * m                 # bonuses d_{s,i,k} of one agent
        t = (m + 1, S + m,                    # L224/L225 snapshots
             m + 1, m * S + m, m * S * m + m * S,   # L232-L235 positive bonuses
             steps + 2, (steps + 1) * (m + 1))      # L244/L246 the while loop
        # binary search in at most nvals sorted bonuses: clog2(nvals + 1) + 1 comparisons;
        # for a polynomial m, nvals + 1 <= 2 m^6
        bisect = (6 * Poly.var("B") + 2) if is_poly(m) else clog2(nvals + 1) + 1
        return (own("main.MainProtocol.preprocess", *t) + ALLOC_INIT
                + S * (self.core_call + own("main.Snapshot.__init__") + self.record())
                + m * (c_sorted(nvals) + nvals)
                + m * S * m * own("main.Snapshot.d")
                + c_D("mid", m)
                + (steps + 1) * m * (self.VR + bisect)
                + steps * self.shrink(self.D_mid))

    def witnesses(self, Pn: int) -> int:
        """``witnesses`` over ``Pn`` snapshots: one subset test of two mark sets each."""
        return own("main.MainProtocol.witnesses", Pn + 1) + Pn * self.m

    def prune(self) -> int:
        m, Q = self.m, self.Q
        P0 = Q * Q
        it = P0                               # at most Q^2 - Q removals, plus the last pass
        t = (it + 1, it * (m + 1), it * m * (m + 1), it * Q + it, it * P0 + it)
        return own("main.MainProtocol.prune", *t) + it * m * m * self.witnesses(P0)

    def sink_component(self) -> int:
        m = self.m
        reach = own(sub("main.MainProtocol.sink_component", "reach"), m + 1, m * m + m)
        t = (m + 1, m * m + m, m + 1, m * m + m, m + 1, m + 1)
        return (own("main.MainProtocol.sink_component", *t) + m * m * m      # `i in succ[k]`
                + 2 * m * reach + 2 * m * m                                  # & and |=
                # the generator expression has two `for` clauses: every instruction runs at most
                # (outer + inner iterations + entries) <= m + m^2 + m + 1 times per component
                + m * ((m * m + 2 * m + 1) * nested(sub("main.MainProtocol.sink_component", "genexpr")) + m * m)
                + m * (nested(sub("main.MainProtocol.sink_component", "lambda")) + m) + m)

    def assert_domination(self, a: int, b: int) -> int:
        return (own("main.MainProtocol.assert_domination", a + 1, a * b + a)
                + a * (2 * self.VA + 2 * GETITEM) + a * b * (self.VA + GETITEM))

    def stage2(self) -> int:
        """``main`` from the zero-residue test to the stage-3 call: all branches together (the
        zero-residue exit, pruning, the sink test of every surviving snapshot and the
        shortcut), without the recursive call."""
        m, Q, n, hr, ha = self.m, self.Q, self.n, self.hr, self.ha
        S = m * Q * Q
        P0 = Q * Q
        cost = (m + 1) * nested(sub("main.MainProtocol.main", "genexpr")) + m * self.VR + m
        # zero-residue exit: H + Id(S), then R* to the first agent
        cost += (self.identity(S) + c_oplus(m, m, ha, n, m, views=[(hr, 2)])
                 + c_give(m, ha, n, m, views=[(hr, 2)]))
        cost += m * self.VR                                                  # eps
        cost += (m + 1) * nested(sub("main.MainProtocol.main", "genexpr", 1)) + m * self.VR
        cost += m * P0 * ((m + 1) * nested(sub("main.MainProtocol.main", "genexpr", 2))
                          + m * own("main.Snapshot.d") + m)                  # marks
        cost += self.prune()
        cost += P0 * self.sink_component()
        # the shortcut (taken at most once)
        cost += (self.identity(S) + c_oplus(m, m, ha, n, m, views=[(hr, 2)]) + c_sorted(m)
                 + m * (self.VR + c_D("shortcut", m) + self.shrink(self.D_short))
                 + c_sorted(m) + self.assert_domination(m, m)
                 + c_oplus(m, m, ha, n, m, views=[(ha, 2)]))
        return cost

    def matching_through(self) -> int:
        m, n = self.m, self.n
        t = (m + 1, m + 1, m * m + m, m + 1, m * m + m, m + 1, m + 1)
        d = digits_agent(n)                   # agents are the integers 0..n-1
        return (own("main.MainProtocol.matching_through", *t) + m * m       # queue.pop(0)
                + 2 * (m * d + c_sorted(m, d)) + m * d)

    def ranks(self) -> int:
        m, Q = self.m, self.Q
        P0 = Q * Q
        cl = m                                # profile classes
        sb = (own(sub("main.MainProtocol.ranks", "strictly_below")) + P0 * m
              + (P0 + 1) * nested(sub(sub("main.MainProtocol.ranks", "strictly_below"), "genexpr")) + P0 * m + P0)
        return (own("main.MainProtocol.ranks", cl + 1, cl + 1)
                + (cl + cl * cl) * own(sub("main.MainProtocol.ranks", "longest"), cl + 1)
                + cl * cl * sb)

    def check_lemma_4_14(self) -> int:
        m, Q = self.m, self.Q
        t = (Q + 1, Q * m + Q, Q * m * m + Q * m, m + 1, m * m + m)
        return (own("main.MainProtocol.check_lemma_4_14", *t)
                + Q * m * m * (self.VR + own("main.Snapshot.d")) + m * m * self.VR)

    def stage3(self) -> int:
        m, Q, n, hr, ha = self.m, self.Q, self.n, self.hr, self.ha
        S = m * Q * Q
        P0 = Q * Q
        K = B.max_requests(m)
        cl = m
        t = (m + 1,                       # L327 profile
             m + 1, m * cl + m,           # L329/L330 classes
             cl + 1,                      # L338 stats
             m + 1, m * m + m,            # L347/L348 pairs
             m * m * P0 + m * m,          # L351 unused witnesses
             P0 + 1,                      # L359 padding
             Q * m + Q,                   # L364 identity matchings
             cl + 1 + m,                  # L371 rank
             m + 1,                       # L380 VR
             Q + 1, Q * m + Q, Q * m * m + Q * m,   # L382-L385 requests
             m + 1,                       # L391 rank requests
             Q + 1 + Q * m,               # L404 A
             m + 1,                       # L405 Ppiece
             m + 1, m + 1, m + 1,         # L411-L413 D, Ah, B
             cl + 1,                      # L418 classes
             2 * m, 2 * m,                # L420/L421 W, q
             2 * m, m * m + m,            # L422/L423 checks
             2 * m,                       # L426 pieces
             2 * m,                       # L429 results
             m + 1,                       # L441 R1
             m + 1, m * m + m,            # L444/L445 envy check
             m + 1,                       # L452 values_R1
             m + 1,                       # L453 shrinks
             m + 1)                       # L458 dominators
        cost = own("main.MainProtocol.stage3", *t)
        cost += m * ((P0 + 1) * nested(sub("main.MainProtocol.stage3", "genexpr")) + P0)   # profiles
        cost += m * cl * P0 * m + m                                   # profile comparisons, list(E)
        preceq = (own(sub("main.MainProtocol.stage3", "preceq"))
                  + (P0 + 1) * nested(sub(sub("main.MainProtocol.stage3", "preceq"), "genexpr")) + P0 * m + P0)
        cost += m * m * preceq
        cost += m * m * (self.witnesses(P0) + 2 * m + self.matching_through())
        cost += c_sorted(Q) + Q * nested(sub("main.MainProtocol.stage3", "lambda"))
        cost += self.ranks()
        cost += m + (m + 1) * nested(sub("main.MainProtocol.stage3", "genexpr", 1)) + m
        cost += m * self.VR                                           # VR
        cost += Q * m * m * own("main.Snapshot.d")                    # requests
        cost += K                                                     # sum(requests)
        cost += (own("bounds.lemma46_queries") + own("bounds.lemma46_cuts") + own("bounds.lemma46_depth")
                 + own("bounds.ceil_log2", 3, 3))
        cost += c_lemma46(m, n, hr)
        cost += m * EMPTY
        cost += (Q * m + 1) * nested(sub("main.MainProtocol.stage3", "genexpr", 2)) + Q * m * ISEMPTY + Q * m
        cost += (m + 1) * nested(sub("main.MainProtocol.stage3", "genexpr", 3)) + m * ISEMPTY + m
        cost += ((Q + 1) * nested(sub("main.MainProtocol.stage3", "genexpr", 4))
                 + Q * ((m + 1) * nested(sub(sub("main.MainProtocol.stage3", "genexpr", 4), "genexpr")) + m) + Q)
        cost += self.check_lemma_4_14()
        cost += m * ((Q + 1) * nested(sub("main.MainProtocol.stage3", "genexpr", 5))
                     + c_piece_concat(Q, ha, n, views=[(hr, Q)]))         # D_h
        cost += m * ((Q + 1) * nested(sub("main.MainProtocol.stage3", "genexpr", 6))
                     + c_piece_concat(Q, ha, n))                          # A_h
        cost += m * ((Q + 1) * nested(sub("main.MainProtocol.stage3", "genexpr", 7))
                     + Q * own("main.Snapshot.b") + Q)                       # B
        cost += ALLOC_INIT                                                   # Z
        # one SubCore call per profile class (sizes e summing to m), Lemma 4.17
        ctx = Ctx(n, m, ha)
        cost += m * c_piece_concat(3, ha, n)                              # W_h
        cost += m * m * self.VA                                              # V_i(W_h) >= q_i
        cost += cl * EMPTY + 2 * m
        cost += ctx.class_subcores(m, pow2=(Poly.var("E") if is_poly(m) else None))
        cost += m * (5 * LEN + 2 * FIRST + IVEQ + c_give(m, ha, n, m)
                     + c_unallocated_part(m, n, ha, h_view=hr))
        cost += m + 1 + c_piece_concat(m + 1, ha, n)                      # R1
        cost += ((S + 1) * nested(sub("main.MainProtocol.stage3", "genexpr", 8)) + self.identity(S)
                 + c_oplus(m, m, ha, n, m, views=[(hr, 2)])
                 + c_oplus(m, m, ha, n, m, views=[(ha, 2)]))                  # X
        cost += m * m * 2 * (GETITEM + self.VA)                              # envy check
        cost += cl * (nested(sub("main.MainProtocol.stage3", "lambda", 1)) + 2) + cl
        cost += m * self.VA                                                  # values_R1
        cost += m * (m + c_D("final", m) + self.shrink(self.D_final))        # the shrinks
        cost += m * m + self.assert_domination(m, m)
        cost += c_oplus(m, m, ha, n, m, views=[(ha, 2)])                      # X + Main(E*, U)
        return cost

    def header(self) -> int:
        m, Q = self.m, self.Q
        return (own("main.MainProtocol.main", m + 1, m * Q * Q + 1, Q * Q + 1, Q * Q * m + Q * Q,
                    Q * Q + 1, m + 1, m + 1)
                + c_sorted(m) + c_set_active(m))

    def total(self) -> int:
        """``Lambda_op(m)``: one ``Main`` call with ``m`` agents, without its recursive call.
        The last term: in each of the sequences ``H``, ``Y``, ``X`` of ``shrink`` calls, at
        most one concatenation per agent appends to a ``K[a]`` that is still a suffix view
        (of a tree of height ``<= ha``)."""
        return (self.header() + self.preprocess() + self.stage2() + self.stage3()
                + 3 * self.m * c_view_join(self.ha, self.ha, self.n))


def max_class_split(m: int, f) -> int:
    """The maximum of ``sum(f(e) for e in E)`` over the partitions ``E`` of ``m`` into at
    least two parts (the sizes of the profile classes), by dynamic programming over
    the part sizes (with :func:`pmax`, an upper bound if ``f`` is a polynomial)."""
    best = [0] * (m + 1)
    for total in range(1, m + 1):
        best[total] = pmax(*[f(e) + best[total - e] for e in range(1, total + 1)])
    # at least two classes: the largest class has at most m - 1 agents
    return pmax(*[f(e) + best[m - e] for e in range(1, m)])


# ============================================================================
# sizes and heights (Lemma 4.2)
# ============================================================================

def core_calls(m: int) -> int:
    """Core calls of one ``Main(m)`` call (all stages): ``m Q^2`` snapshots, ``m Q^3 D_mid``
    in stage 1, ``m D_shortcut`` in stage 2, ``m D_final`` in stage 3."""
    Q = m * (m - 1)
    return m * Q * Q + m * Q ** 3 * D_value("mid", m) + m * D_value("shortcut", m) + m * D_value("final", m)


@lru_cache(maxsize=None)
def residue_max(n: int) -> int:
    """``R_max``: components of any residue during a run on ``n`` agents."""
    total = 1
    for m in range(4, n + 1):
        c = lemma46_counts(m)
        total += (m - 1) * core_calls(m) + c["H"] + m
    return total


def size_res(n: int) -> int:
    """``C_res``: nodes of any residue-derived tree."""
    return residue_max(n) + n + 8


@lru_cache(maxsize=None)
def atoms_max(n: int) -> int:
    """``A_max``: residue-derived pieces concatenated into one piece of the protocol."""
    return 2 + sum(core_calls(m) + 2 * m * (m - 1) + 2 for m in range(4, n + 1))


def size_all(n: int) -> int:
    """``C_all``: nodes of any tree (a concatenation of at most ``A_max`` residue-derived
    pieces)."""
    return atoms_max(n) * size_res(n)


def heights(n: int):
    """``(h_res, h_all)``."""
    from .seq import max_height
    return max_height(size_res(n)), max_height(size_all(n))


# ============================================================================
# the whole run
# ============================================================================

def c_top(n: int) -> int:
    """``run_protocol`` outside ``main``: the protocol object, the table, the cake."""
    return (own("main.run_protocol") + own("main.MainProtocol.__init__") + c_sorted(n)
            + c_value_table_init(n) + own("main.MainProtocol.run") + c_register_cake(n) + CAKE
            + c_cake_tree(n))


def c_main_base(b: int, n: int, hr=None, ha=None) -> int:
    """``main`` with ``b <= 3`` agents: the header and the base case."""
    if ha is None:
        hr, ha = heights(n)
    header = own("main.MainProtocol.main", 0, 0, 0, 0, 0, 0, 0) + c_sorted(b) + c_set_active(b)
    return header + c_base_case(b, n, ha) + c_alloc_init(b)


@lru_cache(maxsize=None)
def lambda_op(m: int, n: int) -> int:
    hr, ha = heights(n)
    return MainCost(m, n, hr, ha).total()


@lru_cache(maxsize=None)
def operation_bound(n: int) -> int:
    """``U(n)``: an upper bound on the operations of ``run_protocol`` with ``n`` agents."""
    if n < 1:
        raise ValueError("need at least one agent")
    base = max(c_main_base(b, n) for b in range(1, min(n, 3) + 1))
    return c_top(n) + base + sum(lambda_op(m, n) for m in range(4, n + 1))


# ============================================================================
# the constant k (PROOF.md §8)
# ============================================================================

import math as _math

PHI_INV_LOG = Fraction(14405, 10000)     # >= 1 / log2(golden ratio) = 1.44042...
N_TAIL = 16                              # the tail bound needs log2 n >= 4


def log2_up(x: int) -> Fraction:
    """A rational upper bound on ``log2(x)`` (exact for powers of two)."""
    if x & (x - 1) == 0:
        return Fraction(x.bit_length() - 1)
    return Fraction(_math.log2(x)) + Fraction(1, 10 ** 9)


def _sum_over_m(p: Poly) -> Poly:
    """An upper bound, as a polynomial in ``n``, ``N = 2^n``, ``M = C(2n, n)`` and
    ``L = log2 n``, on ``sum_{m=4}^{n} p(m)`` for a polynomial ``p`` in ``m``, ``E = 2^m``,
    ``G = C(2m, m)`` and ``B = log2 m`` (other variables kept):
    ``sum m^a B^c <= n^(a+1) L^c``, ``sum m^a 2^m B^c <= 2 n^a L^c 2^n`` and
    ``sum m^a G_m B^c <= (7/5) n^a L^c M`` (Lemma 8.1)."""
    out = Poly()
    for mono, c in p.positive().terms.items():
        d = dict(zip(VARS, mono))
        a, b, g, cc = d.pop("m"), d.pop("E"), d.pop("G"), d.pop("B")
        if b + g > 1:
            raise ValueError("a term with 2^(2m), C(2m, m)^2 or 2^m C(2m, m)")
        rest = Poly({tuple(d.get(v, 0) for v in VARS): c})
        if b == 0 and g == 0:
            out = out + rest * Poly.var("n") ** (a + 1) * Poly.var("L") ** cc
        elif b == 1:
            out = out + 2 * rest * Poly.var("n") ** a * Poly.var("L") ** cc * Poly.var("N")
        else:
            out = out + Fraction(7, 5) * rest * Poly.var("n") ** a * Poly.var("L") ** cc * Poly.var("M")
    return out


def _log2_bound(p: Poly) -> Poly:
    """For a polynomial ``p`` in ``n`` and ``L`` with nonnegative coefficients: an upper
    bound on ``log2(p + 1)`` as a polynomial in ``L``, valid for ``n >= 16``: every
    monomial is at most ``n^a_max L^b_max``, ``log2 L <= L / 2`` for ``L >= 4``."""
    p = p.positive() + 1
    total = sum(p.terms.values())
    a_max = p.degree("n")
    b_max = p.degree("L")
    return log2_up(_math.ceil(total)) + (a_max + Fraction(b_max, 2)) * Poly.var("L")


@lru_cache(maxsize=None)
def tail_heights():
    """``(HR, HA)``: polynomials in ``L = log2 n`` bounding ``h_res(n)`` and ``h_all(n)`` for
    ``n >= 16`` (Lemma 8.2): ``max_height(C) <= 1.4405 log2(C + 1)``."""
    m = Poly.var("m")
    per_m = (m - 1) * core_calls(m) + lemma46_counts(m)["H"] + m
    size_res_p = 1 + _sum_over_m(per_m) + Poly.var("n") + 8
    atoms_p = 2 + _sum_over_m(core_calls(m) + 2 * m * (m - 1) + 2)
    HR = PHI_INV_LOG * _log2_bound(size_res_p)
    HA = PHI_INV_LOG * (_log2_bound(atoms_p) + _log2_bound(size_res_p))
    return HR, HA


@lru_cache(maxsize=None)
def tail_majorant() -> Poly:
    """A polynomial in ``n``, ``N = 2^n`` and ``L = log2 n`` that is at least ``U(n)`` for
    every ``n >= 16`` (Lemma 8.3)."""
    m, n = Poly.var("m"), Poly.var("n")
    hr, ha = Poly.var("hr"), Poly.var("ha")
    lam = MainCost(m, n, hr, ha).total()
    total = _sum_over_m(lam)
    base = pmax(*[c_main_base(b, n, hr, ha) for b in (1, 2, 3)])
    total = total + c_top_sym(n) + base
    HR, HA = tail_heights()
    # substitute the height bounds (all coefficients are nonnegative after positive())
    out = Poly()
    for mono, c in total.positive().terms.items():
        d = dict(zip(VARS, mono))
        e_r, e_a = d.pop("hr"), d.pop("ha")
        rest = Poly({tuple(d.get(v, 0) for v in VARS): c})
        out = out + rest * HR ** e_r * HA ** e_a
    return out


def isqrt_down(x: int) -> int:
    """``floor(sqrt(x))`` for an integer ``x >= 0``."""
    return _math.isqrt(x)


def ratio_bound(majorant: Poly, n0: int, d: int = None):
    """``(r, ok)``: if ``ok``, ``r`` bounds ``majorant(n) / (n^d 4^n)`` for every ``n >= n0``
    (``d = OP_D`` by default).  ``M = C(2n, n) <= 4^n / sqrt(3n + 1)`` (Lemma 8.4), so a
    monomial ``c n^a M L^b`` contributes at most ``c n^(a-d-1/2) L^b / sqrt(3)``, which is
    nonincreasing on ``[n0, oo)`` if ``a <= d`` and ``ln n0 >= b / (d + 1/2 - a)``; a monomial
    ``c n^a N L^b`` contributes ``c n^(a-d) L^b / 2^n``, and ``c n^a L^b`` contributes
    ``c n^(a-d) L^b / 4^n``, both nonincreasing once ``max(0, a - d) + b <= 0.69 n0``.  The
    sum of the values at ``n0`` (with ``1/sqrt(3 n0) <= 1/floor(sqrt(3 n0))``) bounds the
    ratio."""
    if d is None:
        d = OP_D
    if n0 < N_TAIL:
        raise ValueError("the tail bound needs n0 >= 16")
    L0 = log2_up(n0)
    inv_sqrt = Fraction(1, isqrt_down(3 * n0))
    r = Fraction(0)
    ok = True
    for mono, c in majorant.terms.items():
        dd = dict(zip(VARS, mono))
        a, nb, mb, cl = dd["n"], dd["N"], dd["M"], dd["L"]
        if c < 0 or any(e for v, e in dd.items() if v not in ("n", "N", "M", "L")):
            raise ValueError("unexpected term in the majorant")
        if nb + mb > 1:
            ok = False
            continue
        if mb == 1:
            if a > d:
                ok = False                                   # not O(n^d 4^n)
            elif cl and _math.log(n0) < cl / (d + Fraction(1, 2) - a):
                ok = False                                   # not yet decreasing at n0
            r += c * Fraction(n0) ** (a - d) * L0 ** cl * inv_sqrt
        elif nb == 1:
            if (max(0, a - d) + cl) > Fraction(69, 100) * n0:
                ok = False
            r += c * Fraction(n0) ** (a - d) * L0 ** cl / 2 ** n0
        else:
            if (max(0, a - d) + cl) > Fraction(69, 100) * n0:
                ok = False
            r += c * Fraction(n0) ** (a - d) * L0 ** cl / 4 ** n0
    return r, ok


def tail_ratio(n0: int):
    """:func:`ratio_bound` of the operation majorant."""
    return ratio_bound(tail_majorant(), n0, OP_D)


@lru_cache(maxsize=None)
def query_majorant() -> Poly:
    """A polynomial in ``n``, ``N = 2^n``, ``M = C(2n, n)``, ``L = log2 n`` that is at least
    ``bounds.query_bound(n)`` for every ``n >= 4``: ``Lambda_k`` with ``S(k - 1) <= 2^k``,
    ``S(k) <= 2^(k+1)``, ``E(k - 1, k) <= k^2 G_{k-2} <= (3/35) k^2 G`` and
    ``E(k, k) = (k-1) k G_{k-1} / 2 <= (1/7) k^2 G`` (``G = C(2k, k)``), and the Lemma 4.6
    cuts as in :func:`lemma46_counts`, summed over ``k`` (Lemma 8.1)."""
    k = Poly.var("m")
    E = Poly.var("E")
    G = Poly.var("G")
    core_q = k * (k - 1 + E + Fraction(3, 35) * k * k * G)
    lam = (core_q * core_calls(k) + k * lemma46_counts(k)["H"]
           + k * (2 * E + Fraction(1, 7) * k * k * G))
    return _sum_over_m(lam) + Poly.var("n") + 15


def c_top_sym(n):
    return (own("main.run_protocol") + own("main.MainProtocol.__init__") + c_sorted(n)
            + c_value_table_init(n) + own("main.MainProtocol.run") + c_register_cake(n) + CAKE
            + c_cake_tree(n))


@lru_cache(maxsize=None)
def lambda_poly(m: int) -> Poly:
    """``Lambda_op(m)`` as a polynomial in ``n``, ``L``, ``hr`` and ``ha`` (an upper bound on the
    integer formula, evaluated symbolically in the table size and the heights)."""
    return MainCost(m, Poly.var("n"), Poly.var("hr"), Poly.var("ha")).total()


def operation_bound_poly(n: int) -> Fraction:
    """``U(n)`` with every ``Lambda_op(m)`` taken from :func:`lambda_poly` (at least
    :func:`operation_bound`, fast for many ``n``)."""
    hr, ha = heights(n)
    base = max(c_main_base(b, n) for b in range(1, min(n, 3) + 1))
    L = log2_up(n)
    return c_top(n) + base + sum(lambda_poly(m).evaluate(n=n, L=L, hr=hr, ha=ha) for m in range(4, n + 1))


#: ``U(n) <= OP_K n^OP_D 4^n`` for every ``n >= 1`` (equality at ``n = 1``), and
#: ``U(n) <= OP_K2 n^OP_D 4^n`` for every ``n >= 2`` (equality at ``n = 4``): the smallest
#: constants for this bound (Theorem 8.5; checked exactly for ``n <= OP_N_EXACT``, by
#: :func:`tail_ratio` beyond).  ``OP_D = 11`` is the smallest integer exponent.
OP_D = 11
OP_K = Fraction(392)
OP_K2 = Fraction(161700700527, 1073741824)
OP_N_EXACT = 40
QUERY_N_EXACT = 399


def check_operation_constant(n_exact: int = OP_N_EXACT) -> Fraction:
    """Verify Theorem 8.5: ``U(n) <= OP_K n^11 4^n`` for every ``n``, ``<= OP_K2 n^11 4^n`` for
    ``n >= 2``, exactly for ``1 <= n <= n_exact`` and by the majorant beyond; returns the
    largest ratio for ``n >= 2``."""
    worst = Fraction(0)
    for n in range(1, n_exact + 1):
        r = Fraction(operation_bound(n), n ** OP_D * 4 ** n)
        if r > OP_K or (n >= 2 and r > OP_K2):
            raise AssertionError(f"U({n}) / (n^11 4^n) = {float(r)} too large")
        if n >= 2:
            worst = max(worst, r)
    r, ok = tail_ratio(n_exact + 1)
    if not ok or r > OP_K2:
        raise AssertionError(f"tail bound {float(r)} (ok={ok}) does not certify the constants")
    return worst


def check_query_constant(n_exact: int = QUERY_N_EXACT) -> Fraction:
    """Verify ``bounds.query_bound(n) <= n^11 4^n / 4`` for all ``n``, and
    ``<= MONOMIAL_C2 n^11 4^n`` for ``n >= 2`` (``PROOF.md`` §9), exactly for
    ``n <= n_exact`` and by the majorant beyond; returns the largest ratio for ``n >= 2``."""
    worst = Fraction(0)
    for n in range(1, n_exact + 1):
        r = Fraction(B.query_bound(n), n ** B.MONOMIAL_K * 4 ** n)
        if r > B.MONOMIAL_C or (n >= 2 and r > B.MONOMIAL_C2):
            raise AssertionError(f"T({n}) too large")
        if n >= 2:
            worst = max(worst, r)
    r, ok = ratio_bound(query_majorant(), n_exact + 1, B.MONOMIAL_K)
    if not ok or r > B.MONOMIAL_C2:
        raise AssertionError("query tail bound fails")
    return worst
