"""Pieces of cake: half-open intervals and *ordered pieces* (Appendix C.1).

A piece is a finite union of half-open intervals ``[a, b)`` with rational
endpoints.  Following Section C.1 of Ye–Bai, whenever the protocol divides a
piece with several interval components it stores the components in a *fixed
order*; a prefix / suffix is taken with respect to that order, which need not be
the left-to-right order on the cake.  ``[P, A, D]`` denotes the concatenation of
ordered pieces (components of ``P`` first, then ``A``, then ``D``).

Overlaps at single points (cut endpoints) have measure zero for every nonatomic
valuation and are ignored, exactly as in the paper ("up to cut endpoints").
"""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Iterator, Optional, Sequence, Tuple

from . import seq as _seq

Rational = Fraction

ZERO = Fraction(0)
ONE = Fraction(1)


class Interval:
    """Nonempty half-open interval ``[a, b)`` with ``0 <= a < b <= 1``."""

    __slots__ = ("a", "b", "_hash")

    def __init__(self, a: Rational, b: Rational) -> None:
        a = Fraction(a)
        b = Fraction(b)
        if not (a < b):
            raise ValueError(f"empty or reversed interval [{a}, {b})")
        self.a = a
        self.b = b
        self._hash = None

    @property
    def length(self) -> Fraction:
        return self.b - self.a

    def key(self) -> Tuple[Fraction, Fraction]:
        return (self.a, self.b)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Interval) and self.a == other.a and self.b == other.b

    def __hash__(self) -> int:
        h = self._hash
        if h is None:
            h = self._hash = hash((self.a, self.b))
        return h

    def __repr__(self) -> str:
        return f"[{self.a}, {self.b})"

    def split(self, y: Rational) -> Tuple["Interval", "Interval"]:
        """Split at an interior point ``a < y < b``."""
        y = Fraction(y)
        if not (self.a < y < self.b):
            raise ValueError(f"split point {y} not interior to {self}")
        return Interval(self.a, y), Interval(y, self.b)


class OrderedPiece:
    """A piece stored as a sequence of pairwise disjoint components in a fixed order.

    Two representations, interchangeable from the outside:

    * a tuple of components (``OrderedPiece(components)``; used by tests and by
      code outside the protocol's inner loops), or
    * a persistent balanced tree (:mod:`cake_cutting.seq`) owned by a
      :class:`~cake_cutting.bookkeeping.ValueTable`, whose nodes also carry the
      agents' values; the protocol creates its pieces this way so that taking a
      prefix or suffix, concatenating, and evaluating cost ``O(n log C)`` instead of
      ``O(C)`` for ``C`` components.

    A tree-backed piece may be a *suffix view* of its tree: the components from
    number ``k0`` on, the first of them possibly replaced by a suffix ``head`` of
    itself.  Views cost ``O(1)`` to create and are what SubCore works with; they
    are turned into trees only where pieces are concatenated (``O(n log C)``).

    ``components`` materializes the tuple on first use (``O(C)``) and keeps it.
    The empty piece is ``OrderedPiece(())``.  Components are never empty.
    """

    __slots__ = ("_comps", "_tree", "_owner", "_hash", "_k0", "_head", "_first", "_vk0", "_vcache")

    def __init__(self, components: Iterable[Interval] = ()) -> None:
        comps = tuple(components)
        for c in comps:
            if not isinstance(c, Interval):
                raise TypeError("components must be Interval objects")
        self._comps = comps
        self._tree = None
        self._owner = None
        self._hash = None
        self._k0 = 0
        self._head = None
        self._first = None
        self._vk0 = None
        self._vcache = None

    @staticmethod
    def _from_tree(tree, owner, comps: Optional[Tuple[Interval, ...]] = None) -> "OrderedPiece":
        """A piece backed by ``tree`` (``None`` = empty) of the value table ``owner``."""
        p = OrderedPiece.__new__(OrderedPiece)
        p._hash = None
        p._k0 = 0
        p._head = None
        p._vk0 = None
        p._vcache = None
        if tree is None:
            p._comps = ()
            p._tree = None
            p._owner = None
            p._first = None
        else:
            p._comps = comps
            p._tree = tree
            p._owner = owner
            p._first = tree.first
        return p

    @staticmethod
    def _view(tree, owner, k0: int, head: Optional[Interval], first: Interval, vk0: tuple) -> "OrderedPiece":
        """The suffix of ``tree`` from component ``k0`` on, that component replaced by its
        suffix ``head`` if ``head`` is not ``None``; ``first`` is the view's first component
        and ``vk0`` the value vector of component ``k0`` of ``tree``."""
        p = OrderedPiece.__new__(OrderedPiece)
        p._comps = None
        p._tree = tree
        p._owner = owner
        p._hash = None
        p._k0 = k0
        p._head = head
        p._first = first
        p._vk0 = vk0
        p._vcache = None
        return p

    @property
    def is_view(self) -> bool:
        return self._k0 > 0 or self._head is not None

    # -- constructors -----------------------------------------------------
    @staticmethod
    def interval(a: Rational, b: Rational) -> "OrderedPiece":
        return OrderedPiece((Interval(a, b),))

    @staticmethod
    def cake() -> "OrderedPiece":
        return OrderedPiece.interval(ZERO, ONE)

    @staticmethod
    def empty() -> "OrderedPiece":
        return OrderedPiece(())

    @staticmethod
    def concat(pieces: Iterable["OrderedPiece"]) -> "OrderedPiece":
        """``[P_1, P_2, ...]`` -- components of ``P_1`` first, then ``P_2``, ...

        Tree-backed pieces of one value table are joined in ``O(log C)`` node
        creations each (a suffix view by one split of its tree and one join, a tree by
        one concatenation); otherwise the components are copied."""
        nonempty = [p for p in pieces if len(p) > 0]
        if not nonempty:
            return OrderedPiece.empty()
        if len(nonempty) == 1:
            return nonempty[0]
        owner = nonempty[0]._owner
        if owner is not None and all(p._tree is not None and p._owner is owner for p in nonempty):
            t = None
            for p in nonempty:
                if p._k0 or p._head is not None:
                    _, (iv, vals), right = _seq.split3(p._tree, p._k0)
                    if p._head is not None:
                        iv, vals = p._head, owner.vals_of(p._head)
                    t = _seq.join(t, iv, vals, right)
                else:
                    t = _seq.concat(t, p._tree)
            return OrderedPiece._from_tree(t, owner)
        comps: list = []
        for p in nonempty:
            comps.extend(p.components)
        return OrderedPiece(comps)

    # -- basic queries ----------------------------------------------------
    @property
    def components(self) -> Tuple[Interval, ...]:
        c = self._comps
        if c is None:
            lst = _seq.to_list(self._tree)[self._k0:]
            if self._head is not None:
                lst[0] = self._head
            c = self._comps = tuple(lst)
        return c

    def __len__(self) -> int:
        if self._tree is not None:
            return self._tree.size - self._k0
        return len(self._comps)

    @property
    def first(self) -> Interval:
        """First component (``IndexError`` for the empty piece)."""
        if self._tree is not None:
            return self._first
        return self._comps[0]

    @property
    def last(self) -> Interval:
        """Last component (``IndexError`` for the empty piece).  For a one-component
        view with a ``head`` that is the head (its end is that of the tree's last)."""
        if self._tree is not None:
            if self._head is not None and self._k0 == self._tree.size - 1:
                return self._head
            return self._tree.last
        return self._comps[-1]

    def __iter__(self) -> Iterator[Interval]:
        return iter(self.components)

    def __bool__(self) -> bool:
        return len(self) > 0

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @property
    def length(self) -> Fraction:
        """Lebesgue measure of the piece."""
        return sum((c.length for c in self.components), ZERO)

    def __add__(self, other: "OrderedPiece") -> "OrderedPiece":
        return OrderedPiece.concat([self, other])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OrderedPiece):
            return False
        if self._tree is not None and self._tree is other._tree and self._k0 == other._k0 \
                and self._head == other._head:
            return True
        if len(self) != len(other):
            return False
        return self.components == other.components

    def __hash__(self) -> int:
        h = self._hash
        if h is None:
            h = self._hash = hash(self.components)
        return h

    def __repr__(self) -> str:
        return "OrderedPiece(" + ", ".join(repr(c) for c in self.components) + ")"

    # -- slicing in the fixed order ---------------------------------------
    def prefix_at(self, k: int) -> "OrderedPiece":
        """Components strictly before boundary ``k`` (``0 <= k <= len``)."""
        if self._tree is not None:
            a, _ = _seq.split(self._owner.full_tree(self), k)
            return OrderedPiece._from_tree(a, self._owner)
        return OrderedPiece(self._comps[:k])

    def suffix_at(self, k: int) -> "OrderedPiece":
        if self._tree is not None:
            _, b = _seq.split(self._owner.full_tree(self), k)
            return OrderedPiece._from_tree(b, self._owner)
        return OrderedPiece(self._comps[k:])

    # -- set-like helpers (used by the protocol for residues and by tests) --
    def normalized(self) -> Tuple[Interval, ...]:
        """Components sorted left-to-right with touching ones merged."""
        comps = sorted(self.components, key=Interval.key)
        merged: list = []
        for c in comps:
            if merged and merged[-1].b >= c.a:
                if merged[-1].b < c.b:
                    merged[-1] = Interval(merged[-1].a, max(merged[-1].b, c.b))
            else:
                merged.append(c)
        return tuple(merged)

    def normalized_length(self) -> Fraction:
        """Measure of the piece as a set (components may be listed in any order)."""
        return sum((c.length for c in self.normalized()), ZERO)

    def pretty(self) -> str:
        comps = self.normalized()
        if not comps:
            return "{}"
        return " u ".join(f"[{c.a}, {c.b})" for c in comps)


# ---------------------------------------------------------------------------
# Measure-theoretic helpers (exact).  They operate on normalized component lists
# and are used by the protocol only to form residues; the tests use them to
# certify completeness / disjointness of allocations.
# ---------------------------------------------------------------------------

def overlap_length(p: OrderedPiece, q: OrderedPiece) -> Fraction:
    """Measure of ``p ∩ q``."""
    total = ZERO
    pc = p.normalized()
    qc = q.normalized()
    i = j = 0
    while i < len(pc) and j < len(qc):
        lo = max(pc[i].a, qc[j].a)
        hi = min(pc[i].b, qc[j].b)
        if lo < hi:
            total += hi - lo
        if pc[i].b < qc[j].b:
            i += 1
        else:
            j += 1
    return total


def pairwise_disjoint(pieces: Sequence[OrderedPiece]) -> bool:
    """True iff the pieces overlap only in measure-zero sets."""
    events = []
    for p in pieces:
        for c in p.components:
            events.append((c.a, c.b))
    events.sort()
    end = None
    for a, b in events:
        if end is not None and a < end:
            return False
        end = b if end is None else max(end, b)
    return True


def union_measure(pieces: Sequence[OrderedPiece]) -> Fraction:
    """Measure of the union of the pieces."""
    return OrderedPiece.concat(pieces).normalized_length()


def difference(p: OrderedPiece, qs: Sequence[OrderedPiece]) -> OrderedPiece:
    """``p \\ (q_1 ∪ ... ∪ q_k)`` keeping the components of ``p`` in their order.

    Each component of ``p`` is replaced by the (ordered, left-to-right) list of
    its sub-intervals that are not covered by any ``q``.
    """
    cuts = OrderedPiece.concat(qs).normalized()
    out = []
    for c in p.components:
        cur = c.a
        for d in cuts:
            if d.b <= cur:
                continue
            if d.a >= c.b:
                break
            if d.a > cur:
                out.append(Interval(cur, min(d.a, c.b)))
            cur = max(cur, d.b)
            if cur >= c.b:
                break
        if cur < c.b:
            out.append(Interval(cur, c.b))
    return OrderedPiece(out)
