"""Persistent AVL sequences of interval components with per-agent subtree sums.

An ordered piece with ``C`` components is stored as a height-balanced binary
tree whose in-order traversal is the component order.  Every node stores its
component, the component's values for the agents of the value table (a tuple
indexed by agent number, ``None`` for a value that was never recorded), the
subtree's height, size, first and last component, and the subtree's value for
each agent (``None`` if a value in the subtree is unknown), computed when the
node is created from its children's.  Nodes are never modified after
construction, so trees can be shared between pieces (persistence): splitting
or concatenating creates ``O(log C)`` new nodes and leaves the old trees intact.

The join-based algorithms are those of Blelloch, Ferizovic and Sun, *Just Join
for Parallel Ordered Sets* (SPAA 2016), for AVL trees: ``join`` of two trees
and a middle element costs ``O(|h(L) - h(R)| + 1)`` node creations, ``split`` by
position ``O(h)``.  The AVL invariant (heights of siblings differ by at most
one) bounds the height of a tree with ``C`` nodes by ``1.4405 log2(C + 2)``.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

ZERO = Fraction(0)


class _Unset:
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET = _Unset()


class Node:
    __slots__ = ("iv", "vals", "left", "right", "h", "size", "first", "last", "sums")

    def __init__(self, left: Optional["Node"], iv, vals: tuple, right: Optional["Node"]) -> None:
        self.iv = iv
        self.vals = vals
        self.left = left
        self.right = right
        hl = left.h if left is not None else 0
        hr = right.h if right is not None else 0
        self.h = (hl if hl > hr else hr) + 1
        self.size = (left.size if left is not None else 0) + 1 + (right.size if right is not None else 0)
        self.first = left.first if left is not None else iv
        self.last = right.last if right is not None else iv
        # subtree values per agent: one pass over the agents (Lemma 2.1 of PROOF.md)
        if left is None:
            if right is None:
                self.sums = vals
            else:
                self.sums = [None if (v is None or r is None) else v + r
                             for v, r in zip(vals, right.sums)]
        elif right is None:
            self.sums = [None if (u is None or v is None) else u + v
                         for u, v in zip(left.sums, vals)]
        else:
            self.sums = [None if (u is None or v is None or r is None) else u + v + r
                         for u, v, r in zip(left.sums, vals, right.sums)]

    def sum(self, i: int):
        """Value of the subtree for agent number ``i`` (``None`` if some value is unknown)."""
        return self.sums[i]


def min_size(h: int) -> int:
    """Fewest nodes of an AVL tree of height ``h``: ``F(h + 2) - 1`` (Fibonacci)."""
    a, b = 0, 1
    for _ in range(h + 2):
        a, b = b, a + b
    return a - 1


def max_height(size: int) -> int:
    """Largest possible height of an AVL tree with ``size >= 1`` nodes."""
    h = 0
    while min_size(h + 1) <= size:
        h += 1
    return h


def height(t: Optional[Node]) -> int:
    return t.h if t is not None else 0


def size(t: Optional[Node]) -> int:
    return t.size if t is not None else 0


def _rotate_left(t: Node) -> Node:
    r = t.right
    return Node(Node(t.left, t.iv, t.vals, r.left), r.iv, r.vals, r.right)


def _rotate_right(t: Node) -> Node:
    l = t.left
    return Node(l.left, l.iv, l.vals, Node(l.right, t.iv, t.vals, t.right))


def _join_right(tl: Node, iv, vals: tuple, tr: Optional[Node]) -> Node:
    """``join`` when ``height(tl) > height(tr) + 1`` (descend along the right spine of ``tl``)."""
    l, c = tl.left, tl.right
    if height(c) <= height(tr) + 1:
        t1 = Node(c, iv, vals, tr)
        if t1.h <= height(l) + 1:
            return Node(l, tl.iv, tl.vals, t1)
        return _rotate_left(Node(l, tl.iv, tl.vals, _rotate_right(t1)))
    t1 = _join_right(c, iv, vals, tr)
    t2 = Node(l, tl.iv, tl.vals, t1)
    if t1.h <= height(l) + 1:
        return t2
    return _rotate_left(t2)


def _join_left(tl: Optional[Node], iv, vals: tuple, tr: Node) -> Node:
    """Mirror image of :func:`_join_right` (``height(tr) > height(tl) + 1``)."""
    c, r = tr.left, tr.right
    if height(c) <= height(tl) + 1:
        t1 = Node(tl, iv, vals, c)
        if t1.h <= height(r) + 1:
            return Node(t1, tr.iv, tr.vals, r)
        return _rotate_right(Node(_rotate_left(t1), tr.iv, tr.vals, r))
    t1 = _join_left(tl, iv, vals, c)
    t2 = Node(t1, tr.iv, tr.vals, r)
    if t1.h <= height(r) + 1:
        return t2
    return _rotate_right(t2)


def join(tl: Optional[Node], iv, vals: tuple, tr: Optional[Node]) -> Node:
    """The sequence ``tl``, ``iv``, ``tr``."""
    hl, hr = height(tl), height(tr)
    if hl > hr + 1:
        return _join_right(tl, iv, vals, tr)
    if hr > hl + 1:
        return _join_left(tl, iv, vals, tr)
    return Node(tl, iv, vals, tr)


def _split_last(t: Node) -> Tuple[Optional[Node], object, tuple]:
    if t.right is None:
        return t.left, t.iv, t.vals
    rest, iv, vals = _split_last(t.right)
    return join(t.left, t.iv, t.vals, rest), iv, vals


def concat(tl: Optional[Node], tr: Optional[Node]) -> Optional[Node]:
    """The sequence ``tl`` followed by ``tr``."""
    if tl is None:
        return tr
    if tr is None:
        return tl
    rest, iv, vals = _split_last(tl)
    return join(rest, iv, vals, tr)


def split(t: Optional[Node], k: int) -> Tuple[Optional[Node], Optional[Node]]:
    """``(first k elements, the others)`` for ``0 <= k <= size(t)``."""
    if t is None:
        return None, None
    ls = size(t.left)
    if k <= ls:
        a, b = split(t.left, k)
        return a, join(b, t.iv, t.vals, t.right)
    a, b = split(t.right, k - ls - 1)
    return join(t.left, t.iv, t.vals, a), b


def split3(t: Node, k: int) -> Tuple[Optional[Node], Tuple[object, tuple], Optional[Node]]:
    """``(first k elements, the k-th element as (component, values), the elements after
    it)`` for ``0 <= k < size(t)``: one descent, like :func:`split`."""
    ls = size(t.left)
    if k < ls:
        a, x, b = split3(t.left, k)
        return a, x, join(b, t.iv, t.vals, t.right)
    if k == ls:
        return t.left, (t.iv, t.vals), t.right
    a, x, b = split3(t.right, k - ls - 1)
    return join(t.left, t.iv, t.vals, a), x, b


def nth(t: Node, k: int) -> Tuple[object, tuple]:
    """The ``k``-th element (0-based) as ``(component, values)``."""
    while True:
        ls = size(t.left)
        if k < ls:
            t = t.left
        elif k == ls:
            return t.iv, t.vals
        else:
            k -= ls + 1
            t = t.right


def build(items: Sequence[Tuple[object, tuple]]) -> Optional[Node]:
    """A perfectly balanced tree with in-order sequence ``items`` (``(component, values)``)."""
    def rec(lo: int, hi: int) -> Optional[Node]:
        if lo >= hi:
            return None
        mid = (lo + hi) // 2
        iv, vals = items[mid]
        return Node(rec(lo, mid), iv, vals, rec(mid + 1, hi))

    return rec(0, len(items))


def to_list(t: Optional[Node]) -> List[object]:
    """The components in order."""
    out: List[object] = []
    stack: List[Node] = []
    cur = t
    while stack or cur is not None:
        while cur is not None:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        out.append(cur.iv)
        cur = cur.right
    return out


def prefix_sum(t: Optional[Node], i: int, k: int):
    """Value for agent ``i`` of the first ``k`` elements (``None`` if unknown)."""
    acc = ZERO
    while t is not None and k > 0:
        ls = size(t.left)
        if k <= ls:
            t = t.left
            continue
        left_sum = t.left.sums[i] if t.left is not None else ZERO
        v = t.vals[i]
        if left_sum is None or v is None:
            return None
        acc = acc + left_sum + v
        k -= ls + 1
        t = t.right
    return acc


def find_ge(t: Optional[Node], i: int, amount) -> Tuple[int, object]:
    """Smallest ``k`` with ``P_k >= amount`` and ``P_k``, where ``P_k`` is agent ``i``'s
    value of the first ``k`` elements; ``(size(t), total)`` if ``amount`` exceeds the
    total.  Raises ``KeyError`` if a value on the search path is unknown."""
    if amount <= 0:
        return 0, ZERO
    k = 0
    acc = ZERO          # invariant: acc = P_k < amount
    node = t
    while node is not None:
        left_sum = node.left.sums[i] if node.left is not None else ZERO
        v = node.vals[i]
        if left_sum is None or v is None:
            raise KeyError("value not known on the search path")
        if acc + left_sum >= amount:
            node = node.left
            continue
        acc2 = acc + left_sum + v
        k += size(node.left) + 1
        if acc2 >= amount:
            return k, acc2
        acc = acc2
        node = node.right
    return k, acc
