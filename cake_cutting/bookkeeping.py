"""Queries on ordered pieces -- Proposition C.1 and the conventions of Appendix C.2.

The protocol keeps a table with every active agent's value of every interval
component it has created ("Once we know how every agent in N values every
interval component, we can evaluate the ordered piece without further
queries").  The only places where Robertson–Webb queries are issued are:

* :meth:`ValueTable.register_cake` -- one Eval per agent for ``[0, 1]`` at the
  top level;
* :meth:`ValueTable.split_by_cut` -- one Cut for the agent asking for a prefix
  of a given value, plus one Eval for every *other* active agent (the second
  part follows by subtraction) -- exactly the accounting of Proposition C.1;
* :meth:`ValueTable.split_at` -- a split at a point chosen by the protocol
  (used only by the simultaneous-cutting procedure): one Eval per active agent.

Prefixes and suffixes of a prescribed value follow Section C.2: "we examine the
component boundaries from the beginning of this order and use the first one
that gives the requested value.  If none does, one ordinary cut query locates
the cut point inside a component."  A suffix of value ``a`` is the complement
of the prefix of value ``V_i(P) - a``.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

from . import seq
from .oracle import RobertsonWebbOracle
from .pieces import Interval, OrderedPiece, ZERO

Agent = Hashable
#: A position inside an ordered piece: ``(component index k, point)``.  The
#: boundary *before* component ``k`` is ``(k, start of component k)``; the end of
#: the piece is ``(len, 0)``.  Positions compare lexicographically, which is the
#: "left/right" order of Section C.1 (later component, or further point).
Position = Tuple[int, Fraction]


class ValueTable:
    """Known values ``V_i(component)`` for all active agents (Proposition C.1)."""

    def __init__(self, oracle: RobertsonWebbOracle, active: Sequence[Agent]) -> None:
        self.oracle = oracle
        self.active: Tuple[Agent, ...] = tuple(active)
        #: every agent the table will ever hold values for, numbered for the value vectors
        self.agents_all: Tuple[Agent, ...] = tuple(active)
        self.index: Dict[Agent, int] = {a: k for k, a in enumerate(self.agents_all)}
        # (agent, interval) -> V_agent(interval); Interval hashes are cached
        self._known: Dict[Tuple[Agent, Interval], Fraction] = {}
        # (agent, end, value) -> smallest known start z with V_agent([z, end)) == value
        self._min_start: Dict[Tuple[Agent, Fraction, Fraction], Fraction] = {}
        # interval -> list of its values indexed by agent number (None = not recorded);
        # the tree nodes of tree-backed pieces copy these vectors
        self._vec: Dict[Interval, list] = {}

    def __len__(self) -> int:
        return len(self._known)

    def _set(self, agent: Agent, iv: Interval, v: Fraction) -> None:
        key = (agent, iv)
        if key not in self._known:
            self._known[key] = v
            mkey = (agent, iv.b, v)
            cur = self._min_start.get(mkey)
            if cur is None or iv.a < cur:
                self._min_start[mkey] = iv.a
            vec = self._vec.get(iv)
            if vec is None:
                vec = self._vec[iv] = [None] * len(self.agents_all)
            vec[self.index[agent]] = v

    # -- tree-backed pieces ---------------------------------------------------
    def vals_of(self, iv: Interval) -> tuple:
        """The value vector of ``iv`` (``None`` for agents whose value is not recorded)."""
        vec = self._vec.get(iv)
        if vec is None:
            return (None,) * len(self.agents_all)
        return tuple(vec)

    def tree_of(self, piece: OrderedPiece):
        """The full tree of ``piece`` for this table: built from its components on first
        use, and for a suffix view the tree of the view (``O(n log C)``, see
        :meth:`full_tree`)."""
        if piece._tree is not None and piece._owner is self:
            if piece._k0 or piece._head is not None:
                return self.full_tree(piece)
            return piece._tree
        comps = piece.components
        if not comps:
            return None
        t = seq.build([(c, self.vals_of(c)) for c in comps])
        piece._tree = t
        piece._owner = self
        piece._k0 = 0
        piece._head = None
        piece._first = t.first
        piece._vk0 = None
        return t

    def full_tree(self, piece: OrderedPiece):
        """Turn a suffix view into a tree of its own (in place; values unchanged)."""
        if piece._tree is None or piece._owner is not self:
            return self.tree_of(piece)
        if piece._k0 == 0 and piece._head is None:
            return piece._tree
        if piece._head is None:
            t = seq.split(piece._tree, piece._k0)[1]
        else:
            right = seq.split3(piece._tree, piece._k0)[2]
            t = seq.join(None, piece._head, self.vals_of(piece._head), right)
        piece._tree = t
        piece._k0 = 0
        piece._head = None
        piece._vk0 = None
        return t

    def make_piece(self, intervals: Sequence[Interval]) -> OrderedPiece:
        """A tree-backed ordered piece with the given components."""
        comps = tuple(intervals)
        if not comps:
            return OrderedPiece.empty()
        return OrderedPiece._from_tree(seq.build([(c, self.vals_of(c)) for c in comps]), self, comps)

    def _wrap(self, tree) -> OrderedPiece:
        return OrderedPiece._from_tree(tree, self)

    def _base(self, piece: OrderedPiece):
        """``(tree, k0, head)`` of a tree-backed piece of this table (building the tree of
        a tuple-backed piece on first use)."""
        if piece._tree is None or piece._owner is not self:
            self.tree_of(piece)
        return piece._tree, piece._k0, piece._head

    def _comp(self, piece: OrderedPiece, j: int):
        """Component ``j`` of ``piece`` and its value vector (``O(log C)``)."""
        t, k0, head = self._base(piece)
        if j == 0 and k0 + (head is not None) > 0:
            if head is not None:
                return head, self.vals_of(head)
            return piece._first, piece._vk0
        return seq.nth(t, k0 + j)

    def nth_component(self, piece: OrderedPiece, k: int) -> Interval:
        """Component ``k`` (0-based) of ``piece``."""
        return self._comp(piece, k)[0]

    # -- activation ---------------------------------------------------------
    def set_active(self, agents: Iterable[Agent]) -> None:
        """Restrict bookkeeping to the agents of the current ``Main`` call."""
        agents = tuple(agents)
        for a in agents:
            if a not in self.index:
                raise ValueError(f"agent {a!r} was not known when the table was created")
        self.active = agents

    def register_cake(self) -> None:
        """Top level of the protocol: one Eval per agent for the whole cake."""
        whole = Interval(ZERO, Fraction(1))
        for i in self.active:
            self._set(i, whole, self.oracle.eval(i, ZERO, Fraction(1)))

    def register_interval(self, interval: Interval, agents: Optional[Iterable[Agent]] = None) -> None:
        """Make the value of a brand-new interval known (one Eval per agent)."""
        for i in (self.active if agents is None else agents):
            if (i, interval) not in self._known:
                self._set(i, interval, self.oracle.eval(i, interval.a, interval.b))

    # -- evaluation without queries ---------------------------------------
    def value_of_interval(self, agent: Agent, c: Interval) -> Fraction:
        try:
            return self._known[(agent, c)]
        except KeyError:
            raise KeyError(f"value of {c} for agent {agent!r} is not known") from None

    def value(self, agent: Agent, piece: OrderedPiece) -> Fraction:
        """``V_i(piece)`` computed from the known component values (no queries).

        ``O(1)`` for a tree (the root sum, computed once per node and agent); for a
        suffix view one prefix sum, ``O(log C)``, kept per view and agent."""
        if len(piece) == 0:
            return ZERO
        t, k0, head = self._base(piece)
        i = self.index[agent]
        if k0 == 0 and head is None:
            v = t.sums[i]
        else:
            cache = piece._vcache
            if cache is None:
                cache = piece._vcache = {}
            v = cache.get(i, seq.UNSET)
            if v is seq.UNSET:
                total = t.sums[i]
                before = seq.prefix_sum(t, i, k0)
                if total is None or before is None:
                    v = None
                elif head is None:
                    v = total - before
                else:
                    vk0 = piece._vk0[i]
                    hv = self._known.get((agent, head))
                    v = None if (vk0 is None or hv is None) else total - before - vk0 + hv
                cache[i] = v
        if v is None:
            raise KeyError(f"value of a component of the piece for agent {agent!r} is not known")
        return v

    def values(self, piece: OrderedPiece) -> Dict[Agent, Fraction]:
        return {i: self.value(i, piece) for i in self.active}

    def is_known(self, agent: Agent, c: Interval) -> bool:
        return (agent, c) in self._known

    # -- splitting components ---------------------------------------------
    def split_by_cut(self, c: Interval, cutter: Agent, amount: Fraction) -> Tuple[Interval, Interval, Fraction]:
        """Agent ``cutter`` cuts ``c`` into ``[a, y)`` worth ``amount`` and ``[y, b)``.

        Requires ``0 < amount < V_cutter(c)``.  Costs one Cut query and one Eval
        for every other active agent (Proposition C.1).
        """
        total = self.value_of_interval(cutter, c)
        if not (0 < amount < total):
            raise ValueError(f"cut amount {amount} must lie strictly inside (0, {total})")
        y = self.oracle.cut(cutter, c.a, amount)
        if not (c.a < y < c.b):
            raise AssertionError(f"cut point {y} not strictly inside {c}")
        left, right = c.split(y)
        self._set(cutter, left, amount)
        self._set(cutter, right, total - amount)
        for i in self.active:
            if i == cutter:
                continue
            self._learn_split(i, c, left, right)
        return left, right, y

    def _learn_split(self, i: Agent, c: Interval, left: Interval, right: Interval) -> None:
        """Record agent ``i``'s values of the two parts of ``c``: one Eval unless a part is
        already known (then both follow by subtraction, without a query)."""
        known_left = self._known.get((i, left))
        known_right = self._known.get((i, right))
        if known_left is not None and known_right is not None:
            return
        v_total = self.value_of_interval(i, c)
        if known_left is not None:
            self._set(i, right, v_total - known_left)
            return
        if known_right is not None:
            self._set(i, left, v_total - known_right)
            return
        v_left = self.oracle.eval(i, left.a, left.b)
        self._set(i, left, v_left)
        self._set(i, right, v_total - v_left)

    def complement_prefix(self, outer: Interval, suffix: Interval) -> Interval:
        """``[outer.a, suffix.a)`` where ``suffix = [y, outer.b)`` is a known sub-interval.

        No queries: every active agent's value follows by subtraction.
        """
        if not (outer.a < suffix.a and suffix.b == outer.b):
            raise ValueError(f"{suffix} is not a proper suffix of {outer}")
        left = Interval(outer.a, suffix.a)
        for i in self.active:
            if (i, left) not in self._known:
                self._set(i, left, self.value_of_interval(i, outer) - self.value_of_interval(i, suffix))
        return left

    def split_at(self, c: Interval, y: Fraction) -> Tuple[Interval, Interval]:
        """Split ``c`` at a point chosen by the protocol: one Eval per active agent."""
        left, right = c.split(y)
        for i in self.active:
            self._learn_split(i, c, left, right)
        return left, right

    def record_union(self, left: Interval, right: Interval) -> Interval:
        """``[left.a, right.b)`` for two adjacent known intervals: values by addition, no queries."""
        if left.b != right.a:
            raise ValueError(f"{left} and {right} are not adjacent")
        whole = Interval(left.a, right.b)
        for i in self.active:
            if (i, whole) not in self._known:
                self._set(i, whole, self.value_of_interval(i, left) + self.value_of_interval(i, right))
        return whole

    def normalize_left(self, piece: OrderedPiece, agent: Agent, pos: Position) -> Position:
        """Move a position to the left over cake that ``agent`` is *known* to value at zero.

        The suffix of ``piece`` from the returned position has the same value
        for ``agent`` as the suffix from ``pos``; it extends as far left as the
        known component values allow (no queries are made).  Used by SubCore so
        that the comparison of trims does not depend on which of several
        admissible answers a Cut query returned inside a zero-valued stretch.

        The run of zero-valued components immediately before component ``k`` is
        found by one search in the tree: it starts at the smallest boundary ``k'``
        whose prefix value equals the prefix value at ``k``.
        """
        n_c = len(piece)
        k, y = pos
        if n_c == 0:
            return pos
        if k >= n_c:
            k, y = n_c - 1, piece.last.b
        c = self.nth_component(piece, k)
        target = ZERO if y == c.b else self.value_of_interval(agent, Interval(y, c.b))
        best = self._leftmost_start(agent, c, y, target)
        if best is None:
            return pos if y < c.b else (k + 1, ZERO)
        if best != c.a:
            return (k, best)
        if k == 0:
            return (0, c.a)
        # the whole of component k is covered: continue into earlier components,
        # first over the run of components the agent values at zero
        t, k0, head = self._base(piece)
        i = self.index[agent]
        if k0 == 0 and head is None:
            qk = seq.prefix_sum(t, i, k)
            if qk is None:
                raise KeyError(f"value of a component for agent {agent!r} is not known")
            k2 = seq.find_ge(t, i, qk)[0] if qk != 0 else 0
        else:
            v0 = self._comp(piece, 0)[1][i]
            p1 = seq.prefix_sum(t, i, k0 + 1)
            pk = seq.prefix_sum(t, i, k0 + k)
            if v0 is None or p1 is None or pk is None:
                raise KeyError(f"value of a component for agent {agent!r} is not known")
            qk = v0 + pk - p1
            k2 = 0 if qk == 0 else max(1, seq.find_ge(t, i, pk)[0] - k0)
        if k2 > 0:
            prev = self.nth_component(piece, k2 - 1)
            z = self._leftmost_start(agent, prev, prev.b, ZERO)
            if z is not None and z != prev.a:
                return (k2 - 1, z)
        return (k2, self.nth_component(piece, k2).a)

    def _leftmost_start(self, agent: Agent, c: Interval, y: Fraction, target: Fraction) -> Optional[Fraction]:
        """Smallest known ``z <= y`` in ``c`` with ``V_agent([z, c.b)) == target`` (``None`` if only ``y``)."""
        if self.value_of_interval(agent, c) == target:
            return c.a
        # V([z, c.b)) is non-increasing in z, so the smallest known start with the target
        # value is the answer whenever it lies inside (c.a, y); if it were <= c.a then c
        # itself would have the target value, which was excluded above.
        z = self._min_start.get((agent, c.b, target))
        if z is not None and c.a < z < y:
            return z
        return None

    # -- prefixes and suffixes of a given value (Section C.2) --------------
    def _locate(self, piece: OrderedPiece, agent: Agent, amount: Fraction):
        """The Section C.2 rule: ``("boundary", k)`` for the first component boundary
        ``k`` whose cumulative value equals ``amount``, else ``("inside", k, cum)`` for
        the unique component ``k`` that straddles it (``cum`` = value before it).
        One tree search; ``0 <= amount <= V(piece)`` is assumed."""
        t, k0, head = self._base(piece)
        i = self.index[agent]
        if k0 == 0 and head is None:
            k, pk = seq.find_ge(t, i, amount)
            if pk == amount:
                return ("boundary", k)
            return ("inside", k - 1, pk - seq.nth(t, k - 1)[1][i])
        if amount <= 0:
            return ("boundary", 0)
        v0 = self._comp(piece, 0)[1][i]
        if v0 is None:
            raise KeyError(f"value of a component for agent {agent!r} is not known")
        if amount <= v0:
            return ("boundary", 1) if amount == v0 else ("inside", 0, ZERO)
        p1 = seq.prefix_sum(t, i, k0 + 1)
        K, pK = seq.find_ge(t, i, amount - v0 + p1)
        qj = v0 + pK - p1
        if qj == amount:
            return ("boundary", K - k0)
        return ("inside", K - k0 - 1, qj - seq.nth(t, K - 1)[1][i])

    def prefix_of_value(self, piece: OrderedPiece, agent: Agent, amount: Fraction
                        ) -> Tuple[OrderedPiece, OrderedPiece, Position]:
        """Split ``piece`` into a prefix worth exactly ``amount`` to ``agent`` and the rest.

        Returns ``(prefix, suffix, position)``.  The first component boundary
        (from the beginning of the order) whose cumulative value equals
        ``amount`` is used; otherwise one Cut query is made inside the unique
        component that straddles ``amount``.
        """
        total = self.value(agent, piece)
        if not (0 <= amount <= total):
            raise ValueError(f"requested prefix value {amount} outside [0, {total}]")
        n_c = len(piece)
        if n_c == 0:
            return piece, OrderedPiece.empty(), (0, ZERO)
        t = self.tree_of(piece)               # a suffix view becomes a tree here
        loc = self._locate(piece, agent, amount)
        if loc[0] == "boundary":
            k = loc[1]
            if k >= n_c:
                return piece, OrderedPiece.empty(), (n_c, ZERO)
            a, b = seq.split(t, k)
            return self._wrap(a), self._wrap(b), (k, b.first.a)
        _, k, cum = loc
        a, (c, _), b = seq.split3(t, k)
        left, right, y = self.split_by_cut(c, agent, amount - cum)
        prefix = seq.join(a, left, self.vals_of(left), None)
        suffix = seq.join(None, right, self.vals_of(right), b)
        return self._wrap(prefix), self._wrap(suffix), (k, y)

    def position_of_value(self, piece: OrderedPiece, agent: Agent, amount: Fraction) -> Position:
        """The position returned by :meth:`prefix_of_value` (same queries), without
        building the two pieces (works on suffix views)."""
        total = self.value(agent, piece)
        if not (0 <= amount <= total):
            raise ValueError(f"requested prefix value {amount} outside [0, {total}]")
        n_c = len(piece)
        if n_c == 0:
            return (0, ZERO)
        loc = self._locate(piece, agent, amount)
        if loc[0] == "boundary":
            k = loc[1]
            if k >= n_c:
                return (n_c, ZERO)
            return (k, self.nth_component(piece, k).a)
        _, k, cum = loc
        c = self.nth_component(piece, k)
        _, _, y = self.split_by_cut(c, agent, amount - cum)
        return (k, y)

    def suffix_of_value(self, piece: OrderedPiece, agent: Agent, amount: Fraction
                        ) -> Tuple[OrderedPiece, OrderedPiece, Position]:
        """A suffix worth exactly ``amount``: the complement of a prefix worth ``V - amount``.

        Returns ``(prefix, suffix, position)`` where ``prefix`` is the trimmed-off part.
        """
        total = self.value(agent, piece)
        if not (0 <= amount <= total):
            raise ValueError(f"requested suffix value {amount} outside [0, {total}]")
        return self.prefix_of_value(piece, agent, total - amount)

    def suffix_position_of_value(self, piece: OrderedPiece, agent: Agent, amount: Fraction) -> Position:
        """The position returned by :meth:`suffix_of_value` (same queries), without pieces."""
        total = self.value(agent, piece)
        if not (0 <= amount <= total):
            raise ValueError(f"requested suffix value {amount} outside [0, {total}]")
        return self.position_of_value(piece, agent, total - amount)

    def suffix_from(self, piece: OrderedPiece, pos: Position) -> OrderedPiece:
        """The suffix of ``piece`` starting at a position produced by this table, as a
        suffix view (``O(log C)``, no tree nodes are created)."""
        k, y = pos
        n_c = len(piece)
        if k >= n_c:
            return OrderedPiece.empty()
        t, k0, head = self._base(piece)
        if k == 0 and (k0 > 0 or head is not None):
            c, base_vals = (head if head is not None else piece._first), piece._vk0
        else:
            c, base_vals = seq.nth(t, k0 + k)
        if y == c.a:
            if k == 0:
                return piece
            return OrderedPiece._view(t, self, k0 + k, None, c, base_vals)
        if not (c.a < y < c.b):
            raise ValueError(f"position {pos} does not lie in component {k} of the piece")
        part = Interval(y, c.b)
        return OrderedPiece._view(t, self, k0 + k, part, part, base_vals)

    def prefix_to(self, piece: OrderedPiece, pos: Position) -> OrderedPiece:
        k, y = pos
        n_c = len(piece)
        if k >= n_c:
            return piece
        t = self.tree_of(piece)               # a suffix view becomes a tree here
        c = seq.nth(t, k)[0]
        head = seq.split(t, k)[0]
        if y == c.a:
            return self._wrap(head)
        part = Interval(c.a, y)
        return self._wrap(seq.join(head, part, self.vals_of(part), None))
