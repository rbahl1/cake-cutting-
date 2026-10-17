"""Exact fallback for SubCore: a competitive equilibrium of the trimming market.

When Aziz–Mackenzie's SubCore fails (``CORRECTNESS.md`` §7), :func:`allocate`
computes its output from a competitive equilibrium in the sense of Alaei, Jain
and Malekian, *Competitive equilibria in two sided matching markets with
non-transferable utilities* (arXiv:1006.4696).  The agents are the buyers, the
nonempty pieces are the goods, and the *price* of a piece is a trim position:
at price ``x`` a buyer receives the suffix of the piece that starts at ``x``.
So the utility ``V_i(P_j[x:])`` of a piece decreases from ``V_i(P_j)`` (price at
the start, the whole piece) to 0 (price at the end, the empty suffix).

Recursion (AJM Theorem 2).  For buyers ``X`` and goods ``Y`` let ``low(X, Y)`` be
the utilities at the lowest equilibrium and ``high(X, Y)`` the prices at the
highest one.  Then

* ``low(X, Y)_i = max_j V_i(P_j[high(X - i, Y)_j:])`` (the best suffix at the
  highest prices of the market without ``i``; 0 if ``Y`` is empty);
* ``high(X, Y)_j`` = the largest of the start of ``P_j`` and, for each buyer
  ``i`` of ``X`` with ``u_i <= V_i(P_j)``, the position where the suffix of
  ``P_j`` worth ``u_i = low(X, Y - j)_i`` to ``i`` begins.

The second rule is one ``suffix_position_of_value`` (at most one Cut) per buyer
and good.  :func:`allocate` takes the prices ``p = high(I - i0, J)`` of the
market without one agent ``i0`` and a matching of all agents that supports
``p``; it exists by AJM's construction in the proof of Theorem 2 (``i0`` takes
a best piece, the others are rematched by Theorem 5).  ``CORRECTNESS.md`` §8
proves that for valuations with positive density (every nonempty interval has
positive value for every agent) the result has every property of SubCore's
output; :func:`cake_cutting.bounds.E` bounds the number of cuts.

The result is verified from the value table exactly as SubCore's is; on an input
with zero-density stretches, where the proof does not apply, a failed check
returns ``None`` instead of a wrong allocation.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

from .bookkeeping import Position, ValueTable
from .pieces import OrderedPiece, ZERO

Agent = Hashable


class _Market:
    """The recursion of AJM Theorem 2 with memoization.

    Buyers and goods are numbered by their positions in ``agents`` and
    ``goods``; a set of buyers or goods is a bit mask.
    """

    def __init__(self, table: ValueTable, pieces: Sequence[OrderedPiece], agents: Sequence[Agent]) -> None:
        self.table = table
        self.agents = list(agents)
        self.goods = [idx for idx, p in enumerate(pieces) if len(p) > 0]
        self.gp = [pieces[idx] for idx in self.goods]
        self.start: List[Position] = [(0, p.first.a) for p in self.gp]
        self.whole = [[table.value(a, p) for p in self.gp] for a in self.agents]
        self.low_memo: Dict[Tuple[int, int], Dict[int, Fraction]] = {}
        self.high_memo: Dict[Tuple[int, int], Tuple[Dict[int, Position], Dict[int, OrderedPiece]]] = {}
        self.calls = 0  # suffix_position_of_value calls (each at most one Cut)

    def utility(self, b: int, views: Dict[int, OrderedPiece]) -> Fraction:
        """Buyer ``b``'s best value among the suffixes ``views`` (0 if there are none)."""
        a = self.agents[b]
        best = ZERO
        for v in views.values():
            x = self.table.value(a, v)
            if x > best:
                best = x
        return best

    def low(self, X: int, Y: int) -> Dict[int, Fraction]:
        """Utilities of the buyers ``X`` at the lowest equilibrium of ``(X, Y)``."""
        key = (X, Y)
        u = self.low_memo.get(key)
        if u is not None:
            return u
        u = {}
        for b in range(len(self.agents)):
            bit = 1 << b
            if X & bit:
                u[b] = self.utility(b, self.high(X ^ bit, Y)[1])
        self.low_memo[key] = u
        return u

    def high(self, X: int, Y: int) -> Tuple[Dict[int, Position], Dict[int, OrderedPiece]]:
        """Prices of the goods ``Y`` at the highest equilibrium of ``(X, Y)``, with the
        suffixes they leave."""
        key = (X, Y)
        st = self.high_memo.get(key)
        if st is not None:
            return st
        prices: Dict[int, Position] = {}
        views: Dict[int, OrderedPiece] = {}
        for g in range(len(self.goods)):
            bit = 1 << g
            if not Y & bit:
                continue
            u = self.low(X, Y ^ bit)
            piece = self.gp[g]
            best = self.start[g]
            for b, ub in u.items():
                if ub <= self.whole[b][g]:
                    self.calls += 1
                    pos = self.table.suffix_position_of_value(piece, self.agents[b], ub)
                    if pos > best:
                        best = pos
            prices[g] = best
            views[g] = self.table.suffix_from(piece, best)
        st = (prices, views)
        self.high_memo[key] = st
        return st


def _augment(b0: int, adj: List[List[int]], mate_b: List[int], mate_g: List[int]) -> bool:
    """Match the unmatched buyer ``b0`` by an augmenting path (breadth-first search
    over alternating paths ending at an unmatched good); every matched buyer and
    good stays matched.  ``False`` if there is no augmenting path."""
    parent_g = [-1] * len(mate_g)          # good -> buyer it was reached from
    seen_b = [False] * len(mate_b)
    seen_b[b0] = True
    queue = [b0]
    end = -1
    qi = 0
    while qi < len(queue) and end < 0:
        b = queue[qi]
        qi += 1
        for g in adj[b]:
            if parent_g[g] >= 0:
                continue
            parent_g[g] = b
            if mate_g[g] < 0:
                end = g
                break
            nxt = mate_g[g]
            if not seen_b[nxt]:
                seen_b[nxt] = True
                queue.append(nxt)
    if end < 0:
        return False
    g = end
    while g >= 0:
        b = parent_g[g]
        prev = mate_b[b]
        mate_b[b] = g
        mate_g[g] = b
        g = prev if b != b0 else -1
    return True


def _cover_matching(adj: List[List[int]], n_goods: int, need_b: List[bool], need_g: List[bool]
                    ) -> Optional[List[int]]:
    """A matching of the bipartite graph ``adj`` (buyer -> goods) that covers every
    buyer with ``need_b`` and every good with ``need_g``, or ``None`` if there is none.

    Phase 1 covers the needed buyers by augmenting paths (Kuhn).  Phase 2 covers
    each remaining needed good ``g`` by an alternating path from ``g`` that ends at
    an unmatched buyer or at a buyer whose good is not needed; flipping it keeps
    every matched buyer and every other matched good matched.  If some matching
    covers both sets, such a path exists (the component of ``g`` in the
    symmetric difference with that matching is one), so ``None`` means that no
    matching covers both.
    """
    nb = len(adj)
    mate_b = [-1] * nb
    mate_g = [-1] * n_goods
    radj: List[List[int]] = [[] for _ in range(n_goods)]
    for b in range(nb):
        for g in adj[b]:
            radj[g].append(b)
    for b0 in range(nb):
        if need_b[b0] and mate_b[b0] < 0 and not _augment(b0, adj, mate_b, mate_g):
            return None
    for g0 in range(n_goods):
        if not need_g[g0] or mate_g[g0] >= 0:
            continue
        # BFS over alternating paths from good g0: good -(edge)- buyer -(matching)- good
        parent_b = [-1] * nb               # buyer -> good it was reached from
        queue = [g0]
        seen_g = [False] * n_goods
        seen_g[g0] = True
        end = -1
        qi = 0
        while qi < len(queue) and end < 0:
            g = queue[qi]
            qi += 1
            for b in radj[g]:
                if parent_b[b] >= 0:
                    continue
                parent_b[b] = g
                h = mate_b[b]
                if h < 0 or not need_g[h]:
                    end = b
                    break
                if not seen_g[h]:
                    seen_g[h] = True
                    queue.append(h)
        if end < 0:
            return None
        b = end
        h = mate_b[b]
        if h >= 0:
            mate_g[h] = -1                 # a good that is not needed becomes free
        while b >= 0:
            g = parent_b[b]
            prev = mate_g[g]
            mate_b[b] = g
            mate_g[g] = b
            b = prev if g != g0 else -1
    return mate_b


def allocate(table: ValueTable, pieces: Sequence[OrderedPiece], agents: Sequence[Agent],
             benchmarks: Dict[Agent, Fraction]
             ) -> Tuple[Optional[Dict[Agent, Tuple[int, OrderedPiece]]], int]:
    """SubCore's output computed from a competitive equilibrium.

    Returns ``(allocation, calls)`` with ``allocation[j] = (piece index, suffix)``
    as :func:`cake_cutting.subcore.subcore` does, or ``(None, calls)`` if the
    equilibrium has no supporting matching or the result fails one of SubCore's
    checks (both impossible for positive densities, ``CORRECTNESS.md`` §8);
    ``calls`` counts ``suffix_position_of_value`` calls (each at most one Cut),
    at most ``bounds.E(len(agents), number of nonempty pieces)``.
    """
    agents = list(agents)
    m = len(agents)
    if m == 0:
        return {}, 0
    mk = _Market(table, pieces, agents)
    k = len(mk.goods)
    prices, views = mk.high(((1 << m) - 1) ^ 1, (1 << k) - 1)
    u = [mk.utility(b, views) for b in range(m)]
    adj: List[List[int]] = []
    for b in range(m):
        a = agents[b]
        adj.append([g for g in range(k) if table.value(a, views[g]) == u[b]])
    need_b = [x > 0 for x in u]
    need_g = [prices[g] != mk.start[g] for g in range(k)]
    mate = _cover_matching(adj, k, need_b, need_g)
    if mate is None:
        return None, mk.calls
    alloc: Dict[Agent, Tuple[int, OrderedPiece]] = {}
    used = set()
    for b in range(m):
        g = mate[b]
        if g >= 0:
            alloc[agents[b]] = (mk.goods[g], views[g])
            used.add(mk.goods[g])
    # an unmatched agent has utility 0: she receives the (empty) suffix of a piece
    # nobody holds, an empty piece if there is one
    free = sorted((idx for idx in range(len(pieces)) if idx not in used),
                  key=lambda idx: (len(pieces[idx]) > 0, idx))
    fi = 0
    for b in range(m):
        if mate[b] < 0:
            if fi >= len(free):
                return None, mk.calls
            alloc[agents[b]] = (free[fi], OrderedPiece.empty())
            used.add(free[fi])
            fi += 1
    if not _verify(table, pieces, alloc, benchmarks):
        return None, mk.calls
    return alloc, mk.calls


def _verify(table: ValueTable, pieces: Sequence[OrderedPiece],
            alloc: Dict[Agent, Tuple[int, OrderedPiece]], benchmarks: Dict[Agent, Fraction]) -> bool:
    """SubCore's postconditions, from the value table (no queries): required minima,
    no envy between the agents, no agent prefers a piece nobody holds."""
    held = {idx for idx, _ in alloc.values()}
    for j, (_, suffix) in alloc.items():
        v = table.value(j, suffix)
        if v < benchmarks[j]:
            return False
        for k2, (_, other) in alloc.items():
            if k2 != j and table.value(j, other) > v:
                return False
        for idx, p in enumerate(pieces):
            if idx not in held and table.value(j, p) > v:
                return False
    return True
