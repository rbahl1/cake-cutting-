"""Cutting several pieces at once -- Lemma 4.6 of Ye–Bai.

Given the active agents ``N``, an ordered residue ``R`` whose component values
are stored for every agent, requests ``lambda_j >= 0`` (``sum_j lambda_j <= 1``)
and a tolerance ``0 < delta < 1``, produce pairwise disjoint pieces
``C_j ⊆ R`` with

    |V_i(C_j) - lambda_j V_i(R)| <= delta V_i(R)      for all i in N, j in J.

Lemma 4.6 asks for *some* deterministic bounded procedure with this guarantee.
The procedure of its proof (cut ``R`` into cells worth ``gamma V_i(R)`` for every
agent, ``gamma = delta^2 / (2 ln(2n|J| + 2))``, then search all labelings of the
cells) needs ``n^2 / gamma`` queries -- about ``10^23`` at ``n = 13`` -- and an
exhaustive search over ``(|J| + 1)^{#cells}`` labelings.  This module uses a
different procedure with the same guarantee, a proof, a much smaller query
count and polynomial running time:

**Procedure ``rounding`` (the one used by the protocol).**  Write ``A+`` for the
``r`` agents with ``V_i(R) > 0`` (the others value every piece at zero and impose
no condition).  Requests with ``lambda_j <= delta`` get the empty piece, which
satisfies the inequality.  The remaining ``m'`` labels and one label "unused"
(target ``1 - sum lambda_j``) are the leaves of a balanced binary tree of depth
``D = ceil(log2(m' + 1))``.  Starting with the cells of ``R`` (its components) at
the root, every internal node splits its cells between its two children so that
the left child receives the fraction ``p = lambda(left labels) / lambda(node
labels)`` of the node's value for *every* agent of ``A+``, up to an error of
``(r/2) gamma V_i(R)`` with ``gamma = 2 delta / (r D)``.  One split is linear-
algebra rounding:

* every cell gets the fractional weight ``y = p``; the ``r`` sums
  ``sum_K y_K V_i(K)`` (``i`` in ``A+``) are kept *exactly* equal to
  ``p V_i(node)`` throughout;
* cells are added one at a time to a set ``F`` of fractional cells; whenever
  ``|F| = r + 1`` there is a nonzero ``z`` with ``sum_K z_K V_i(K) = 0`` for all
  ``i`` (``r`` equations, ``r + 1`` unknowns); moving ``y`` along ``z`` until a
  coordinate reaches 0 or 1 makes at least one cell integral (it leaves ``F``);
* a cell is *fine* if every agent of ``A+`` values it at most ``gamma V_i(R)``;
  while some cell of ``F`` is not fine, every such cell is halved by the agent
  with the largest share ``V_i(K)/V_i(R)`` (one Cut, ``n - 1`` Evals), both
  halves keep the weight ``y`` (the sums are unchanged), and ``F`` is reduced to
  ``r`` cells again;
* finally every cell of ``F`` (at most ``r``, all fine) is rounded to the nearer
  of 0 and 1, which moves each sum by at most ``(r/2) gamma V_i(R)``.

Errors add up along the at most ``D`` splits above a leaf (each child's target
is ``p`` times its parent's), so every leaf ``j`` satisfies the inequality with
``D (r/2) gamma = delta``.  An agent halves a cell only while her share exceeds
``gamma``, so along any chain of halvings she cuts at most ``ceil(log2(1/gamma))``
times and a chain has length at most ``H = r ceil(log2(1/gamma))``; every round
of halvings deepens every non-fine cell of ``F``, so a split makes at most ``H``
rounds of at most ``r`` cuts.  Hence the procedure makes at most
``m' r^2 ceil(log2(1/gamma))`` cuts, each costing ``n`` queries (Proposition C.1):
``bounds.lemma46_queries``; ``PROOF.md`` §6 has the counts used by the operation bound.

**Procedure ``paper`` (tests only).**  The procedure of the proof of Lemma 4.6,
verbatim, with a caller-supplied coarse ``gamma`` (``force_fallback=True``).  The
protocol never uses it.

Both procedures interact with the agents only through the value table (hence
the counting oracle), use exact rational arithmetic, and return the pieces as
ordered pieces whose components appear in the order of ``R``, together with the
unused part of ``R`` (also in that order).
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

from .bookkeeping import ValueTable
from .pieces import Interval, OrderedPiece, ZERO

Agent = Hashable

HALF = Fraction(1, 2)


class SimultaneousCutError(RuntimeError):
    """The requested pieces could not be produced (only the test-only ``paper``
    procedure can fail; for ``rounding`` this signals a broken invariant)."""


def kernel_vector(rows: List[List[Fraction]], ncols: int) -> List[Fraction]:
    """A nonzero ``z`` with ``rows z = 0`` when ``len(rows) < ncols`` (Gauss–Jordan).

    Every pivot row is scaled to a leading 1 and eliminated from all other rows;
    a column without pivot exists because there are fewer rows than columns, and
    ``z`` is 1 there, ``-A[i][free]`` at the pivot of row ``i`` and 0 elsewhere.
    """
    A = [list(row) for row in rows]
    nrows = len(A)
    pivot_cols: List[int] = []
    prow = 0
    for col in range(ncols):
        if prow == nrows:
            break
        piv = -1
        for rr in range(prow, nrows):
            if A[rr][col] != 0:
                piv = rr
                break
        if piv < 0:
            continue
        A[prow], A[piv] = A[piv], A[prow]
        lead = A[prow][col]
        A[prow] = [x / lead for x in A[prow]]
        for rr in range(nrows):
            if rr != prow:
                f = A[rr][col]
                if f != 0:
                    A[rr] = [a - f * b for a, b in zip(A[rr], A[prow])]
        pivot_cols.append(col)
        prow += 1
    pivot_set = set(pivot_cols)
    free = next(c for c in range(ncols) if c not in pivot_set)
    z = [ZERO] * ncols
    z[free] = Fraction(1)
    for i, pc in enumerate(pivot_cols):
        z[pc] = -A[i][free]
    return z


class _Frac:
    """A cell of one split: a contiguous run of ``R`` (an ordered piece), its weight ``y``
    and its position key ``(index in R of its first component's origin, start point)``."""

    __slots__ = ("piece", "y", "key")

    def __init__(self, piece: OrderedPiece, y: Fraction, key: Tuple[int, Fraction]) -> None:
        self.piece = piece
        self.y = y
        self.key = key


class SimultaneousCutter:
    """One application of Lemma 4.6 (see the module docstring)."""

    def __init__(self, table: ValueTable, agents: Sequence[Agent], residue: OrderedPiece,
                 requests: Dict[Hashable, Fraction], delta: Fraction, *,
                 budget: Optional[int] = None, force_fallback: bool = False,
                 gamma: Optional[Fraction] = None, max_fallback_cuts: int = 2_000_000) -> None:
        self.table = table
        self.agents = list(agents)
        self.residue = residue
        self.keys = sorted(requests, key=repr)
        self.lam = {k: Fraction(requests[k]) for k in self.keys}
        self.delta = Fraction(delta)
        if not (0 < self.delta < 1):
            raise ValueError("delta must lie in (0, 1)")
        for k in self.keys:
            if self.lam[k] < 0:
                raise ValueError("requested fractions must be nonnegative")
        if sum(self.lam.values(), ZERO) > 1:
            raise ValueError("requested fractions must sum to at most 1")
        self.budget = budget
        self.force_fallback = force_fallback
        self.gamma_override = gamma
        self.total = {i: table.value(i, residue) for i in self.agents}
        self.tol = {i: self.delta * self.total[i] for i in self.agents}
        self.queries_start = table.oracle.queries
        self.phase_used: Optional[str] = None
        self.cuts = 0
        self.kernel_steps = 0
        self.max_fallback_cuts = max_fallback_cuts

    # -- helpers ------------------------------------------------------------
    def _v(self, i: Agent, iv: Interval) -> Fraction:
        return self.table.value_of_interval(i, iv)

    def used_queries(self) -> int:
        return self.table.oracle.queries - self.queries_start

    def verify(self, pieces: Dict[Hashable, OrderedPiece]) -> bool:
        """Exact check of the lemma's inequality for every agent and request."""
        for k in self.keys:
            p = pieces[k]
            for i in self.agents:
                v = self.table.value(i, p)
                if abs(v - self.lam[k] * self.total[i]) > self.tol[i]:
                    return False
        return True

    # -- entry point --------------------------------------------------------
    def run(self) -> Tuple[Dict[Hashable, OrderedPiece], OrderedPiece]:
        if self.force_fallback:
            pieces, leftover = self._paper_procedure()
            if not self.verify(pieces):
                raise SimultaneousCutError("the labeling search returned an invalid labeling")
            self.phase_used = "paper"
            return pieces, leftover
        pieces, leftover = self._rounding()
        if self.budget is not None and self.used_queries() > self.budget:
            raise SimultaneousCutError("Lemma 4.6 rounding exceeded its proved query bound")
        if not self.verify(pieces):
            raise SimultaneousCutError("Lemma 4.6 rounding violated its proved guarantee")
        self.phase_used = "rounding"
        return pieces, leftover

    # ======================================================================
    # Procedure ``rounding``
    # ======================================================================
    def _rounding(self) -> Tuple[Dict[Hashable, OrderedPiece], OrderedPiece]:
        plus = [i for i in self.agents if self.total[i] > 0]
        nonempty = [k for k in self.keys if self.lam[k] > self.delta]
        empty = OrderedPiece.empty()
        if not nonempty or not plus:
            return {k: empty for k in self.keys}, self.residue
        self.plus = plus
        r = len(plus)
        labels: List[Optional[Hashable]] = list(nonempty) + [None]
        target: Dict[Optional[Hashable], Fraction] = {k: self.lam[k] for k in nonempty}
        target[None] = 1 - sum((self.lam[k] for k in nonempty), ZERO)
        depth = (len(labels) - 1).bit_length()          # ceil(log2(len(labels)))
        self.gamma = 2 * self.delta / (r * depth)
        self.fine_bound = {i: self.gamma * self.total[i] for i in plus}
        # the whole residue is the single initial cell; cells are only ever halved
        cells = [(self.residue, (0, self.residue.first.a))]
        out: Dict[Optional[Hashable], List[Tuple[OrderedPiece, Tuple[int, Fraction]]]] = {}
        self._assign(cells, labels, target, out)
        pieces: Dict[Hashable, OrderedPiece] = {}
        for k in self.keys:
            got = out.get(k) if k in target else None
            pieces[k] = self._collect(got) if got else empty
        return pieces, self._collect(out.get(None) or [])

    @staticmethod
    def _collect(cells) -> OrderedPiece:
        """Concatenate cells in the order of ``R`` (their keys)."""
        cells = sorted(cells, key=lambda e: e[1])
        return OrderedPiece.concat([p for p, _ in cells])

    def _assign(self, cells, labels, target, out) -> None:
        """Distribute ``cells`` among ``labels`` along the balanced binary tree."""
        if len(labels) == 1:
            out[labels[0]] = cells
            return
        mid = len(labels) // 2
        left, right = labels[:mid], labels[mid:]
        lam_left = sum((target[k] for k in left), ZERO)
        lam_all = lam_left + sum((target[k] for k in right), ZERO)
        p = lam_left / lam_all            # lam_all > 0: an internal node holds a request > delta
        s1, s2 = self._split(cells, p)
        self._assign(s1, left, target, out)
        self._assign(s2, right, target, out)

    def _fine(self, piece: OrderedPiece) -> bool:
        return all(self.table.value(i, piece) <= self.fine_bound[i] for i in self.plus)

    def _split(self, cells, p: Fraction):
        """One binary split: the side-1 cells carry ``p`` of every ``A+`` agent's value
        of ``cells`` up to ``(r/2) gamma V_i(R)``."""
        if p == 0:
            return [], list(cells)
        if p == 1:
            return list(cells), []
        r = len(self.plus)
        side1: List[Tuple[OrderedPiece, Tuple[int, Fraction]]] = []
        side2: List[Tuple[OrderedPiece, Tuple[int, Fraction]]] = []
        F: List[_Frac] = []
        for piece, key in cells:
            F.append(_Frac(piece, p, key))
            if len(F) > r:
                self._reduce(F, side1, side2)
        while True:
            coarse = [e for e in F if not self._fine(e.piece)]
            if not coarse:
                break
            kept = [e for e in F if self._fine(e.piece)]
            for e in coarse:
                kept.extend(self._halve(e))
            F[:] = kept
            while len(F) > r:
                self._reduce(F, side1, side2)
        for e in F:
            (side1 if e.y >= HALF else side2).append((e.piece, e.key))
        return side1, side2

    def _halve(self, e: _Frac) -> List[_Frac]:
        """Cut ``e``'s cell into two halves of equal value for the agent with the largest
        share ``V_i(K) / V_i(R)`` (first in agent order on ties): one prefix of
        prescribed value, i.e. at most one Cut and ``n - 1`` Evals (Section C.2)."""
        best = None
        for i in self.plus:
            share = self.table.value(i, e.piece) / self.total[i]
            if best is None or share > best[0]:
                best = (share, i)
        cutter = best[1]
        half = self.table.value(cutter, e.piece) / 2
        prefix, suffix, (k, y) = self.table.prefix_of_value(e.piece, cutter, half)
        self.cuts += 1
        return [_Frac(prefix, e.y, e.key), _Frac(suffix, e.y, (e.key[0] + k, y))]

    def _reduce(self, F: List[_Frac], side1, side2) -> None:
        """One kernel step on the first ``r + 1`` fractional cells (``len(F) > r``)."""
        r = len(self.plus)
        block = F[: r + 1]
        rows = [[self.table.value(i, e.piece) for e in block] for i in self.plus]
        z = kernel_vector(rows, r + 1)
        t = None
        for e, zj in zip(block, z):
            if zj > 0:
                cand = (1 - e.y) / zj
            elif zj < 0:
                cand = e.y / (-zj)
            else:
                continue
            if t is None or cand < t:
                t = cand
        self.kernel_steps += 1
        keep: List[_Frac] = []
        for e, zj in zip(block, z):
            if zj != 0:
                e.y = e.y + t * zj
            if e.y == 1:
                side1.append((e.piece, e.key))
            elif e.y == 0:
                side2.append((e.piece, e.key))
            else:
                keep.append(e)
        F[: r + 1] = keep

    # ======================================================================
    # Procedure ``paper`` -- the procedure of the proof of Lemma 4.6 (tests only)
    # ======================================================================
    def _paper_procedure(self) -> Tuple[Dict[Hashable, OrderedPiece], OrderedPiece]:
        n = len(self.agents)
        m = len(self.keys)
        if self.gamma_override is not None:
            gamma = Fraction(self.gamma_override)
        else:
            g = float(self.delta) ** 2 / (2.0 * math.log(2 * n * m + 2))
            gamma = Fraction(g * (1 - 1e-9))  # at most the paper's gamma
        if gamma <= 0:
            raise SimultaneousCutError("gamma underflowed")
        if n * math.ceil(1 / gamma) > self.max_fallback_cuts:
            raise SimultaneousCutError(
                f"the procedure of Lemma 4.6 would need about {n * math.ceil(1 / gamma)} cuts "
                f"(gamma = {float(gamma):.3e}); this is beyond the practical limit of this implementation")
        refined = self.residue
        for i in self.agents:
            if self.total[i] == 0:
                continue
            step = gamma * self.total[i]
            k = 1
            rest = refined
            done: List[Interval] = []
            while k * gamma < 1 and self.table.value(i, rest) > step:
                prefix, rest, _ = self.table.prefix_of_value(rest, i, step)
                done.extend(prefix.components)
                k += 1
            refined = OrderedPiece(done + list(rest.components))
        cells = list(refined.components)
        labeling = self._greedy_labeling(cells)
        if labeling is None or not self._labeling_ok(cells, labeling):
            labeling = self._exhaustive_labeling(cells)
        pieces = {k: OrderedPiece([c for c, lab in zip(cells, labeling) if lab == k]) for k in self.keys}
        leftover = OrderedPiece([c for c, lab in zip(cells, labeling) if lab is None])
        return pieces, leftover

    def _labeling_ok(self, cells: List[Interval], labeling: Sequence[Optional[Hashable]]) -> bool:
        pieces = {k: OrderedPiece([c for c, lab in zip(cells, labeling) if lab == k]) for k in self.keys}
        return self.verify(pieces)

    def _greedy_labeling(self, cells: List[Interval]) -> Optional[List[Optional[Hashable]]]:
        """Assign cells one at a time, always minimizing the largest normalized deviation."""
        agents = [i for i in self.agents if self.total[i] > 0]
        acc = {k: {i: ZERO for i in agents} for k in self.keys}
        labeling: List[Optional[Hashable]] = []
        options: List[Optional[Hashable]] = [None] + list(self.keys)
        for c in cells:
            best_opt, best_dev = None, None
            for opt in options:
                dev = ZERO
                for k in self.keys:
                    for i in agents:
                        v = acc[k][i] + (self._v(i, c) if opt == k else ZERO)
                        d = abs(v - self.lam[k] * self.total[i]) / self.total[i]
                        if d > dev:
                            dev = d
                if best_dev is None or dev < best_dev:
                    best_opt, best_dev = opt, dev
            labeling.append(best_opt)
            if best_opt is not None:
                for i in agents:
                    acc[best_opt][i] += self._v(i, c)
        return labeling

    def _exhaustive_labeling(self, cells: List[Interval]) -> List[Optional[Hashable]]:
        """Algorithm 4's search over all labelings, in a fixed order."""
        options: List[Optional[Hashable]] = [None] + list(self.keys)
        if len(options) ** len(cells) > 5_000_000:
            raise SimultaneousCutError(
                f"exhaustive labeling search over {len(cells)} cells and {len(options)} options is infeasible")
        for labeling in itertools.product(options, repeat=len(cells)):
            if self._labeling_ok(cells, labeling):
                return list(labeling)
        raise SimultaneousCutError("no valid labeling exists (gamma too coarse)")


def simultaneous_cut(table: ValueTable, agents: Sequence[Agent], residue: OrderedPiece,
                     requests: Dict[Hashable, Fraction], delta: Fraction, *,
                     budget: Optional[int] = None, force_fallback: bool = False,
                     gamma: Optional[Fraction] = None
                     ) -> Tuple[Dict[Hashable, OrderedPiece], OrderedPiece, SimultaneousCutter]:
    """Lemma 4.6: returns ``(pieces, unused part of the residue, cutter object)``."""
    cutter = SimultaneousCutter(table, agents, residue, requests, delta, budget=budget,
                                force_fallback=force_fallback, gamma=gamma)
    pieces, leftover = cutter.run()
    return pieces, leftover, cutter
