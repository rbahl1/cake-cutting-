"""Core and the modified SubCore (Ye–Bai Section 4.1, Appendix D; AM Algorithms 1-2).

``Core(N, R, c)`` (Proposition 4.3): the cutter ``c`` divides the residue ``R``
into ``n`` pieces she values equally; SubCore allocates (possibly trimmed)
pieces to the other agents; ``c`` takes an untrimmed unallocated piece.  The
cutter and at least one other agent receive whole pieces, hence
``V_c(R') <= (n-2)/n * V_c(R)``.

``SubCore`` is Aziz–Mackenzie's SubCore (extended arXiv version, with
benchmarks / required minima) with the modification of Ye–Bai, Appendix D,
"Omitted proof of Proposition 4.4": AM's line 12 (the final recursive call made
once ``|W| = m-1``) is skipped.  If ``|W| = m-1`` as soon as the trims are made,
every contested piece has a different rightmost trimmer and each winner receives
the part beginning at her own trim; if the while loop runs, the tentative
allocation of its final iteration is kept.  Consequently SubCore on ``m`` agents
makes at most ``S_m = 2^{m+1} - 2m - 2`` trims (cuts): introducing the ``m``-th
agent costs at most ``m(m-1)`` trims, and the recursive calls inside the while
loop have distinct sizes in ``{1, ..., m-2}``.

Deterministic choices.  Section C.2 refers to AM's tie-breaking by "imaginary
values" (AM, Section 3.1.1).  That scheme is not specified precisely enough to
be implemented literally, and Bechler–Bitikofer (2017) showed that the literal
protocol can reach "Case 2" of AM's Claim 8 (an unallocated contested piece
without a non-winner trim), which AM's proof declares impossible.  The proof
needs two properties that the text only asserts: an agent kicked out to a
piece ``a`` keeps preferring ``a`` over physically equal pieces for the rest of
the step, and the tentative allocation of the previous iteration of the while
loop stays attainable.  The concrete realization used here supplies both via
*hints* passed to the recursive calls -- ``(identity, guaranteed suffix)`` where
the identity is the piece the agent holds at the start of the current step of
the enclosing call (none for the newcomer) and the guaranteed suffix is the part
of a piece that the enclosing call's tentative allocation gives her:

* An agent's favorite piece is one of maximum value; among tied unallocated
  pieces she prefers her identity piece, then the lowest index.
* The loser of a contest takes her favorite uncontested piece; among ties, her
  identity piece, then the lowest index (AM's "ties broken in favour of a").
* When an agent trims the piece she holds (or is guaranteed) to the value of
  that holding, the trim is placed exactly at the start of the holding.
* Ties between trims at the same position of a piece are won by (1) an agent
  whose guaranteed suffix is exactly the suffix from that position, (2) the
  agent whose identity is the piece, (3) the agent tentatively holding the
  piece in this call, (4) the lowest index.
* Trim positions are compared after moving them left over cake the trimmer is
  known to value at zero, so the comparison does not depend on which admissible
  point a Cut query returned.
* The cutter in Core takes the lowest-index unallocated piece; an unallocated
  contested piece carrying a non-winner trim is chosen by lowest index.

Every postcondition (required minima, envy-freeness, neatness, distinct pieces)
is verified at the end of each call *from the value table*, i.e. without extra
queries; a violation raises :class:`ProtocolError` instead of returning a wrong
allocation.  "Case 2" is likewise detected and reported, never silently repaired.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

from . import equilibrium
from .bookkeeping import Position, ValueTable
from .pieces import Interval, OrderedPiece, ZERO

Agent = Hashable

#: number of SubCore calls that used the equilibrium fallback (for tests and stats)
FALLBACKS = [0]


class ProtocolError(RuntimeError):
    """A checked guarantee failed at run time."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def suffix_start_position(table: ValueTable, piece: OrderedPiece, cand: OrderedPiece
                          ) -> Optional[Tuple[Position, Interval]]:
    """Position in ``piece`` at which ``cand`` starts, for a nonempty ``cand`` that ends
    where ``piece`` ends, together with the component of ``piece`` containing it
    (``None`` if ``cand`` does not start inside ``piece``).

    Pieces of one SubCore family are pairwise disjoint and each is a suffix of a
    piece of the top-level call, so two nonempty pieces of the family with the
    same last end are suffixes of the same top-level piece; ``cand`` then starts in
    component ``len(piece) - len(cand)`` of ``piece`` if it starts inside ``piece``
    at all.  Equivalent to scanning ``piece`` for the component containing the
    first point of ``cand``, in ``O(log C)``.
    """
    k = len(piece) - len(cand)
    if k < 0:
        return None
    c = table.nth_component(piece, k)
    point = cand.first.a
    if c.a <= point < c.b:
        return (k, point), c
    return None


# ---------------------------------------------------------------------------
# SubCore
# ---------------------------------------------------------------------------

class SubCoreCall:
    """One (possibly recursive) SubCore call -- AM Algorithm 2 as modified by Ye–Bai."""

    def __init__(self, table: ValueTable, pieces: Sequence[OrderedPiece], agents: Sequence[Agent],
                 benchmarks: Dict[Agent, Fraction],
                 hints: Optional[Dict[Agent, Tuple[Optional[OrderedPiece], OrderedPiece]]] = None,
                 depth: int = 0) -> None:
        self.table = table
        self.pieces = list(pieces)
        self.agents = list(agents)
        self.benchmarks = dict(benchmarks)
        # hint: (identity, guaranteed suffix) or None.  The identity is the piece the agent
        # holds at the start of the current step of the enclosing call (she prefers it among
        # physically equal pieces); the guaranteed suffix is the part of a piece that the
        # enclosing call's tentative allocation gives her.  They are kept separately: the
        # guaranteed suffix may be empty (a trim "to value zero" at the end of the piece)
        # and may lie on a different piece than the identity.
        self.hints: Dict[Agent, Optional[Tuple[Optional[OrderedPiece], OrderedPiece]]] = {
            a: (hints or {}).get(a) for a in self.agents}
        self.depth = depth
        if len(self.agents) > len(self.pieces) - 1:
            raise ValueError("SubCore needs at least one more piece than agents")
        self.order = {a: k for k, a in enumerate(self.agents)}
        # tentative allocation: agent -> (piece index, suffix); piece index -> agent
        self.hold: Dict[Agent, Tuple[int, OrderedPiece]] = {}
        self.holder: Dict[int, Agent] = {}
        self.trims_made = 0  # number of trims placed (each is at most one Cut query)

    # -- values ---------------------------------------------------------
    def val(self, agent: Agent, piece: OrderedPiece) -> Fraction:
        return self.table.value(agent, piece)

    def hint_index(self, agent: Agent) -> Optional[int]:
        """Index of the piece (of this call) that the agent's hint is a suffix of.

        Pieces of one call are pairwise disjoint, so the end of the last
        component identifies a piece; the current margin of the piece may lie
        to the right of the start of the hint (the hint is then a superset of
        the piece), which still counts as "her piece".
        """
        h = self.hints.get(agent)
        if h is None or h[0] is None or h[0].is_empty:
            return None
        end = h[0].last.b
        for idx, p in enumerate(self.pieces):
            if len(p) > 0 and p.last.b == end:
                return idx
        return None

    def hint_suffix(self, agent: Agent) -> Optional[OrderedPiece]:
        h = self.hints.get(agent)
        return None if h is None else h[1]

    # -- favorite -------------------------------------------------------
    def favorite_unallocated(self, agent: Agent) -> Optional[int]:
        """If some unallocated piece attains the agent's maximum value, return it."""
        best = max(self.val(agent, p) for p in self.pieces)
        cands = [idx for idx, p in enumerate(self.pieces) if idx not in self.holder and self.val(agent, p) == best]
        if not cands:
            return None
        h = self.hint_index(agent)
        if h in cands:
            return h
        return cands[0]

    # -- trims ----------------------------------------------------------
    def place_trim(self, agent: Agent, idx: int, beta: Fraction) -> Position:
        """Trim piece ``idx`` so that the remaining suffix is worth ``beta`` to ``agent``."""
        piece = self.pieces[idx]
        held = self.hold.get(agent)
        hint = self.hint_suffix(agent)
        # "imaginary values": trimming the piece one holds (or was guaranteed) to
        # the value of that holding yields exactly that holding -- the smallest
        # such suffix if both the current holding and the hint qualify.
        best: Optional[Position] = None
        for cand in ((held[1] if held and held[0] == idx else None), hint):
            if cand is not None and not cand.is_empty and len(piece) > 0 \
                    and cand.last.b == piece.last.b:
                found = suffix_start_position(self.table, piece, cand)
                if found is not None:
                    pos, c = found
                    # suffix_from(piece, pos) has len(cand) components and first component
                    # [pos[1], c.b); it equals cand iff cand's first component ends at c.b
                    if cand.first.b == c.b and self.val(agent, cand) == beta \
                            and (best is None or pos > best):
                        best = pos
        if best is not None:
            return best
        self.trims_made += 1
        pos = self.table.suffix_position_of_value(piece, agent, beta)
        # A trim is compared with other trims by position.  Cake that the agent
        # is known to value at zero just before her trim must not decide a
        # contest in her favour (nor against her), so the position is moved
        # left over such cake; the suffix keeps the value beta for her.
        return self.table.normalize_left(piece, agent, pos)

    def rightmost(self, trims: Dict[Tuple[Agent, int], Position], idx: int, among: Sequence[Agent]
                  ) -> Tuple[Optional[Position], List[Agent]]:
        best: Optional[Position] = None
        tied: List[Agent] = []
        for j in among:
            pos = trims.get((j, idx))
            if pos is None:
                continue
            if best is None or pos > best:
                best, tied = pos, [j]
            elif pos == best:
                tied.append(j)
        return best, tied

    def break_tie(self, idx: int, tied: Sequence[Agent], pos: Position) -> Agent:
        """Ties between trims at the same position of piece ``idx``.

        1. an agent whose guaranteed suffix (her tentative piece in the enclosing
           call) is exactly the suffix from the tie position -- she is merely
           reclaiming what the previous tentative allocation gave her, and the
           benchmark invariant of AM's induction needs her to keep it;
        2. the agent whose hint identity is this piece;
        3. the agent tentatively holding the piece in this call;
        4. the lowest index.
        """
        piece = self.pieces[idx]
        k = pos[0]
        if k < len(piece):
            # the suffix from pos: len(piece) - k components, first component [pos[1], c.b)
            c = self.table.nth_component(piece, k)
            want_len, want_first = len(piece) - k, (pos[1], c.b)
            for a in tied:
                hs = self.hint_suffix(a)
                if hs is not None and not hs.is_empty and len(hs) == want_len \
                        and (hs.first.a, hs.first.b) == want_first:
                    return a
        for a in tied:
            if self.hint_index(a) == idx:
                return a
        h = self.holder.get(idx)
        if h in tied:
            return h
        return min(tied, key=lambda a: self.order[a])

    # -- the main loop ----------------------------------------------------
    def run(self) -> Dict[Agent, Tuple[int, OrderedPiece]]:
        for step, agent in enumerate(self.agents):
            fav = self.favorite_unallocated(agent)
            if fav is not None:
                self.hold[agent] = (fav, self.pieces[fav])
                self.holder[fav] = agent
                continue
            self.contest(step)
        self.verify()
        return self.hold

    def contest(self, step: int) -> None:
        group = self.agents[: step + 1]
        contested = sorted(self.hold[a][0] for a in group[:-1])  # the m-1 tentatively allocated pieces
        uncontested = [idx for idx in range(len(self.pieces)) if idx not in self.holder]
        assert len(contested) == step and uncontested

        beta: Dict[Agent, Fraction] = {}
        for j in group:
            best_u = max(self.val(j, self.pieces[u]) for u in uncontested)
            beta[j] = max(self.benchmarks[j], best_u)

        trims: Dict[Tuple[Agent, int], Position] = {}
        for j in group:
            for idx in contested:
                if self.val(j, self.pieces[idx]) >= beta[j]:
                    trims[(j, idx)] = self.place_trim(j, idx, beta[j])

        winners: set = set()
        winner_of: Dict[int, Agent] = {}
        for idx in contested:
            pos, tied = self.rightmost(trims, idx, group)
            if pos is None:
                raise ProtocolError("a contested piece carries no trim (AM Claim A violated)")
            w = self.break_tie(idx, tied, pos)
            winners.add(w)
            winner_of[idx] = w

        new_hold: Dict[Agent, Tuple[int, OrderedPiece]] = {}
        if len(winners) == len(contested):
            # Ye–Bai: give each contested piece to its rightmost trimmer, from her trim.
            for idx, w in winner_of.items():
                new_hold[w] = (idx, self.table.suffix_from(self.pieces[idx], trims[(w, idx)]))
        else:
            tentative: Dict[Agent, Tuple[int, OrderedPiece]] = {}
            while len(winners) < len(contested):
                nonwinners = [j for j in group if j not in winners]
                margins: Dict[int, OrderedPiece] = {}
                for idx in contested:
                    pos, _ = self.rightmost(trims, idx, nonwinners)
                    margins[idx] = self.pieces[idx] if pos is None else self.table.suffix_from(self.pieces[idx], pos)
                sub_agents = [a for a in self.agents if a in winners]
                # Each winner's hint for the recursive call: the piece she holds at the
                # start of this step (the piece she took as her favourite or was kicked
                # out to), together with the suffix of it she is guaranteed (from her
                # own trim on it if she placed one, else her current holding).  AM's
                # proof of Claim 8 (Case 2) needs every agent to keep preferring that
                # piece over physically equal ones throughout the step, which the
                # literal protocol does not ensure (Bechler–Bitikofer); the hint is
                # therefore the same in every iteration of the while loop.  The newcomer
                # (agent m, who holds nothing yet) has no identity hint; the guaranteed
                # suffix is her tentative piece once she has one, else the suffix from
                # her trim on the lowest-index piece she won.
                sub_hints: Dict[Agent, Tuple[Optional[OrderedPiece], OrderedPiece]] = {}
                for a in sub_agents:
                    identity: Optional[OrderedPiece] = None
                    if a in self.hold:
                        hidx = self.hold[a][0]
                        identity = self.pieces[hidx]
                        if a in tentative:
                            suffix = tentative[a][1]
                        elif (a, hidx) in trims:
                            suffix = self.table.suffix_from(self.pieces[hidx], trims[(a, hidx)])
                        else:
                            suffix = self.hold[a][1]
                    elif a in tentative:
                        suffix = tentative[a][1]
                    else:
                        hidx = min(idx for idx in contested if winner_of[idx] == a)
                        suffix = self.table.suffix_from(self.pieces[hidx], trims[(a, hidx)])
                    sub_hints[a] = (identity, suffix)
                sub = SubCoreCall(self.table, [margins[idx] for idx in contested], sub_agents,
                                  {a: beta[a] for a in sub_agents}, sub_hints, self.depth + 1)
                result = sub.run()
                self.trims_made += sub.trims_made
                tentative = {a: (contested[local], suffix) for a, (local, suffix) in result.items()}
                taken = {p for p, _ in tentative.values()}
                added = False
                for idx in contested:
                    if idx in taken:
                        continue
                    pos, tied = self.rightmost(trims, idx, nonwinners)
                    if pos is None:
                        continue
                    i = self.break_tie(idx, tied, pos)
                    winners.add(i)
                    tentative[i] = (idx, self.table.suffix_from(self.pieces[idx], pos))
                    added = True
                    break
                if not added:
                    raise ProtocolError(
                        "SubCore: no unallocated contested piece carries a non-winner trim "
                        "(the situation of the Bechler–Bitikofer counterexample)")
                for j in winners:
                    beta[j] = self.val(j, tentative[j][1])
            new_hold = tentative

        losers = [j for j in group if j not in winners]
        assert len(losers) == 1
        loser = losers[0]
        best_u = max(self.val(loser, self.pieces[u]) for u in uncontested)
        cands = [idx for idx in uncontested if self.val(loser, self.pieces[idx]) == best_u]
        # AM, proof of Claim 8 (Case 2): an agent kicked out to a piece keeps preferring that
        # piece over physically equal ones ("ties broken in favour of a"), so the piece she was
        # kicked out to in the enclosing call (her hint) wins ties; otherwise the lowest index.
        h = self.hint_index(loser)
        u = h if h in cands else cands[0]
        new_hold[loser] = (u, self.pieces[u])

        self.hold = new_hold
        self.holder = {idx: a for a, (idx, _) in self.hold.items()}
        if len(self.holder) != len(self.hold):
            raise ProtocolError("SubCore assigned one piece to two agents")

    # -- verification (no queries) ----------------------------------------
    def verify(self) -> None:
        for j, (idx, suffix) in self.hold.items():
            if not _is_suffix(self.table, self.pieces[idx], suffix):
                raise ProtocolError("SubCore allocated something that is not a suffix of its piece")
            v = self.val(j, suffix)
            if v < self.benchmarks[j]:
                raise ProtocolError(f"SubCore: agent {j!r} received {v} < required minimum {self.benchmarks[j]}")
            for k, (_, other) in self.hold.items():
                if k != j and self.val(j, other) > v:
                    raise ProtocolError(f"SubCore: agent {j!r} envies agent {k!r}")
            for idx2, p in enumerate(self.pieces):
                if idx2 not in self.holder and self.val(j, p) > v:
                    raise ProtocolError(f"SubCore: agent {j!r} prefers the unallocated piece {idx2}")


def _is_suffix(table: ValueTable, piece: OrderedPiece, suffix: OrderedPiece) -> bool:
    """``suffix`` is a suffix of ``piece`` (both from one SubCore family, see
    :func:`suffix_start_position`): it ends where ``piece`` ends and starts inside
    component ``len(piece) - len(suffix)`` of ``piece``."""
    if suffix.is_empty:
        return True
    if len(suffix) > len(piece) or suffix.last.b != piece.last.b:
        return False
    first_p = table.nth_component(piece, len(piece) - len(suffix))
    first_s = suffix.first
    return first_p.b == first_s.b and first_p.a <= first_s.a


def subcore(table: ValueTable, pieces: Sequence[OrderedPiece], agents: Sequence[Agent],
            benchmarks: Dict[Agent, Fraction]) -> Tuple[Dict[Agent, Tuple[int, OrderedPiece]], int]:
    """Run SubCore; returns ``(allocation, number_of_trims)``.

    ``allocation[j] = (piece index, suffix of that piece given to j)``.

    Aziz–Mackenzie's procedure runs first.  If one of its checks fails
    (``CORRECTNESS.md`` §7), the output is computed from a competitive
    equilibrium instead (:func:`cake_cutting.equilibrium.allocate`), which always
    succeeds for valuations with positive density (``CORRECTNESS.md`` §8); the
    trims of the failed attempt are counted with the cuts of the equilibrium.
    """
    call = SubCoreCall(table, pieces, agents, benchmarks)
    try:
        result = call.run()
    except ProtocolError as exc:
        result2, calls = equilibrium.allocate(table, pieces, agents, benchmarks)
        FALLBACKS[0] += 1
        if result2 is None:
            raise ProtocolError(f"{exc}; the equilibrium fallback failed as well "
                                "(possible only with zero-density stretches)") from exc
        return result2, call.trims_made + calls
    return result, call.trims_made


def unallocated_part(table: ValueTable, piece: OrderedPiece, suffix: OrderedPiece) -> OrderedPiece:
    """The prefix of ``piece`` that is not in the allocated ``suffix`` (values by subtraction)."""
    if suffix.is_empty:
        return piece
    k = len(piece) - len(suffix)
    first_p, first_s = table.nth_component(piece, k), suffix.first
    if first_s.a == first_p.a:
        return piece.prefix_at(k)
    left = table.complement_prefix(first_p, first_s)      # records the values, no queries
    return table.prefix_to(piece, (k, left.b))


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def equal_division(table: ValueTable, residue: OrderedPiece, cutter: Agent, n: int) -> List[OrderedPiece]:
    """The cutter divides ``residue`` into ``n`` ordered pieces of equal value.

    Section C.2: if she values the residue at zero, a fixed rule is used (the
    whole residue becomes piece 1, the other pieces are empty); she values all
    pieces equally in either case.
    """
    total = table.value(cutter, residue)
    if total == 0:
        return [residue] + [OrderedPiece.empty() for _ in range(n - 1)]
    pieces: List[OrderedPiece] = []
    rest = residue
    share = total / n
    for k in range(n - 1):
        prefix, rest, _ = table.prefix_of_value(rest, cutter, share)
        pieces.append(prefix)
    pieces.append(rest)
    return pieces


def core(table: ValueTable, agents: Sequence[Agent], residue: OrderedPiece, cutter: Agent
         ) -> Tuple[Dict[Agent, OrderedPiece], OrderedPiece, int]:
    """``Core(N, R, c)`` -- returns ``(allocation, new residue, number of trims)``.

    The allocation gives every agent of ``N`` a (possibly empty) ordered piece;
    the residue keeps the unallocated parts of the ``n`` pieces in piece order.
    """
    agents = list(agents)
    n = len(agents)
    if cutter not in agents:
        raise ValueError("the cutter must be one of the agents")
    pieces = equal_division(table, residue, cutter, n)
    others = [a for a in agents if a != cutter]
    alloc_idx, trims = subcore(table, pieces, others, {a: ZERO for a in others})
    allocation: Dict[Agent, OrderedPiece] = {}
    used = set()
    for a, (idx, suffix) in alloc_idx.items():
        allocation[a] = suffix
        used.add(idx)
    free = [idx for idx in range(n) if idx not in used]
    cutter_idx = free[0]
    allocation[cutter] = pieces[cutter_idx]
    used.add(cutter_idx)
    leftovers: List[OrderedPiece] = []
    for idx in range(n):
        if idx == cutter_idx:
            continue
        holder = [a for a, (i2, _) in alloc_idx.items() if i2 == idx]
        if holder:
            leftovers.append(unallocated_part(table, pieces[idx], allocation[holder[0]]))
        else:
            leftovers.append(pieces[idx])
    new_residue = OrderedPiece.concat(leftovers)
    # Proposition 4.3 (i)/(ii) sanity (no queries): the cutter's residue value shrinks.
    if n >= 2 and table.value(cutter, new_residue) > Fraction(n - 2, n) * table.value(cutter, residue):
        raise ProtocolError("Core: the cutter's residue did not shrink by the factor (n-2)/n")
    return allocation, new_residue, trims
