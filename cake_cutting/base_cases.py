"""Base cases of ``Main`` for at most three agents (Ye–Bai, Section 4.3).

* ``n = 1``: the sole agent receives the subcake.
* ``n = 2``: cut-and-choose -- one agent divides the subcake into two pieces she
  values equally, the other chooses a piece she prefers, the first agent
  receives the remaining piece.
* ``n = 3``: the Selfridge–Conway protocol.

"When C has several interval components, we apply these protocols to the fixed
order defined in Section C.1 and implement the resulting queries as explained
there": every division is a prefix of prescribed value and every trim removes a
prefix of the ordered piece (Section C.2 rule), so the pieces produced are
ordered pieces and all values are known from the table.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, Hashable, List, Sequence

from .bookkeeping import ValueTable
from .pieces import OrderedPiece
from .subcore import ProtocolError

Agent = Hashable


def _argmax(values: Sequence[Fraction]) -> int:
    best = max(values)
    return min(i for i, v in enumerate(values) if v == best)


def _equal_pieces(table: ValueTable, piece: OrderedPiece, agent: Agent, k: int) -> List[OrderedPiece]:
    """``agent`` divides ``piece`` into ``k`` ordered pieces of equal value (k-1 prefix cuts)."""
    total = table.value(agent, piece)
    share = total / k
    out: List[OrderedPiece] = []
    rest = piece
    for _ in range(k - 1):
        prefix, rest, _ = table.prefix_of_value(rest, agent, share)
        out.append(prefix)
    out.append(rest)
    return out


def cut_and_choose(table: ValueTable, agents: Sequence[Agent], cake: OrderedPiece) -> Dict[Agent, OrderedPiece]:
    """Two agents: the lower-indexed one cuts, the other chooses (ties -> first piece)."""
    a, b = sorted(agents)  # type: ignore[type-var]
    halves = _equal_pieces(table, cake, a, 2)
    choice = _argmax([table.value(b, h) for h in halves])
    return {b: halves[choice], a: halves[1 - choice]}


def selfridge_conway(table: ValueTable, agents: Sequence[Agent], cake: OrderedPiece) -> Dict[Agent, OrderedPiece]:
    """Selfridge–Conway on an ordered piece (at most 5 cuts, i.e. 15 queries here)."""
    a1, a2, a3 = sorted(agents)  # type: ignore[type-var]
    pieces = _equal_pieces(table, cake, a1, 3)

    # Agent 2 trims her favorite piece down to the value of her second favorite.
    vals2 = [table.value(a2, p) for p in pieces]
    top = _argmax(vals2)
    second = max(v for i, v in enumerate(vals2) if i != top)
    trimmings = OrderedPiece.empty()
    trimmed_index = None
    if vals2[top] > second:
        trimmings, trimmed_piece, _ = table.suffix_of_value(pieces[top], a2, second)
        pieces[top] = trimmed_piece
        trimmed_index = top

    # Choosing: agent 3, then agent 2 (who must take the trimmed piece if it is
    # still there), then agent 1 takes the last piece.
    remaining = list(range(3))
    pick3 = max(remaining, key=lambda i: (table.value(a3, pieces[i]), -i))
    remaining.remove(pick3)
    if trimmed_index is not None and trimmed_index in remaining:
        pick2 = trimmed_index
    else:
        pick2 = max(remaining, key=lambda i: (table.value(a2, pieces[i]), -i))
    remaining.remove(pick2)
    pick1 = remaining[0]
    alloc: Dict[Agent, OrderedPiece] = {a1: pieces[pick1], a2: pieces[pick2], a3: pieces[pick3]}

    if trimmed_index is not None and not trimmings.is_empty:
        # The non-cutter who did NOT receive the trimmed piece divides the
        # trimmings into three equal pieces; the holder of the trimmed piece
        # chooses first, then agent 1, then the divider.
        holder = a2 if pick2 == trimmed_index else a3
        divider = a3 if holder == a2 else a2
        parts = _equal_pieces(table, trimmings, divider, 3)
        left = list(range(3))
        for who in (holder, a1, divider):
            pick = max(left, key=lambda i: (table.value(who, parts[i]), -i))
            left.remove(pick)
            alloc[who] = alloc[who] + parts[pick]
    return alloc


def base_case(table: ValueTable, agents: Sequence[Agent], cake: OrderedPiece) -> Dict[Agent, OrderedPiece]:
    """``Main(N, C)`` for ``|N| <= 3``."""
    agents = sorted(agents)  # type: ignore[type-var]
    if len(agents) == 1:
        return {agents[0]: cake}
    if len(agents) == 2:
        alloc = cut_and_choose(table, agents, cake)
    elif len(agents) == 3:
        alloc = selfridge_conway(table, agents, cake)
    else:
        raise ValueError("base cases handle at most three agents")
    # envy-freeness check from the table (no queries)
    for i in agents:
        vi = table.value(i, alloc[i])
        for j in agents:
            if table.value(i, alloc[j]) > vi:
                raise ProtocolError(f"base case produced envy: {i!r} envies {j!r}")
    return alloc
