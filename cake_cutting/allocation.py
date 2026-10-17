"""Allocations and the ``⊕`` operation of Section 4.1."""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, Hashable, Iterable, Iterator, Mapping, Optional

from .pieces import OrderedPiece, pairwise_disjoint, union_measure

Agent = Hashable


class Allocation:
    """A partial allocation: every agent receives an ordered piece (possibly empty).

    ``X ⊕ Y`` gives every agent the union (concatenation) of her pieces; an
    agent missing from one of the allocations receives the empty piece there
    (Section 4.1).
    """

    def __init__(self, pieces: Optional[Mapping[Agent, OrderedPiece]] = None) -> None:
        self._pieces: Dict[Agent, OrderedPiece] = dict(pieces or {})

    def __getitem__(self, agent: Agent) -> OrderedPiece:
        return self._pieces.get(agent, OrderedPiece.empty())

    def __contains__(self, agent: Agent) -> bool:
        return agent in self._pieces

    def __iter__(self) -> Iterator[Agent]:
        return iter(self._pieces)

    def agents(self) -> Iterable[Agent]:
        return self._pieces.keys()

    def items(self):
        return self._pieces.items()

    def pieces(self) -> Dict[Agent, OrderedPiece]:
        return dict(self._pieces)

    def oplus(self, other: "Allocation") -> "Allocation":
        out = dict(self._pieces)
        for a, p in other._pieces.items():
            out[a] = out[a] + p if a in out else p
        return Allocation(out)

    def __add__(self, other: "Allocation") -> "Allocation":
        return self.oplus(other)

    def give(self, agent: Agent, piece: OrderedPiece) -> "Allocation":
        """Add ``piece`` to ``agent``'s share (returns a new allocation)."""
        return self.oplus(Allocation({agent: piece}))

    # -- certification helpers (exact, used by tests) ---------------------
    def is_pairwise_disjoint(self) -> bool:
        return pairwise_disjoint(list(self._pieces.values()))

    def total_measure(self) -> Fraction:
        return union_measure(list(self._pieces.values()))

    def is_complete(self, cake: OrderedPiece) -> bool:
        """Pieces pairwise disjoint and their union equals ``cake`` up to measure zero."""
        if not self.is_pairwise_disjoint():
            return False
        pieces = list(self._pieces.values())
        if any(c.a < 0 or c.b > 1 for p in pieces for c in p.components):
            return False
        inside = union_measure(pieces + [cake]) == cake.normalized_length()  # pieces ⊆ cake
        return inside and self.total_measure() == cake.normalized_length()

    def __repr__(self) -> str:
        return "Allocation(" + ", ".join(f"{a!r}: {p.pretty()}" for a, p in self._pieces.items()) + ")"
