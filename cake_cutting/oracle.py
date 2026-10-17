"""The Robertson–Webb query oracle (Section 2 of Ye–Bai).

The protocol never touches a valuation directly; it calls

* ``eval(i, x, y)``  -- returns ``V_i([x, y])``;
* ``cut(i, x, a)``   -- returns ``y >= x`` with ``V_i([x, y]) = a`` (``0 <= a <= V_i([x, 1])``).

Every call is validated, counted (per agent and in total) and optionally logged.
A query budget can be set so that a runaway computation fails fast instead of
hanging; the protocol itself never relies on the budget.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, Hashable, List, Optional, Tuple

Agent = Hashable


class QueryError(ValueError):
    """An ill-formed Robertson–Webb query."""


class QueryBudgetExceeded(RuntimeError):
    """The oracle's query budget was exhausted."""


class RobertsonWebbOracle:
    """Counts Eval and Cut queries to a set of agents.

    ``valuations`` maps an agent id to an object with methods ``value(x, y)`` and
    ``cut(x, a, rule)`` (see :mod:`cake_cutting.valuations`).  ``cut_rule`` selects
    which of the admissible answers a Cut query returns ("left" = smallest point,
    "right" = largest point); both are valid oracles for the protocol.
    """

    def __init__(self, valuations: Dict[Agent, object], *, cut_rule: str = "left",
                 budget: Optional[int] = None, log: bool = False) -> None:
        self._valuations = dict(valuations)
        self.cut_rule = cut_rule
        self.budget = budget
        self.evals = 0
        self.cuts = 0
        self.evals_by_agent: Dict[Agent, int] = {a: 0 for a in self._valuations}
        self.cuts_by_agent: Dict[Agent, int] = {a: 0 for a in self._valuations}
        self.log: Optional[List[Tuple]] = [] if log else None

    # -- bookkeeping ------------------------------------------------------
    @property
    def agents(self) -> List[Agent]:
        return list(self._valuations)

    @property
    def queries(self) -> int:
        return self.evals + self.cuts

    def _charge(self) -> None:
        if self.budget is not None and self.queries >= self.budget:
            raise QueryBudgetExceeded(f"query budget {self.budget} exhausted")

    # -- the two Robertson–Webb queries -----------------------------------
    def eval(self, agent: Agent, x: Fraction, y: Fraction) -> Fraction:
        """``Eval_i(x, y) = V_i([x, y])``."""
        if agent not in self._valuations:
            raise QueryError(f"unknown agent {agent!r}")
        if not (0 <= x <= y <= 1):
            raise QueryError(f"Eval on invalid interval [{x}, {y}]")
        self._charge()
        self.evals += 1
        self.evals_by_agent[agent] += 1
        v = Fraction(self._valuations[agent].value(x, y))
        if self.log is not None:
            self.log.append(("eval", agent, x, y, v))
        return v

    def cut(self, agent: Agent, x: Fraction, a: Fraction) -> Fraction:
        """``Cut_i(x, a)``: a point ``y >= x`` with ``V_i([x, y]) = a``."""
        if agent not in self._valuations:
            raise QueryError(f"unknown agent {agent!r}")
        if not (0 <= x <= 1):
            raise QueryError(f"Cut from {x}: outside the cake")
        if a < 0:
            raise QueryError(f"Cut for negative value {a}")
        val = self._valuations[agent]
        if a > val.value(x, Fraction(1)):
            raise QueryError(f"Cut for value {a} exceeds V_i([{x}, 1]) = {val.value(x, Fraction(1))}")
        self._charge()
        self.cuts += 1
        self.cuts_by_agent[agent] += 1
        y = Fraction(val.cut(x, a, self.cut_rule))
        if self.log is not None:
            self.log.append(("cut", agent, x, a, y))
        assert x <= y <= 1 and val.value(x, y) == a, "oracle returned an invalid cut point"
        return y

    # -- test-only access -------------------------------------------------
    def true_value(self, agent: Agent, piece) -> Fraction:
        """The true valuation of a piece.  **For tests and demos only** -- the
        protocol must never call this (it would bypass the query model)."""
        val = self._valuations[agent]
        return sum((Fraction(val.value(c.a, c.b)) for c in piece.components), Fraction(0))
