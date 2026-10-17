"""Single-exponential envy-free cake cutting (Ye & Bai, arXiv:2609.05191).

The package implements the protocol ``Main(N, C)`` of Ye and Bai, *Cutting Down
the Tower: Single-Exponential Envy-Free Cake Cutting*, together with the Core and
(modified) SubCore subroutines of Aziz and Mackenzie that it relies on.  All
arithmetic is exact (``fractions.Fraction``); agents are accessed exclusively
through a Robertson–Webb oracle that counts Eval and Cut queries.

Module map (paper section -> module):

* Section 2 / Appendix C.1 (queries, ordered pieces)  -> :mod:`cake_cutting.pieces`,
  :mod:`cake_cutting.oracle`, :mod:`cake_cutting.bookkeeping`
* Section 4.1, Appendix D (Core / SubCore, Props 4.3-4.5) -> :mod:`cake_cutting.subcore`
* Section 4.2, Algorithm 4 (Lemma 4.6)                 -> :mod:`cake_cutting.simultaneous`
* Section 4.3 base cases (n <= 3)                       -> :mod:`cake_cutting.base_cases`
* Sections 4.3-4.6, Algorithms 1-3 (Main)              -> :mod:`cake_cutting.main`
* Section 4.6 / Prop 4.21 (explicit query bound)        -> :mod:`cake_cutting.bounds`
* valuations used only by tests and the demo            -> :mod:`cake_cutting.valuations`
"""

from .allocation import Allocation
from .main import MainProtocol, run_protocol
from .oracle import RobertsonWebbOracle, QueryError, QueryBudgetExceeded
from .pieces import Interval, OrderedPiece
from .subcore import ProtocolError
from .valuations import PiecewiseConstantValuation

__all__ = [
    "Allocation",
    "Interval",
    "MainProtocol",
    "OrderedPiece",
    "PiecewiseConstantValuation",
    "ProtocolError",
    "QueryBudgetExceeded",
    "QueryError",
    "RobertsonWebbOracle",
    "run_protocol",
]
