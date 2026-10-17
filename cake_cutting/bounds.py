"""Explicit Robertson–Webb query accounting (Ye–Bai Props 4.3, 4.4, 4.21, C.1; Sec. 4.6).

The paper proves ``T(n) <= n^{O(1)} 2^n`` without naming the constants.  This
module pins every constant down by following the paper's accounting for the
implementation, so that the tests can assert ``queries <= query_bound(n)`` for a
concrete number.  SubCore may fall back to an equilibrium computation
(``equilibrium.py``, ``CORRECTNESS.md`` §8) whose cuts grow like ``4^n``; with it the
accounting satisfies the closed form

    query_bound(n) <= MONOMIAL_C * n**MONOMIAL_K * 4**n

with ``MONOMIAL_K = 11`` and ``MONOMIAL_C = 1/4`` (attained at ``n = 1``; for
``n >= 2`` the ratio is at most ``MONOMIAL_C2 = 0.047556...``, attained at
``n = 34``), and 11 is the smallest integer exponent for which any constant
works (see :data:`MONOMIAL_C`).  ``PROOF.md`` §9 has the proof.

* a *split* of one interval component (one Cut by one agent, or one point
  chosen by the protocol) costs ``n`` queries for ``n`` active agents --
  one Cut plus ``n-1`` Evals, or ``n`` Evals (Proposition C.1);
* SubCore on ``m`` agents makes at most ``S_m = 2^{m+1} - 2m - 2`` trims
  (Appendix D, proof of Proposition 4.4), also when it fails, and the
  equilibrium fallback on ``k`` nonempty pieces at most
  ``E(m, k) = (m-1) k C(m+k-3, m-2)`` cuts;
* Core on ``n`` agents makes ``n-1`` cutter cuts and at most
  ``S_{n-1} + E(n-1, n)`` trims and cuts;
* Stage 1 makes ``nQ^2`` snapshot Core calls plus at most ``nQ^3`` prescribed
  sequences of at most ``D_mid`` calls (Lemma 4.7);
* Stage 2's shortcut makes at most ``n`` sequences of ``D_2`` calls (Lemma 4.10),
  Stage 3 at most ``n`` sequences of ``D_3`` calls (Section 4.6);
* Lemma 4.6 (the rounding procedure of ``simultaneous``) makes at most
  ``m n^2 ceil(log2(n D / (2 delta)))`` cuts for ``m`` requests and ``n`` agents
  (``D = ceil(log2(m + 1))``), each costing at most ``n`` queries
  (:func:`lemma46_cuts`, :func:`lemma46_queries`);
* the SubCore calls of Lemma 4.17 make at most ``S_n + E(n, n)`` trims and cuts in
  total (``S`` and ``e -> E(e, e)`` are superadditive);
* the top level evaluates the cake once per agent; the base cases use at most
  2 (``n = 2``) or 15 (``n = 3``) queries in this ordered-piece implementation.

All other work (evaluating snapshot pieces, marks, graphs, matchings, the
labeling search of Lemma 4.6, the equilibrium recursion's utilities) uses only
values already stored in the table.
"""

from __future__ import annotations

import functools
import math
from fractions import Fraction
from typing import Dict


def S(m: int) -> int:
    """Maximum number of trims (cuts) made by SubCore on ``m`` agents."""
    if m <= 1:
        return 0
    return 2 ** (m + 1) - 2 * m - 2


def binom(a: int, b: int) -> int:
    """The binomial coefficient ``C(a, b)`` (0 unless ``0 <= b <= a``), by ``min(b, a-b)``
    exact multiplications and divisions."""
    if b < 0 or b > a:
        return 0
    b = min(b, a - b)
    r = 1
    for t in range(1, b + 1):
        r = r * (a - b + t) // t
    return r


def E(m: int, k: int) -> int:
    """Maximum number of cuts of the equilibrium fallback of SubCore on ``m`` agents and
    ``k`` nonempty pieces: ``(m-1) k C(m+k-3, m-2)`` (``CORRECTNESS.md`` §8.5)."""
    if m <= 1 or k <= 0:
        return 0
    return (m - 1) * k * binom(m + k - 3, m - 2)


def subcore_cuts(m: int, k: int) -> int:
    """Cuts of one SubCore call on ``m`` agents and at most ``k`` nonempty pieces: the
    trims of Aziz–Mackenzie's procedure (``S_m``, also when it fails) plus the cuts of
    the equilibrium fallback."""
    return S(m) + E(m, k)


def core_queries(n: int) -> int:
    """Queries of one ``Core`` call with ``n`` active agents (``n-1`` cuts + the SubCore
    call on ``n-1`` agents and ``n`` pieces)."""
    if n <= 1:
        return 0
    return n * ((n - 1) + subcore_cuts(n - 1, n))


def subcore_queries(m: int, active: int) -> int:
    """Queries of a direct SubCore call on ``m`` agents (``m`` pieces ``W_h`` and one empty
    piece) while ``active`` agents are bookkept."""
    return active * subcore_cuts(m, m)


def Q_of(n: int) -> int:
    return n * (n - 1)


def ceil_log(n: int, beta: Fraction) -> int:
    """``ceil(log_{n/(n-2)} beta)`` computed exactly for ``n >= 3`` and ``beta >= 1``."""
    beta = Fraction(beta)
    if beta <= 1:
        return 0
    d = 0
    ratio = Fraction(n, n - 2)
    cur = Fraction(1)
    while cur < beta:
        cur *= ratio
        d += 1
    return d


def D_mid(n: int) -> int:
    """Length of the prescribed sequences eliminating intermediate bonuses (Section 4.3)."""
    Q = Q_of(n)
    return ceil_log(n, Fraction(4 * n * n * (Q + 1)))


def D_shortcut(n: int) -> int:
    """Length of the prescribed sequences in the sink-SCC shortcut (Lemma 4.10)."""
    Q = Q_of(n)
    return ceil_log(n, Fraction(Q * Q))


def D_final(n: int) -> int:
    """Length of the prescribed sequences before the recursion (Section 4.6)."""
    Q = Q_of(n)
    return ceil_log(n, Fraction(4 * n * (Q + 1) * Q * Q))


def stage3_delta(n: int) -> Fraction:
    """``delta = L / (16 n (Q+1))`` with ``L = 1/(nQ^2)`` (Section 4.5)."""
    Q = Q_of(n)
    L = Fraction(1, n * Q * Q)
    return L / (16 * n * (Q + 1))


def max_requests(n: int) -> int:
    """Number of pieces requested from Lemma 4.6: ``nQ`` additions and at most ``n-1`` rank pieces."""
    return n * Q_of(n) + (n - 1)


def ceil_log2(x) -> int:
    """``ceil(log2(x))`` for a rational ``x >= 1``, computed exactly (0 for ``x <= 1``)."""
    x = Fraction(x)
    if x <= 1:
        return 0
    a, b = x.numerator, x.denominator
    e = (a // b).bit_length()          # 2^e > a // b, so 2^e >= a / b up to the remainder
    while e > 0 and (b << (e - 1)) >= a:
        e -= 1
    while (b << e) < a:
        e += 1
    return e


def lemma46_depth(m: int) -> int:
    """Depth ``ceil(log2(m + 1))`` of the label tree of the rounding procedure with
    ``m >= 1`` nonempty requests (``m`` request leaves and one "unused" leaf)."""
    return m.bit_length()


def lemma46_cuts(n: int, m: int, delta: Fraction) -> int:
    """Upper bound on the cuts of one call of the rounding procedure of Lemma 4.6 with
    ``n`` agents and ``m`` requests: ``m n^2 ceil(log2(n D / (2 delta)))``,
    ``D = lemma46_depth(m)`` (proof in the module docstring of ``simultaneous``
    and ``PROOF.md`` §6)."""
    if m < 1 or n < 1:
        return 0
    D = lemma46_depth(m)
    return m * n * n * ceil_log2(Fraction(n * D) / (2 * Fraction(delta)))


def lemma46_queries(n: int, m: int, delta: Fraction) -> int:
    """Queries of one Lemma 4.6 call: every cut is one Cut plus at most ``n - 1`` Evals."""
    return n * lemma46_cuts(n, m, delta)


def lemma46_budget(n: int) -> int:
    """Queries of the Lemma 4.6 call of one ``Main`` call with ``n`` agents: ``m`` is at
    most ``max_requests(n)`` and ``delta = stage3_delta(n)``."""
    return lemma46_queries(n, max_requests(n), stage3_delta(n))


def core_calls_bound(n: int) -> Dict[str, int]:
    """Maximum numbers of Core calls per stage of one ``Main`` call with ``n >= 4`` agents."""
    Q = Q_of(n)
    return {
        "snapshots": n * Q * Q,
        "stage1_shrinks": n * Q ** 3 * D_mid(n),
        "stage2_shrinks": n * D_shortcut(n),
        "stage3_shrinks": n * D_final(n),
    }


@functools.lru_cache(maxsize=None)
def lambda_n(n: int) -> int:
    """``Lambda_n``: queries of one ``Main`` call with ``n >= 4`` agents outside its recursive call."""
    if n < 4:
        raise ValueError("Lambda_n is defined for n >= 4")
    calls = core_calls_bound(n)
    total_core_calls = sum(calls.values())
    return (core_queries(n) * total_core_calls
            + lemma46_budget(n)
            + subcore_queries(n, n))


BASE_CASE_QUERIES = {1: 0, 2: 2, 3: 15}

# Proved bound query_bound(n) <= MONOMIAL_C * n**11 * 4**n (PROOF.md §9).
# The ratio query_bound(n) / (n^11 4^n) is computed exactly for n <= 399: 1/4 at n = 1,
# at most MONOMIAL_C2 (attained at n = 34) for n >= 2.  For n >= 400,
# opbounds.query_majorant() is a polynomial in n, 2^n, C(2n, n) and log2 n that bounds
# query_bound(n), and opbounds.ratio_bound(query_majorant(), 400, 11) shows that each of
# its terms divided by n^11 4^n is nonincreasing on [400, oo), with total below
# MONOMIAL_C2.  11 is forced: stage 1 makes n Q^3 D_mid(n) Core calls of at least
# n E(n-1, n) = n^2 (n-2) C(2n-4, n-3) queries each, which grows like n^10.5 log(n) 4^n.
MONOMIAL_C = Fraction(1, 4)
MONOMIAL_C2 = Fraction(985174819882600647950452934699818905, 20716090166726403909801555666040520704)
MONOMIAL_K = 11


def query_bound(n: int) -> int:
    """Explicit bound on the total number of queries of ``Main`` on ``n`` agents, including
    recursion: ``n`` initial Evals + ``max_{m<=3} base(m)`` + ``sum_{k=4}^{n} Lambda_k``
    (Theorem 1.1's proof, with ``Lambda`` nondecreasing)."""
    if n < 1:
        raise ValueError("need at least one agent")
    if n <= 3:
        return n + BASE_CASE_QUERIES[n]
    total = n + BASE_CASE_QUERIES[3]
    for k in range(4, n + 1):
        total += lambda_n(k)
    return total


def monomial_query_bound(n: int) -> int:
    """``n**11 * 4**n / 4``, proved from the accounting inequalities (``PROOF.md`` §9)."""
    if n < 1:
        raise ValueError("need at least one agent")
    return n**MONOMIAL_K * 4**(n - 1)
