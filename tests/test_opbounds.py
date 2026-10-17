"""The operation bound of :mod:`cake_cutting.opbounds` (proof in ``PROOF.md``).

* the frozen instruction profiles match the code (a change of the code must be
  followed by regenerating them with ``python -m cake_cutting.opcount`` and
  re-checking the proof);
* measured instruction counts of real runs stay below the per-function bounds
  (a necessary condition, see ``tests/opmeter.py``);
* the polynomial arithmetic used for large ``n`` is sound;
* the constant ``k``.
"""

import random
from fractions import Fraction
from math import comb

import pytest

from cake_cutting import bounds, opbounds as O, opcount
from cake_cutting._opprofiles import PROFILES
from cake_cutting.bookkeeping import ValueTable
from cake_cutting.main import run_protocol
from cake_cutting.oracle import RobertsonWebbOracle
from cake_cutting.pieces import OrderedPiece
from cake_cutting.polybound import Poly, pmax, sum_below
from cake_cutting.subcore import core

from .families import FAMILIES, make_valuations
from .opbound_targets import targets
from .opmeter import Meter

F = Fraction


def test_profiles_match_the_code():
    assert opcount.generate() == PROFILES


def test_profile_of_a_loop():
    prof = PROFILES["seq.nth"]
    assert prof["outside"] >= 1 and len(prof["loops"]) == 1
    _, kind, cycle, entry = prof["loops"][0]
    assert kind == "While" and cycle > 0


# ---------------------------------------------------------------- the meter ----

def _check(meter):
    bad = meter.violations()
    assert not bad, bad[:5]


@pytest.mark.parametrize("seed", range(4))
def test_core_sequences_respect_the_bounds(seed):
    fams = sorted(FAMILIES)
    for n in (4, 5):
        vals = make_valuations(fams[(seed * 3 + n) % len(fams)], n, seed)
        oracle = RobertsonWebbOracle(vals, cut_rule="left" if seed % 2 else "right")
        agents = sorted(vals)
        table = ValueTable(oracle, agents)
        table.register_cake()
        R = OrderedPiece.cake()
        with Meter(targets()) as meter:
            for t in range(10):
                _, R, _ = core(table, agents, R, agents[t % n])
                if R.is_empty:
                    break
        _check(meter)
        assert meter.results["core"] and meter.results["subcore"]


@pytest.mark.parametrize("family", ["random", "concentrated", "pairs", "zero_regions"])
def test_main_runs_respect_the_bounds(family):
    for n in (2, 3, 4):
        vals = make_valuations(family, n, 0)
        oracle = RobertsonWebbOracle(vals)
        with Meter(targets()) as meter:
            run_protocol(oracle)
        _check(meter)
        (measured, bound, _), = meter.results["run_protocol"]
        assert measured <= bound <= O.OP_K * n ** O.OP_D * 4 ** n


# ------------------------------------------------------- polynomial bounds ----

def test_poly_arithmetic_and_bounds():
    m, n = Poly.var("m"), Poly.var("n")
    p = (m - 1) * (m + 2) + 3 * n
    assert p.evaluate(m=5, n=2) == 4 * 7 + 6
    q = pmax(p, m * m * 2)
    for mv in range(0, 6):
        for nv in range(0, 4):
            assert q.evaluate(m=mv, n=nv) >= max(p.evaluate(m=mv, n=nv), 2 * mv * mv)
    # sum_below with a polynomial upper limit bounds the exact sum
    f = lambda s: (s - 1) * (s + 3) * 2 + 5 * s * s * s      # noqa: E731
    w = Poly.var("m") + 1
    bound = sum_below(f, "s", w)
    for mv in range(0, 12):
        exact = sum(f(s) for s in range(1, mv + 1))
        assert bound.evaluate(m=mv) >= exact


def test_symbolic_main_cost_dominates_the_exact_one():
    lam = O.MainCost(Poly.var("m"), Poly.var("n"), Poly.var("hr"), Poly.var("ha")).total().positive()
    for mv in (4, 5, 6, 8, 10):
        for nv in (mv, mv + 3):
            hr, ha = O.heights(nv)
            exact = O.MainCost(mv, nv, hr, ha).total()
            sym = lam.evaluate(m=mv, E=2 ** mv, G=comb(2 * mv, mv), B=O.log2_up(mv), n=nv,
                               L=O.log2_up(nv), hr=hr, ha=ha)
            assert sym >= exact


def test_symbolic_subcore_bound_dominates_the_exact_recursion():
    ctx = O.Ctx(5, 5, 20)
    Z = ctx.zinf()
    for w in range(1, 12):
        assert ctx.T(w) <= Fraction(2 ** w, 2) * Z


def test_tail_heights_dominate():
    HR, HA = O.tail_heights()
    for nv in (16, 20, 30, 40):
        hr, ha = O.heights(nv)
        L = O.log2_up(nv)
        assert HR.evaluate(L=L) >= hr and HA.evaluate(L=L) >= ha


def test_tail_majorant_dominates_the_bound():
    T = O.tail_majorant()
    for nv in (16, 18, 22):
        assert T.evaluate(n=nv, N=2 ** nv, M=comb(2 * nv, nv), L=O.log2_up(nv)) >= O.operation_bound(nv)
    Tq = O.query_majorant()
    for nv in (4, 10, 20, 30):
        assert (Tq.evaluate(n=nv, N=2 ** nv, M=comb(2 * nv, nv), L=O.log2_up(max(nv, 2)))
                >= bounds.query_bound(nv))


# ---------------------------------------------------------------- constants ----

def test_operation_constant_small_n_and_tail():
    assert F(O.operation_bound(1), 4) == O.OP_K
    assert F(O.operation_bound(4), 4 ** O.OP_D * 4 ** 4) == O.OP_K2
    for nv in range(1, 13):
        assert O.operation_bound(nv) <= O.OP_K * nv ** O.OP_D * 4 ** nv
        assert nv == 1 or O.operation_bound(nv) <= O.OP_K2 * nv ** O.OP_D * 4 ** nv
    r, ok = O.tail_ratio(O.OP_N_EXACT + 1)
    assert ok and r <= O.OP_K2


def test_exponent_eleven_is_needed():
    """``U(n) / (n^10 4^n)`` keeps growing (the fallback's share is about n^10.5 log n 4^n)."""
    ratios = [F(O.operation_bound(nv), nv ** 10 * 4 ** nv) for nv in (10, 14, 18, 22)]
    assert all(a < b for a, b in zip(ratios, ratios[1:]))
    q = [F(bounds.query_bound(nv), nv ** 10 * 4 ** nv) for nv in (10, 20, 40, 80, 160)]
    assert all(a < b for a, b in zip(q, q[1:]))


@pytest.mark.slow
def test_operation_constant_all_n():
    assert O.check_operation_constant() == O.OP_K2


def test_query_constant():
    worst = O.check_query_constant()
    assert worst == bounds.MONOMIAL_C2
    assert F(bounds.query_bound(34), 34 ** 11 * 4 ** 34) == bounds.MONOMIAL_C2
    assert F(bounds.query_bound(1), 4) == bounds.MONOMIAL_C


# ------------------------------------------------- the equilibrium fallback ----

def test_equilibrium_counts_are_exact():
    """Lemma 5.3: the recursion of ``allocate`` reaches exactly the counted states and calls."""
    from cake_cutting import equilibrium
    from .test_core_subcore import _fallback_instance
    seen = {}
    orig_high, orig_low = equilibrium._Market.high, equilibrium._Market.low

    def high(self, X, Y):
        seen["HC"] += 1
        if (X, Y) not in self.high_memo:
            seen["HS"] += 1
            seen["HXY"] += bin(X).count("1") * bin(Y).count("1")
            seen["VIEWS"] += bin(Y).count("1")
        return orig_high(self, X, Y)

    def low(self, X, Y):
        seen["LC"] += 1
        if (X, Y) not in self.low_memo:
            seen["LS"] += 1
            seen["LX"] += bin(X).count("1")
            seen["LXY"] += bin(X).count("1") * bin(Y).count("1")
        return orig_low(self, X, Y)

    rng = random.Random(5)
    try:
        equilibrium._Market.high, equilibrium._Market.low = high, low
        for _ in range(60):
            oracle, table, pieces, agents, q, _direct = _fallback_instance(rng, zero=False)
            seen.clear()
            seen.update(HC=0, HS=0, HXY=0, VIEWS=0, LC=0, LS=0, LX=0, LXY=0)
            _, calls = equilibrium.allocate(table, pieces, agents, q)
            c = O.eq_counts(len(agents), sum(1 for p in pieces if len(p) > 0))
            assert seen == {key: c[key] for key in seen}
            assert calls <= c["HXY"]
    finally:
        equilibrium._Market.high, equilibrium._Market.low = orig_high, orig_low


def test_symbolic_fallback_counts_dominate():
    for mv in range(4, 25):
        G = comb(2 * mv, mv)
        exact = O.eq_counts(mv - 1, mv)
        sym = O.eq_counts_core_sym(Poly.var("m"))
        for key, val in exact.items():
            assert Poly.lift(sym[key]).evaluate(m=mv, G=G) >= val, (mv, key)
    for e in range(1, 25):
        Ge = comb(2 * e, e)
        exact = O.eq_counts(e, e)
        sym = O.eq_counts_class_sym(Poly.var("m"))
        for key, val in exact.items():
            assert Poly.lift(sym[key]).evaluate(m=e, G=Ge) >= val, (e, key)


def test_central_binomial_bounds():
    """``C(2n, n) <= 4^n / sqrt(3n + 1)`` and the ratios used for the summation lemma."""
    for nv in range(1, 300):
        g = comb(2 * nv, nv)
        assert g * g * (3 * nv + 1) <= 16 ** nv
        if nv >= 4:
            assert 2 * g >= 7 * comb(2 * nv - 2, nv - 1)


def test_fallback_runs_respect_the_bounds():
    """The counterexample of CORRECTNESS.md §7 with the right cut rule: SubCore falls back to
    the equilibrium computation, and its measured instructions stay below the bound."""
    from cake_cutting import subcore as subcore_module
    from .test_main import SUBCORE_COUNTEREXAMPLE, _pcv
    vals = {i: _pcv(b, d) for i, (b, d) in SUBCORE_COUNTEREXAMPLE.items()}
    oracle = RobertsonWebbOracle(vals, cut_rule="right")
    before = subcore_module.FALLBACKS[0]
    with Meter(targets()) as meter:
        run_protocol(oracle)
    _check(meter)
    assert subcore_module.FALLBACKS[0] > before
    assert meter.results["allocate"]
