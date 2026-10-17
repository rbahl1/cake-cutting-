"""The explicit query accounting of :mod:`cake_cutting.bounds`."""

from fractions import Fraction

from cake_cutting import bounds


def test_subcore_recurrence():
    # S_m = S_{m-1} + m(m-1) + sum_{j=1}^{m-2} S_j  (Appendix D)
    for m in range(2, 12):
        rec = bounds.S(m - 1) + m * (m - 1) + sum(bounds.S(j) for j in range(1, m - 1))
        assert bounds.S(m) == rec == 2 ** (m + 1) - 2 * m - 2
    assert bounds.S(0) == bounds.S(1) == 0 and bounds.S(2) == 2 and bounds.S(3) == 8


def test_ceil_log_is_exact():
    assert bounds.ceil_log(4, Fraction(16)) == 4          # (4/2)^4 = 16
    assert bounds.ceil_log(4, Fraction(17)) == 5
    assert bounds.ceil_log(5, Fraction(1000)) == 14       # (5/3)^13 < 1000 <= (5/3)^14
    assert bounds.ceil_log(6, Fraction(1)) == 0
    for n in range(4, 9):
        for beta in (Fraction(2), Fraction(100), Fraction(10 ** 6)):
            d = bounds.ceil_log(n, beta)
            r = Fraction(n, n - 2)
            assert r ** d >= beta and (d == 0 or r ** (d - 1) < beta)


def test_stage_parameters_match_the_paper():
    for n in range(4, 10):
        Q = n * (n - 1)
        assert bounds.D_mid(n) == bounds.ceil_log(n, Fraction(4 * n * n * (Q + 1)))
        assert bounds.D_shortcut(n) == bounds.ceil_log(n, Fraction(Q * Q))
        assert bounds.D_final(n) == bounds.ceil_log(n, Fraction(4 * n * (Q + 1) * Q * Q))
        assert bounds.stage3_delta(n) == Fraction(1, n * Q * Q) / (16 * n * (Q + 1))
        assert bounds.core_queries(n) == n * (n - 1 + bounds.S(n - 1) + bounds.E(n - 1, n))


def test_bound_is_single_exponential_and_monotone():
    prev = None
    for n in range(4, 40):
        lam = bounds.lambda_n(n)
        if prev is not None:
            assert lam >= prev
        prev = lam
        T = bounds.query_bound(n)
        assert T >= bounds.query_bound(n - 1) + (lam if n > 4 else 0)
        assert T <= bounds.monomial_query_bound(n)
    assert bounds.query_bound(1) == 1 and bounds.query_bound(2) == 4 and bounds.query_bound(3) == 18
