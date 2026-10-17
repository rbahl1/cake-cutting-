"""Exact polynomials in nonnegative variables, used to evaluate the operation bound of
:mod:`cake_cutting.opbounds` symbolically (``PROOF.md`` §8).

A :class:`Poly` is a finite sum of rational multiples of monomials in the variables
:data:`VARS`; every variable stands for a nonnegative number.  The cost functions of
:mod:`cake_cutting.opbounds` accept a ``Poly`` wherever they accept an integer.  Their
arithmetic (``+``, ``-``, ``*``, ``**``) is exact; the only non-polynomial operations
are replaced by polynomial upper bounds, valid for all nonnegative values of the
variables:

* ``pmax(a, b)`` -- the coefficientwise maximum, which is at least ``max(a, b)``
  because all variables are nonnegative;
* ``dec(x)`` = ``max(0, x - 1)`` -- ``x`` itself;
* the special functions of :mod:`cake_cutting.opbounds` (``clog2``, ``digits``, ...)
  are given polynomial upper bounds there.

``positive()`` drops the monomials with negative coefficients; for nonnegative
values of the variables this can only increase the value.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, Tuple

#: m = agents of a Main call, E = 2^m, B = log2 m, G = C(2m, m), n = agents of the
#: table, N = 2^n, M = C(2n, n), L = log2 n, hr / ha = tree heights,
#: s / j / u = summation indices
VARS = ("m", "E", "B", "G", "n", "N", "M", "L", "hr", "ha", "s", "j", "u")
_IDX = {v: k for k, v in enumerate(VARS)}
_ZERO_MONO = (0,) * len(VARS)

Mono = Tuple[int, ...]


class Poly:
    __slots__ = ("terms",)

    def __init__(self, terms: Dict[Mono, Fraction] = None) -> None:
        self.terms: Dict[Mono, Fraction] = {}
        if terms:
            for mono, c in terms.items():
                if c != 0:
                    self.terms[mono] = Fraction(c)

    # -- construction ---------------------------------------------------------
    @staticmethod
    def var(name: str) -> "Poly":
        mono = [0] * len(VARS)
        mono[_IDX[name]] = 1
        return Poly({tuple(mono): Fraction(1)})

    @staticmethod
    def const(c) -> "Poly":
        return Poly({_ZERO_MONO: Fraction(c)})

    @staticmethod
    def lift(x) -> "Poly":
        return x if isinstance(x, Poly) else Poly.const(x)

    # -- arithmetic -------------------------------------------------------------
    def __add__(self, other) -> "Poly":
        other = Poly.lift(other)
        out = dict(self.terms)
        for mono, c in other.terms.items():
            out[mono] = out.get(mono, 0) + c
        return Poly(out)

    __radd__ = __add__

    def __neg__(self) -> "Poly":
        return Poly({mono: -c for mono, c in self.terms.items()})

    def __sub__(self, other) -> "Poly":
        return self + (-Poly.lift(other))

    def __rsub__(self, other) -> "Poly":
        return Poly.lift(other) - self

    def __mul__(self, other) -> "Poly":
        if not isinstance(other, Poly):
            other = Fraction(other)
            return Poly({mono: c * other for mono, c in self.terms.items()})
        out: Dict[Mono, Fraction] = {}
        for m1, c1 in self.terms.items():
            for m2, c2 in other.terms.items():
                mono = tuple(a + b for a, b in zip(m1, m2))
                out[mono] = out.get(mono, 0) + c1 * c2
        return Poly(out)

    __rmul__ = __mul__

    def __truediv__(self, other) -> "Poly":
        return self * (1 / Fraction(other))

    def __pow__(self, k: int) -> "Poly":
        out = Poly.const(1)
        for _ in range(k):
            out = out * self
        return out

    # -- bounds -------------------------------------------------------------------
    def positive(self) -> "Poly":
        return Poly({mono: c for mono, c in self.terms.items() if c > 0})

    def evaluate(self, **values) -> Fraction:
        total = Fraction(0)
        vals = [Fraction(values.get(v, 0)) for v in VARS]
        for mono, c in self.terms.items():
            t = c
            for e, x in zip(mono, vals):
                if e:
                    t *= x ** e
            total += t
        return total

    def degree(self, name: str) -> int:
        k = _IDX[name]
        return max((mono[k] for mono in self.terms), default=0)

    def by_power(self, name: str) -> Dict[int, "Poly"]:
        """``{e: coefficient polynomial of name^e}``."""
        k = _IDX[name]
        out: Dict[int, Dict[Mono, Fraction]] = {}
        for mono, c in self.terms.items():
            rest = list(mono)
            e = rest[k]
            rest[k] = 0
            out.setdefault(e, {})[tuple(rest)] = c
        return {e: Poly(t) for e, t in out.items()}

    def is_nonnegative(self) -> bool:
        return all(c >= 0 for c in self.terms.values())

    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        parts = []
        for mono, c in sorted(self.terms.items()):
            names = "*".join(f"{v}^{e}" if e > 1 else v for v, e in zip(VARS, mono) if e)
            parts.append(f"{c}" + (f"*{names}" if names else ""))
        return " + ".join(parts)


def is_poly(*xs) -> bool:
    return any(isinstance(x, Poly) for x in xs)


def pmax(*xs):
    """``max`` for numbers; for polynomials the coefficientwise maximum (an upper bound
    for nonnegative values of the variables)."""
    if not is_poly(*xs):
        return max(xs)
    out: Dict[Mono, Fraction] = {}
    for x in xs:
        for mono, c in Poly.lift(x).terms.items():
            out[mono] = max(out.get(mono, Fraction(0)), c)
    return Poly(out)


def dec(x):
    """``max(0, x - 1)``; for a polynomial, ``x`` (an upper bound)."""
    if isinstance(x, Poly):
        return x
    return max(0, x - 1)


def sum_below(f, var: str, upper):
    """``sum_{v=1}^{upper-1} f(v)``.  For an integer ``upper`` the exact sum; for a
    polynomial ``upper`` the bound obtained from ``f(v)`` (evaluated at the variable
    ``var``): drop the negative monomials, then replace ``v^k`` by
    ``upper^(k+1)/(k+1)``, since ``sum_{v=1}^{w-1} v^k <= w^(k+1)/(k+1)`` for every
    integer ``w >= 1`` and ``k >= 0``."""
    if not isinstance(upper, Poly):
        return sum((f(v) for v in range(1, upper)), 0)
    body = Poly.lift(f(Poly.var(var))).positive()
    out = Poly()
    for k, coef in body.by_power(var).items():
        out = out + coef * (upper ** (k + 1)) / (k + 1)
    return out
