"""The instruction profiles of ``cake_cutting.opcount`` against measured executions.

Lemma 1.1 of PROOF.md: one call of a function executes at most

    outside + sum over loops [cycle * (iterations + entries) + entry * entries]

of its own instructions.  For functions whose loop counts are known exactly, the
number of executed instructions (counted with ``sys.monitoring``) must not exceed
this expression.  The functions cover every construct whose bytecode places
instructions away from their source position: ``for`` (``FOR_ITER`` carries the
position of the iterable), nested loops, ``while``, comprehensions, generators,
``yield from`` (a hidden loop over the delegated values), ``try``/``except`` inside a
loop (handlers are placed after the loop, the entry point has no position),
``break``/``continue``, ``for``-``else``, and loops so long that the backward jump
targets an ``EXTENDED_ARG`` prefix.
"""

import sys

import pytest

from cake_cutting.opcount import profile

TOOL = 4


def executed(fn, *args, consume=False):
    code = fn.__code__
    n = [0]

    def cb(code_, offset):
        n[0] += 1

    sys.monitoring.use_tool_id(TOOL, "opcount-test")
    try:
        sys.monitoring.register_callback(TOOL, sys.monitoring.events.INSTRUCTION, cb)
        sys.monitoring.set_local_events(TOOL, code, sys.monitoring.events.INSTRUCTION)
        r = fn(*args)
        if consume:
            for _ in r:
                pass
        sys.monitoring.set_local_events(TOOL, code, 0)
    finally:
        sys.monitoring.register_callback(TOOL, sys.monitoring.events.INSTRUCTION, None)
        sys.monitoring.free_tool_id(TOOL)
    return n[0]


def bound(fn, counts):
    """counts[line] = (iterations, entries) for every loop of fn."""
    p = profile(fn)
    total = p["outside"]
    for line, kind, cycle, entry in p["loops"]:
        it, en = counts[line]
        total += cycle * (it + en) + entry * en
    return total


def _line(fn, marker):
    import inspect
    src, first = inspect.getsourcelines(fn)
    for i, s in enumerate(src):
        if marker in s:
            return first + i
    raise KeyError(marker)


# ------------------------------------------------------------------ functions

def f_for(L):
    s = 0
    for v in L:  # L1
        s += v
    return s


def f_nested(L):
    s = 0
    for v in L:  # L1
        for w in L:  # L2
            s += v * w
    return s


def f_while(N):
    i = 0
    s = 0
    while i < N and s >= 0:  # L1
        s += i
        i += 1
    return s


def f_comps(L):
    a = [v * 2 for v in L if v % 2 == 0]  # L1
    b = {v for v in L}  # L2
    c = {v: v + 1 for v in L}  # L3
    d = [(v, w) for v in L for w in L]  # L4
    return len(a) + len(b) + len(c) + len(d)


def f_gen(L):
    for v in L:  # L1
        yield v * v


def f_inner_gen(L):
    for v in L:  # L1
        yield v


def f_yield_from(L):
    yield from f_inner_gen(L)  # L1
    yield from f_inner_gen(L)  # L2


def f_try(L):
    s = 0
    for v in L:  # L1
        try:
            s += 10 // v
        except ZeroDivisionError:
            s -= 1
    return s


def f_break_continue(L):
    s = 0
    for v in L:  # L1
        if v % 3 == 0:
            continue
        if v > 50:
            break
        s += v
    else:
        s += 1000
    return s


def f_while_try(N):
    i = 0
    s = 0
    while True:  # L1
        try:
            if i >= N:
                raise StopIteration
            s += i
        except StopIteration:
            break
        i += 1
    return s


def f_long_nested(L):
    """An inner body longer than 255 code units: the backward jumps target an
    EXTENDED_ARG prefix of FOR_ITER."""
    s = 0
    for v in L:  # L1
        for w in L:  # L2
            s += v * w + 0
            s += v * w + 1
            s += v * w + 2
            s += v * w + 3
            s += v * w + 4
            s += v * w + 5
            s += v * w + 6
            s += v * w + 7
            s += v * w + 8
            s += v * w + 9
            s += v * w + 10
            s += v * w + 11
            s += v * w + 12
            s += v * w + 13
            s += v * w + 14
            s += v * w + 15
            s += v * w + 16
            s += v * w + 17
            s += v * w + 18
            s += v * w + 19
            s += v * w + 20
            s += v * w + 21
            s += v * w + 22
            s += v * w + 23
            s += v * w + 24
            s += v * w + 25
            s += v * w + 26
            s += v * w + 27
            s += v * w + 28
            s += v * w + 29
            s += v * w + 30
            s += v * w + 31
            s += v * w + 32
            s += v * w + 33
            s += v * w + 34
            s += v * w + 35
            s += v * w + 36
            s += v * w + 37
            s += v * w + 38
            s += v * w + 39
            s += v * w + 40
            s += v * w + 41
            s += v * w + 42
            s += v * w + 43
            s += v * w + 44
            s += v * w + 45
            s += v * w + 46
            s += v * w + 47
            s += v * w + 48
            s += v * w + 49
            s += v * w + 50
            s += v * w + 51
            s += v * w + 52
            s += v * w + 53
            s += v * w + 54
            s += v * w + 55
            s += v * w + 56
            s += v * w + 57
            s += v * w + 58
            s += v * w + 59
            s += v * w + 60
            s += v * w + 61
            s += v * w + 62
            s += v * w + 63
            s += v * w + 64
            s += v * w + 65
            s += v * w + 66
            s += v * w + 67
            s += v * w + 68
            s += v * w + 69
    return s


CASES = [
    (f_for, lambda N: ((list(range(N)),), {"L1": (N, 1)}), False),
    (f_nested, lambda N: ((list(range(N)),), {"L1": (N, 1), "L2": (N * N, N)}), False),
    (f_while, lambda N: ((N,), {"L1": (N, 1)}), False),
    (f_comps, lambda N: ((list(range(N)),), {"L1": (N, 1), "L2": (N, 1), "L3": (N, 1),
                                               "L4": (N + N * N, 1 + N)}), False),
    (f_gen, lambda N: ((list(range(N)),), {"L1": (N, 1)}), True),
    (f_yield_from, lambda N: ((list(range(N)),), {"L1": (N + 1, 1), "L2": (N + 1, 1)}), True),
    (f_try, lambda N: (([0] * N,), {"L1": (N, 1)}), False),
    (f_try, lambda N: ((list(range(1, N + 1)),), {"L1": (N, 1)}), False),
    (f_break_continue, lambda N: ((list(range(N)),), {"L1": (N, 1)}), False),
    (f_while_try, lambda N: ((N,), {"L1": (N + 1, 1)}), False),
    (f_long_nested, lambda N: ((list(range(N)),), {"L1": (N, 1), "L2": (N * N, N)}), False),
]


def _resolve(fn, counts):
    out = {}
    for marker, v in counts.items():
        out[_line(fn, "# " + marker)] = v
    return out


@pytest.mark.parametrize("case", range(len(CASES)))
def test_profile_bounds_measured_executions(case):
    fn, make, consume = CASES[case]
    lines = {lp[0] for lp in profile(fn)["loops"]}
    for N in (0, 1, 2, 7, 30):
        args, counts = make(N)
        counts = _resolve(fn, counts)
        assert set(counts) == lines, (fn.__name__, profile(fn))
        ex = executed(fn, *args, consume=consume)
        assert ex <= bound(fn, counts), (fn.__name__, N, ex, bound(fn, counts), profile(fn))
