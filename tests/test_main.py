"""End-to-end tests of ``Main``: complete, exactly envy-free allocations within the bound."""

import os
import subprocess
import sys
from fractions import Fraction

import pytest

from cake_cutting import bounds
from cake_cutting import valuations as V
from cake_cutting.main import MainProtocol, run_protocol
from cake_cutting.oracle import RobertsonWebbOracle
from cake_cutting.pieces import OrderedPiece
from cake_cutting import subcore as subcore_module

from .families import DEEP_FAMILIES, FAMILIES, certify, make_valuations

F = Fraction
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: largest observed query count per n (updated by the tests, reported at the end of the run)
OBSERVED = {}


def check_accounting(stats, n):
    """The per-stage counts of the run stay within the paper's accounting (Lemma 4.7,
    Lemma 4.10, Section 4.6, Prop. 4.4), summed over the recursive calls actually made."""
    sizes = [m for m in stats["recursion"] if m >= 4]
    assert stats["recursion"][0] == n
    assert all(a > b for a, b in zip(stats["recursion"], stats["recursion"][1:])), "recursion must shrink"
    assert stats["core_calls_snapshots"] == sum(m * bounds.Q_of(m) ** 2 for m in sizes)
    assert stats["core_calls_stage1_shrink"] <= sum(m * bounds.Q_of(m) ** 3 * bounds.D_mid(m) for m in sizes)
    assert stats["core_calls_stage2_shrink"] <= sum(m * bounds.D_shortcut(m) for m in sizes)
    assert stats["core_calls_stage3_shrink"] <= sum(m * bounds.D_final(m) for m in sizes)
    assert stats["lemma46_queries"] <= sum(bounds.lemma46_budget(m) for m in sizes)
    assert stats["max_core_queries"] <= bounds.core_queries(n)
    assert stats["core_query_bound_ok"]
    assert all(phase == "rounding" for phase in stats["lemma46_phase"])


def run_instance(family, n, seed, cut_rule="left", log=False):
    vals = make_valuations(family, n, seed)
    oracle = RobertsonWebbOracle(vals, cut_rule=cut_rule, log=log)
    allocation, stats = run_protocol(oracle)
    certify(oracle, allocation, sorted(vals))
    check_accounting(stats, n)
    prev = OBSERVED.get(n, (0, 0))
    OBSERVED[n] = (max(prev[0], oracle.queries), max(prev[1], stats["max_core_queries"]))
    return oracle, allocation, stats


# ------------------------------------------------------------- n = 2, 3 ----
@pytest.mark.parametrize("cut_rule", ["left", "right"])
@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("n", [1, 2, 3])
def test_small_n(n, family, cut_rule):
    for seed in range(6):
        run_instance(family, n, seed, cut_rule, log=True)


# ----------------------------------------------------------------- n = 4 ----
@pytest.mark.parametrize("family", list(FAMILIES))
def test_n4_all_families(family):
    for seed in range(5):
        run_instance(family, 4, seed, "left" if seed % 2 == 0 else "right", log=(seed == 0))


# ----------------------------------------------------------------- n = 5 ----
# Instances whose residue survives pre-processing cost 5-50 s each (thousands of Core
# calls on rationals with thousands of bits); the default suite keeps the cheaper ones.
N5_DEFAULT = [
    ("random", 0, "left"), ("random", 1, "right"), ("many_segments", 0, "left"),
    ("identical", 0, "left"), ("identical", 1, "right"), ("pairs", 0, "left"), ("pairs", 2, "right"),
    ("types3", 0, "left"), ("types3", 1, "right"), ("perturbed", 0, "left"), ("perturbed", 1, "right"),
    ("zero_regions", 0, "left"), ("zero_regions", 1, "right"), ("blocky", 1, "right"), ("blocky", 0, "left"),
    ("uniform", 0, "left"), ("hotspot", 0, "left"), ("hotspot", 1, "right"),
    ("concentrated", 0, "left"), ("concentrated_pairs", 0, "left"), ("concentrated_pairs", 2, "right"),
    ("concentrated_perturbed", 0, "left"), ("nearly_identical_pairs", 0, "left"),
]


@pytest.mark.parametrize("family,seed,cut_rule", N5_DEFAULT)
def test_n5(family, seed, cut_rule):
    run_instance(family, 5, seed, cut_rule)


# ----------------------------------------------------------------- n = 6 ----
N6_DEFAULT = [
    ("random", 0, "left"), ("random", 1, "right"), ("random", 2, "left"),
    ("pairs", 0, "left"), ("pairs", 1, "right"), ("types3", 0, "left"), ("perturbed", 1, "right"),
    ("blocky", 0, "right"), ("identical", 0, "left"), ("hotspot", 0, "left"), ("many_segments", 0, "left"),
    ("concentrated", 0, "left"),
]


@pytest.mark.parametrize("family,seed,cut_rule", N6_DEFAULT)
def test_n6(family, seed, cut_rule):
    run_instance(family, 6, seed, cut_rule)


# ----------------------------------------------------------------- n = 7 ----
@pytest.mark.parametrize("seed", [0, 1])
def test_n7_random(seed):
    run_instance("random", 7, seed, "left" if seed == 0 else "right")


# ------------------------------------------------------------ slow runs ----
N5_SLOW = [
    ("pairs", 1, "right"), ("concentrated", 1, "right"), ("concentrated_pairs", 1, "right"),
    ("concentrated_perturbed", 1, "right"), ("many_segments", 1, "right"), ("zero_regions", 2, "left"),
    ("blocky", 2, "left"), ("concentrated", 2, "left"), ("concentrated_perturbed", 2, "left"),
]


@pytest.mark.slow
@pytest.mark.parametrize("family,seed,cut_rule", N5_SLOW)
def test_slow_n5(family, seed, cut_rule):
    run_instance(family, 5, seed, cut_rule)


N6_SLOW = [
    ("zero_regions", 0, "right"), ("zero_regions", 1, "left"), ("zero_regions", 2, "right"),
    ("blocky", 1, "left"), ("perturbed", 0, "left"), ("concentrated", 1, "right"),
    ("concentrated_pairs", 0, "left"), ("concentrated_pairs", 1, "right"),
    ("concentrated_perturbed", 0, "left"), ("concentrated_perturbed", 1, "right"),
    ("nearly_identical_pairs", 0, "left"), ("nearly_identical_pairs", 1, "right"),
]


@pytest.mark.slow
@pytest.mark.parametrize("family,seed,cut_rule", N6_SLOW)
def test_slow_n6(family, seed, cut_rule):
    run_instance(family, 6, seed, cut_rule)


@pytest.mark.slow
@pytest.mark.parametrize("family,seed,cut_rule", [("random", 2, "left"), ("random", 3, "right"),
                                                  ("identical", 0, "left"), ("hotspot", 0, "right")])
def test_slow_n7(family, seed, cut_rule):
    run_instance(family, 7, seed, cut_rule)


# ------------------------------------------------- coverage of the stages ----
def test_stage_coverage_on_concentrated_instances():
    """The concentrated/perturbed families are the ones that survive pre-processing with a
    residue of positive value; collectively they exercise every branch of Stages 2-3."""
    seen = {"stage3": 0, "shortcut": 0, "stage1_shrinks": 0, "rank_pieces": 0, "witness_pairs": 0,
            "recursion_into_two": 0, "additions": 0}
    for family in DEEP_FAMILIES:
        for seed in range(4):
            _, _, stats = run_instance(family, 4, seed, "left" if seed % 2 == 0 else "right")
            seen["stage3"] += len(stats["lemma46_phase"])
            seen["shortcut"] += stats["shortcuts"]
            seen["stage1_shrinks"] += stats["stage1_shrink_steps"]
            seen["rank_pieces"] += stats["rank_pieces_nonempty"]
            seen["witness_pairs"] += stats["witness_pairs"]
            seen["additions"] += stats["additions_nonempty"]
            seen["recursion_into_two"] += int(len(stats["recursion"]) >= 2 and stats["recursion"][1] == 2)
    assert seen["stage3"] > 0, "Stage 3 (Lemma 4.6, SubCore per class) never reached"
    assert seen["shortcut"] > 0, "the sink-SCC shortcut (Lemma 4.10) never taken"
    assert seen["stage1_shrinks"] > 0, "the intermediate-bonus loop (Lemma 4.7) never ran"
    assert seen["rank_pieces"] > 0, "no rank piece was ever cut (Case 2 never occurred)"
    assert seen["witness_pairs"] > 0, "no witnessed pair was ever matched (Case 1 never occurred)"
    assert seen["additions"] > 0, "no non-empty addition A_{s,k} was ever cut (no positive small bonus)"
    assert seen["recursion_into_two"] > 0


def test_determinism():
    a = run_instance("concentrated", 4, 0)
    b = run_instance("concentrated", 4, 0)
    assert a[0].queries == b[0].queries
    assert a[1].pieces() == b[1].pieces()


# ------------------------------------------- valuations only via the oracle ----
class _Guard:
    inside = False
    running = False


class _GuardedValuation:
    def __init__(self, inner):
        self._inner = inner

    def value(self, x, y):
        if not _Guard.inside:
            raise AssertionError("valuation accessed outside the oracle")
        return self._inner.value(x, y)

    def cut(self, x, a, rule="left"):
        if not _Guard.inside:
            raise AssertionError("valuation accessed outside the oracle")
        return self._inner.cut(x, a, rule)

    def __getattr__(self, name):
        raise AssertionError(f"direct access to valuation attribute {name!r}")


class _GuardedOracle(RobertsonWebbOracle):
    def eval(self, agent, x, y):
        _Guard.inside = True
        try:
            return super().eval(agent, x, y)
        finally:
            _Guard.inside = False

    def cut(self, agent, x, a):
        _Guard.inside = True
        try:
            return super().cut(agent, x, a)
        finally:
            _Guard.inside = False

    def true_value(self, agent, piece):
        if _Guard.running:
            raise AssertionError("the protocol called the test-only true_value")
        _Guard.inside = True
        try:
            return super().true_value(agent, piece)
        finally:
            _Guard.inside = False


@pytest.mark.parametrize("family,n,seed", [("random", 4, 0), ("concentrated", 4, 0),
                                           ("concentrated_perturbed", 4, 1), ("random", 3, 0)])
def test_protocol_touches_valuations_only_through_the_oracle(family, n, seed):
    vals = make_valuations(family, n, seed)
    oracle = _GuardedOracle({i: _GuardedValuation(v) for i, v in vals.items()})
    _Guard.running = True
    try:
        allocation, stats = run_protocol(oracle)
    finally:
        _Guard.running = False
    certify(oracle, allocation, sorted(vals))


# ------------------------------------------------- SubCore counterexample ----
def _pcv(breakpoints, densities):
    return V.PiecewiseConstantValuation([F(x) for x in breakpoints], [F(d) for d in densities])


#: Five agents for which the first Core call of ``Main`` (cutter 0, ``cut_rule="right"``)
#: reaches a SubCore step with no unallocated contested piece carrying a non-winner
#: trim (CORRECTNESS.md §7).  Agent 2 values both uncontested pieces of that step at 0,
#: trims two contested pieces to their empty end and wins both.
SUBCORE_COUNTEREXAMPLE = {
    0: (["0", "1/3", "1/2", "5/6", "1"], [1, 1, 0, 2]),
    1: (["0", "1/6", "1/2", "2/3", "1"], [2, 2, 0, 1]),
    2: (["0", "1/8", "13/48", "17/48", "7/16", "13/24", "9/16", "3/4", "5/6", "1"],
        [2, 0, 0, 0, 2, 0, 0, 3, 0]),
    3: (["0", "1/24", "1/6", "3/16", "5/12", "23/48", "25/48", "7/12", "11/16", "37/48", "47/48", "1"],
        [1, 0, 2, 0, 3, 1, 2, 1, 2, 3, 0]),
    4: (["0", "1/8", "1/4", "1/2", "3/4", "1"], [0, 1, 1, 1, 1]),
}


@pytest.mark.parametrize("rule", ["left", "right"])
def test_subcore_counterexample(rule):
    """Aziz–Mackenzie's SubCore fails on this input with the right cut rule
    (CORRECTNESS.md §7); the equilibrium fallback (§8) completes the run."""
    vals = {i: _pcv(b, d) for i, (b, d) in SUBCORE_COUNTEREXAMPLE.items()}
    oracle = RobertsonWebbOracle(vals, cut_rule=rule)
    before = subcore_module.FALLBACKS[0]
    allocation, _ = run_protocol(oracle)
    certify(oracle, allocation, sorted(vals))
    assert subcore_module.FALLBACKS[0] - before == (1 if rule == "right" else 0)


def test_subcore_counterexample_has_a_valid_allocation():
    """The SubCore input of the failing Core call (cutter 0's equal division of the cake,
    agents 1-4) admits a neat envy-free allocation of suffixes of distinct pieces."""
    vals = {i: _pcv(b, d) for i, (b, d) in SUBCORE_COUNTEREXAMPLE.items()}
    oracle = RobertsonWebbOracle(vals)
    cuts = ["0", "1/6", "1/3", "5/6", "11/12", "1"]
    pieces = [OrderedPiece.interval(F(a), F(b)) for a, b in zip(cuts, cuts[1:])]
    assert len({oracle.true_value(0, p) for p in pieces}) == 1          # equal for the cutter
    alloc = {1: pieces[0], 2: OrderedPiece.interval(F(71, 96), F(5, 6)), 3: pieces[3], 4: pieces[1]}
    for i in alloc:
        own = oracle.true_value(i, alloc[i])
        assert all(oracle.true_value(i, alloc[j]) <= own for j in alloc)
        assert oracle.true_value(i, pieces[4]) <= own                    # the unallocated piece


# ------------------------------------------------------------------ CLI ----
def test_cli_subprocess():
    env = dict(os.environ, PYTHONPATH=ROOT)
    out = subprocess.run([sys.executable, "-m", "cake_cutting", "--agents", "4", "--seed", "2",
                          "--family", "concentrated"], cwd=ROOT, env=env, capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert "complete: True, envy-free: True" in out.stdout
    assert "queries:" in out.stdout and "agent 3:" in out.stdout


def test_observed_query_counts_are_below_the_bound(capsys):
    for n, (q, c) in sorted(OBSERVED.items()):
        assert q <= bounds.query_bound(n)
        assert c <= bounds.core_queries(n)
    with capsys.disabled():
        print("\nmax observed queries per n in this run (total / largest single Core call): "
              + ", ".join(f"n={n}: {q} / {c} (bounds {bounds.query_bound(n):.2e} / {bounds.core_queries(n)})"
                          for n, (q, c) in sorted(OBSERVED.items())))
