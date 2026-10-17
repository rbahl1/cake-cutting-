"""Command-line demo: ``python -m cake_cutting --agents 4 --seed 1``.

Builds seeded random piecewise-constant valuations, runs ``Main`` against a
counting Robertson–Webb oracle and prints the allocation, every agent's value
of every piece (true values, computed outside the protocol), the query count
and the explicit bound of :mod:`cake_cutting.bounds`.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from fractions import Fraction

from . import bounds
from . import valuations as V
from .main import run_protocol
from .oracle import RobertsonWebbOracle
from .pieces import OrderedPiece

FAMILIES = ("random", "identical", "perturbed", "zero", "concentrated", "blocky", "uniform", "pairs")


def make_valuations(n: int, seed: int, family: str):
    rng = random.Random(seed)
    vals = {}
    for i in range(n):
        if family == "random":
            vals[i] = V.random_valuation(rng)
        elif family == "identical":
            vals[i] = vals[0] if i > 0 else V.random_valuation(rng)
        elif family == "perturbed":
            vals[i] = V.perturbed(vals[0], rng, Fraction(1, 10 ** 6)) if i > 0 else V.random_valuation(rng)
        elif family == "zero":
            vals[i] = V.random_valuation(rng, segments=5, grid=20, max_weight=3, zero_prob=0.5)
        elif family == "concentrated":
            vals[i] = V.concentrated(rng, grid=200, width=rng.randint(1, 3))
        elif family == "blocky":
            vals[i] = V.blocky(rng)
        elif family == "uniform":
            vals[i] = V.uniform()
        elif family == "pairs":
            vals[i] = vals[i - 1] if i % 2 == 1 else V.random_valuation(rng, segments=4, grid=12, max_weight=3)
        else:
            raise SystemExit(f"unknown family {family!r}; choose from {FAMILIES}")
    return vals


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cake_cutting", description=__doc__)
    parser.add_argument("--agents", type=int, default=4, help="number of agents (default 4)")
    parser.add_argument("--seed", type=int, default=0, help="random seed for the valuations")
    parser.add_argument("--family", choices=FAMILIES, default="random", help="valuation family")
    parser.add_argument("--cut-rule", choices=("left", "right"), default="left",
                        help="which admissible point a Cut query returns")
    parser.add_argument("--verbose", action="store_true", help="trace the stages of Main")
    args = parser.parse_args(argv)
    if args.agents < 1:
        parser.error("--agents must be positive")

    vals = make_valuations(args.agents, args.seed, args.family)
    oracle = RobertsonWebbOracle(vals, cut_rule=args.cut_rule)
    t0 = time.perf_counter()
    allocation, stats = run_protocol(oracle, trace=(print if args.verbose else None))
    elapsed = time.perf_counter() - t0

    agents = sorted(vals)
    print(f"agents: {len(agents)}, seed: {args.seed}, family: {args.family}, cut rule: {args.cut_rule}")
    print("allocation (pieces as unions of half-open intervals):")
    for a in agents:
        piece = allocation[a]
        print(f"  agent {a}: {len(piece.components)} component(s), measure {piece.normalized_length()} = "
              f"{float(piece.normalized_length()):.6f}")
        print(f"           {piece.pretty() if len(piece.components) <= 12 else '(too many components to print)'}")
    print("values V_i(X_j) (rows: agent i, columns: piece of agent j; diagonal = own piece):")
    header = "        " + "".join(f"{'X_' + str(j):>14}" for j in agents)
    print(header)
    for i in agents:
        own = oracle.true_value(i, allocation[i])
        row = []
        for j in agents:
            v = oracle.true_value(i, allocation[j])
            row.append(f"{float(v):>14.8f}")
        print(f"  V_{i}:  " + "".join(row) + f"   (own = {own})")
    complete = allocation.is_complete(OrderedPiece.cake())
    envy_free = all(oracle.true_value(i, allocation[i]) >= oracle.true_value(i, allocation[j])
                    for i in agents for j in agents)
    print(f"complete: {complete}, envy-free: {envy_free}")
    print(f"queries: {oracle.queries} (evals {oracle.evals}, cuts {oracle.cuts}); "
          f"bound T({len(agents)}) = {bounds.query_bound(len(agents)):.3e}")
    print(f"core calls: {stats['core_calls']} (snapshots {stats['core_calls_snapshots']}, "
          f"stage-1 shrinks {stats['core_calls_stage1_shrink']}, stage-2 {stats['core_calls_stage2_shrink']}, "
          f"stage-3 {stats['core_calls_stage3_shrink']}); Lemma 4.6 queries: {stats['lemma46_queries']} "
          f"{stats['lemma46_phase']}; recursion sizes: {stats['recursion']}; shortcuts: {stats['shortcuts']}")
    print(f"time: {elapsed:.2f}s")
    return 0 if (complete and envy_free) else 1


if __name__ == "__main__":
    sys.exit(main())
