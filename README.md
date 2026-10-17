# Single-exponential envy-free cake cutting (Ye–Bai 2026) in exact arithmetic

A faithful Python implementation of the protocol `Main(N, C)` of

> Qilin Ye and Yannan Bai, *Cutting Down the Tower: Single-Exponential Envy-Free
> Cake Cutting*, arXiv:2609.05191 (September 2026),

including the Core and (modified) SubCore subroutines of Aziz and Mackenzie
(2016) on which it relies, and an exact fallback for SubCore where the published
SubCore fails.  The protocol interacts with the agents only through Robertson–Webb
queries and returns a **complete envy-free allocation** of the cake `[0, 1]` among
`n` agents:

* **always**, when every agent values every nonempty interval positively
  (*positive density*; proved from the code in `CORRECTNESS.md`, Theorem D);
* with arbitrary nonnegative, additive, nonatomic valuations it never returns a
  wrong allocation, but it can raise `ProtocolError` on inputs with zero-value
  stretches (`CORRECTNESS.md` §9; this case is open, also in the literature).

It uses at most `n^{11} 4^n / 4` queries and `392 · n^{11} · 4^n` operations
(`PROOF.md`).

Aziz and Mackenzie (2016) gave the first bounded protocol with a bound of
`n^{n^{n^{n^{n^n}}}}` (a tower of six `n`s); Sokolov (2023) sharpened the analysis
of the same protocol to `n^{8 n^2 (1 + o(1))}`; Ye and Bai replace the tower of
partial allocations by polynomially many and claim `n^{O(1)} 2^n`.  All three rely
on Aziz–Mackenzie's SubCore, whose correctness proof fails on degenerate inputs
(Bechler–Bitikofer 2017; `CORRECTNESS.md` §7).  The fallback used here is proved
for positive densities and costs up to about `4^n` cuts per SubCore call, so the
proved bound of this implementation is `n^{11} 4^n` rather than `n^{O(1)} 2^n`.
The lower bound of Procaccia (2009) is `Ω(n^2)`.

Everything here is exact: cut points and values are `fractions.Fraction`s, the
protocol never reads a valuation (it only sees the answers of a counting
`Eval`/`Cut` oracle), and the tests certify completeness and envy-freeness
*exactly* against the true valuations.

## The theorem that is implemented

**Theorem 1.1 (Ye–Bai).** For every `n ≥ 1` and every collection of
nonnegative, additive, nonatomic valuations there is a deterministic bounded
Robertson–Webb protocol that returns a complete envy-free allocation using at
most `n^{O(1)} 2^n` queries.

What is proved for this code (`CORRECTNESS.md`, `PROOF.md`): with positive
densities every run returns a complete envy-free allocation with at most
`n^{11} 4^n / 4` queries and `392 n^{11} 4^n` operations; with general valuations
every run that returns is correct, within the same bounds.

The tests check, for every instance they run:

* the returned pieces are pairwise disjoint and their union is `[0, 1]` up to
  endpoints (completeness, exact);
* `V_i(X_i) ≥ V_i(X_j)` for all `i, j` with the *true* valuations (exact);
* every query was well formed (the oracle rejects ill-formed ones);
* the total query count is at most the explicit bound `T(n)` of
  `cake_cutting/bounds.py`, obtained by instantiating the paper's own
  accounting (see *Query complexity* below), and every per-stage count stays
  within the paper's per-stage bounds (`nQ^2` snapshot Core calls, at most
  `nQ^3` elimination sequences of `D_mid` calls each, …);
* traced runs stay below the per-function operation bounds of
  `cake_cutting/opbounds.py` (see *Running time* below).

## Walkthrough of `Main` (Sections 3–4 of the paper)

`Main(N, C)` with `n = |N|`.  For `n ≤ 3` the base protocols are used
(`cake_cutting/base_cases.py`: the sole agent takes everything; cut-and-choose;
Selfridge–Conway on ordered pieces).  For `n ≥ 4` set `Q = n(n − 1)`.

**Stage 1 – pre-processing** (`MainProtocol.preprocess`, Algorithm 2, Section 4.3).
Every agent in turn makes `Q^2` consecutive `Core` calls as the cutter on the
current residue; the `nQ^2` envy-free allocations are the *snapshots*
`c_{s,k}`.  Agent `i`'s *bonus* over label `k` in snapshot `s` is
`d_{s,i,k} = V_i(c_{s,i}) − V_i(c_{s,k}) ≥ 0`.  With
`Γ_i = V_i(U)/Q^2`, `η_i = Γ_i/(4n(Q+1))`, `ε_i = η_i/n` a bonus is *small* if
`≤ ε_i`, *large* if `≥ Γ_i`, *intermediate* otherwise.  While some agent has an
intermediate bonus she runs a prescribed sequence (`Shrink`) of
`D_mid = ⌈log_{n/(n−2)} 4n^2(Q+1)⌉` Core calls, which makes that bonus
permanently large (Lemma 4.7).  The final residue is `R*`.

**Stage 2 – pruning** (`MainProtocol.main`, `prune`, `sink_component`,
Section 4.4).  If nobody values `R*`, finish with `H ⊕ Id(S)` and give `R*` to
one agent.  Otherwise take the smallest agent `c` with `V_c(R*) > 0` and her
`Q^2` snapshots; agent `i` *marks* label `k` in snapshot `s` if `d_{s,i,k} > ε_i`
(then the bonus is large by Lemma 4.7).  Snapshot `s` *witnesses* `(i, h)` if
`i` marks a label `h` does not.  While some pair has between 1 and `Q − 1`
witnesses, all of them are discarded (Lemma 4.9: at least `Q` snapshots remain,
every witness set is empty or has size `≥ Q`, and at least two profile classes
remain because the cutter marks a label in each of her snapshots, Lemma 4.8).
For each surviving snapshot the graph `G_s` has the edge `i → k` iff `i` does
not mark `k`.  If some `G_s` is not strongly connected, a sink component `S`
dominates `N \ S` after prescribed sequences of `⌈log_{n/(n−2)} Q^2⌉` calls for
its members (Lemma 4.10) and the protocol recurses on `N \ S`.

**Stage 3 – the partial allocation** (`MainProtocol.stage3`, Algorithm 3,
Sections 4.5–4.6).  Agents with equal mark profiles form *profile classes*.
For every ordered pair `(i, h)` of agents with `F_i ⋠ F_h` an unused witness
snapshot is reserved and a perfect matching `μ_s` of `G_s` through the edge
`h → k` (`k` marked by `i`, not by `h`) is built from a shortest path
(Lemmas 4.11–4.12); `J` is padded with identity matchings to exactly `Q`
snapshots.  Ranks `ℓ_E` are longest `≺`-chains of classes (Lemma 4.13).
Lemma 4.6 (`cake_cutting/simultaneous.py`) cuts from `R*`, simultaneously, an
*addition* `A_{s,k}` of relative size `r_{s,k} + δ` for every `s ∈ J, k ∈ N`
and a *rank piece* `P_h` of relative size `L(ℓ_h − ½) + δ` for every agent of
positive rank (`L = 1/(nQ^2)`, `δ = L/(16n(Q+1))`, Lemma 4.14).  Each agent `h`
gets `W_h = [P_h, A_h, D_h]` with `D_h = ∪_s c_{s,μ_s(h)}` and `A_h` the
matching additions; within each class a direct SubCore call with required
minima `q_i = B_i + max{0, L(ℓ_E − ½)} V_i(R*)` and one empty piece
redistributes the `W_h` (Proposition 4.5, Lemma 4.17; the snapshot pieces are
placed last, so no trim enters them).  The result `X = H ⊕ Id(S \ J) ⊕ Z` is
envy-free with `Δ_ij(X) ≥ η_i` across classes (Lemmas 4.18–4.19).  A proper
class `E*` is chosen; every outside agent valuing the residue positively runs a
prescribed sequence of `⌈log_{n/(n−2)} 4n(Q+1)Q^2⌉` calls, after which she
dominates `E*` (Proposition 4.20), and the protocol returns
`X ⊕ Main(E*, R_2)`.

Several guarantees proved in the paper are re-checked at run time from stored
values (no extra queries): envy-freeness of every snapshot and of `X`,
Lemma 4.14 (i)–(iii), Lemma 4.17 (ii) and (iv), the existence of unused
witnesses (Lemma 4.12), `|P| ≥ Q`, at least two classes, and the domination
conditions before every recursion.  A violation raises `ProtocolError` instead
of returning a wrong allocation (`CORRECTNESS.md`, Theorem A).

## Paper-to-code map

| Paper | Code |
|---|---|
| Section 2, Appendix C.1: Robertson–Webb queries, ordered pieces, Prop. C.1 bookkeeping | `cake_cutting/oracle.py` (`RobertsonWebbOracle.eval/cut`, query counting), `cake_cutting/pieces.py` (`Interval`, `OrderedPiece`, `[P, A, D]` = `OrderedPiece.concat`), `cake_cutting/seq.py` (persistent AVL sequences storing the components with their values and subtree sums), `cake_cutting/bookkeeping.py` (`ValueTable`: values of all components, `split_by_cut` = 1 Cut + `n−1` Evals, `split_at` = `n` Evals, `prefix_of_value` / `suffix_of_value` = the C.2 "first boundary" rule, suffix views) |
| Section 4.1, Appendix D: Core, modified SubCore (AM line 12 skipped), Props. 4.3–4.5 | `cake_cutting/subcore.py` (`core`, `equal_division`, `SubCoreCall`, `subcore`, `unallocated_part`) |
| SubCore fallback (not in the paper): competitive equilibrium of the trimming market, Alaei–Jain–Malekian (arXiv:1006.4696) | `cake_cutting/equilibrium.py` (`allocate`, `_Market.low/high`, `_cover_matching`); proof in `CORRECTNESS.md` §8 |
| Section 4.2, Algorithm 4, Lemma 4.6 | `cake_cutting/simultaneous.py` (`SimultaneousCutter`: the rounding procedure; the literal procedure of the proof for tests only) |
| Section 4.3 base cases (`n ≤ 3`) | `cake_cutting/base_cases.py` (`cut_and_choose`, `selfridge_conway`, `base_case`) |
| Algorithm 2 `Preprocess`, Lemma 4.7, `Shrink` | `MainProtocol.preprocess`, `MainProtocol.shrink` in `cake_cutting/main.py` |
| Section 4.4, Lemmas 4.8–4.10 (marks, witnesses, pruning, `G_s`, sink component) | `MainProtocol.main`, `witnesses`, `prune`, `sink_component` |
| Lemmas 4.11–4.13 (matchings, `J`, ranks) | `MainProtocol.stage3` (first half), `matching_through`, `ranks` |
| Algorithm 3 `Allocate`, Lemmas 4.14–4.18 | `MainProtocol.stage3` (second half), `check_lemma_4_14` |
| Section 4.6, Prop. 4.20 (domination, recursion) | end of `MainProtocol.stage3`, `assert_domination`; proof for the code in `CORRECTNESS.md` |
| Prop. 4.21, Theorem 1.1 (query bound) | `cake_cutting/bounds.py` (`core_queries`, `S`, `E`, `subcore_cuts`, `D_mid`, `D_shortcut`, `D_final`, `lemma46_cuts`, `lemma46_queries`, `lambda_n`, `query_bound`) |
| running time (not in the paper) | `PROOF.md`, `cake_cutting/opbounds.py`, `cake_cutting/opcount.py`, `cake_cutting/polybound.py` |
| `X ⊕ Y`, `Id(T)` | `cake_cutting/allocation.py` (`Allocation.oplus`), `main.identity` |
| Appendix C.2 deterministic choices | `TIE_BREAKING` in `main.py`, module docstring of `subcore.py` |

## Query complexity

The paper proves `Λ_n = K n^d 2^n` for unspecified constants.
`cake_cutting/bounds.py` instantiates the accounting of Propositions 4.3, 4.4,
C.1 and Lemma 4.6 with explicit constants:

* a split of one interval (one `Cut` by one agent, or a point chosen by the
  protocol) costs `n` queries with `n` active agents (Prop. C.1);
* `SubCore` on `m` agents makes at most `S_m = 2^{m+1} − 2m − 2` trims
  (Appendix D), also when it fails, and the equilibrium fallback on `k` nonempty
  pieces at most `E(m, k) = (m − 1) k C(m + k − 3, m − 2)` cuts; `Core` on `n`
  agents makes `n − 1` cutter cuts and at most `S_{n−1} + E(n−1, n)` trims and
  cuts, hence `core_queries(n) = n (n − 1 + S_{n−1} + E(n−1, n))`;
* Stage 1: `nQ^2` snapshot calls plus at most `nQ^3 · D_mid(n)` calls in
  elimination sequences (Lemma 4.7); Stage 2: at most `n · D_shortcut(n)`;
  Stage 3: at most `n · D_final(n)` calls (Section 4.6);
* Lemma 4.6 (the rounding procedure, see below): at most
  `m n^2 ⌈log2(n D / (2δ))⌉` cuts for `m` requests (`D = ⌈log2(m + 1)⌉`), each
  costing at most `n` queries;
* the direct SubCore calls of Lemma 4.17 make at most `S_n + E(n, n)` trims and
  cuts in total;
* `T(n) = n + 15 + Σ_{k=4}^{n} Λ_k` (Theorem 1.1's proof; the base cases use at
  most 2 resp. 15 queries in the ordered-piece implementation of this repository).

All other computation (bonuses, marks, graphs, matchings, the linear algebra in
Lemma 4.6, all run-time checks) uses only stored values and costs no queries.
The same `T(n)` satisfies

    T(n) ≤ n^{11} · 4^n / 4          for every n ≥ 1   (equality at n = 1),
    T(n) ≤ 0.0475561 · n^{11} · 4^n  for n ≥ 2         (maximum at n = 34),

proved in `PROOF.md` §9 (exact values for `n ≤ 399`, an explicit majorant beyond;
the second constant is `bounds.MONOMIAL_C2`, an exact fraction).  The exponent 11
is the smallest integer with any such constant: `Λ_n` grows as
`Θ(n^{10.5} ln n · 4^n)` from the fallback cuts charged to the `n Q^3 D_mid(n)` Core
calls of stage 1.  Without the fallback the accounting gives
`T(n) ≤ n^{10} 2^n / 2`.  The Lemma 4.6 procedure of the paper's proof
(`n^2/γ ≈ 10^{12}` queries at `n = 4`) is not used by the protocol and does not
enter `T(n)` (see *Deviations and limitations*).  The run-time statistics
(`stats` returned by `run_protocol`) additionally expose the per-stage Core-call counts, the Lemma 4.6
query count and the largest single Core call, and `tests/test_main.py` asserts
each against its own per-stage bound.

## Running time

`PROOF.md` proves an explicit bound `U(n)` on the number of **operations** of
`run_protocol` with `n` agents, for every instance: one operation is one executed
bytecode instruction of the package (CPython 3.13), with arithmetic on numbers
(`int`, `Fraction`), dictionary operations and oracle queries counting one each and
built-ins that iterate over data counting one per element (`cake_cutting/opcount.py`).

    U(n) ≤ 392 · n^{11} · 4^n   for every n ≥ 1     (equality at n = 1),
    U(n) ≤ k2 · n^{11} · 4^n    for every n ≥ 2,    k2 = U(4) / (4^{11} · 4^4) = 161700700527/1073741824 ≈ 150.60,

and both constants are the smallest for which this holds (the ratio is 0.87 at
`n = 2`, 150.60 at `n = 4`, about 80 for `5 ≤ n ≤ 11`, and decreases afterwards:
69.5 at `n = 20`, 48.9 at `n = 40`); no constant works with `n^{10} 4^n`.
Asymptotically `U(n) = O(n^{10.5} log n · 4^n)`.  The bound is built function by
function from instruction profiles of the code (`cake_cutting/_opprofiles.py`,
regenerated by `python -m cake_cutting.opcount` and checked against the code by the
tests) in `cake_cutting/opbounds.py`; for large `n` the same formulas are evaluated
on exact polynomials in `n`, `2^n`, `C(2n, n)`, `log2 n` (`cake_cutting/polybound.py`),
with `C(2n, n) ≤ 4^n / sqrt(3n + 1)`.  Two
implementation choices make such a bound possible: ordered pieces are persistent
balanced trees, so cutting, trimming and concatenating cost `O(n log C)` rather than
`O(C)` for `C` components, and SubCore works on *suffix views* that cost `O(1)` to
create.  The bound is `U(4) ≈ 1.61·10^{11}`; traced runs with `n = 4` executed at most
`2·10^7` instructions.  The fallback's share is computed from exact counts of its
recursion (`PROOF.md` Lemma 5.3, checked against the code by the tests).

## Running

Python 3.13 (the operation bound is stated for CPython 3.13 bytecode, and the
tests check the instruction profiles against it).  From the project directory:

```
python -m pip install -r requirements-dev.txt     # pytest, for the tests
python -m cake_cutting --agents 4 --seed 2 --family concentrated [--cut-rule right] [--verbose]
```

prints the allocation, every agent's value of every piece, completeness /
envy-freeness, the query count and the bound.  Families: `random`,
`identical`, `perturbed`, `zero`, `concentrated`, `blocky`, `uniform`, `pairs`.

Tests (pytest; `networkx` is not used):

```
python -m pytest -q            # default suite, about 7-9 minutes
python -m pytest -q -m slow    # optional, can take hours: n = 5, 6 with persistent residues, n = 7
```

## Test runs (sanity checks)

The bounds in this README (`T(n)`, `U(n)` and their constants) are
deduced mathematically in `PROOF.md`, for every `n`; no test run is part of
that argument.  The runs below only check correctness on concrete instances and
that observed counts stay below the proved bounds.

Default suite (Python 3.13, Linux, one core):

```
425 passed, 26 deselected in 506.69s (0:08:26)
max observed queries per n in this run (total / largest single Core call):
n=1: 1 / 0 (bounds 1.00e+00 / 0), n=2: 4 / 0 (bounds 4.00e+00 / 2), n=3: 18 / 0 (bounds 1.80e+01 / 21),
n=4: 8937 / 44 (bounds 1.21e+07 / 172), n=5: 44979 / 100 (bounds 7.68e+08 / 1255),
n=6: 147828 / 211 (bounds 2.94e+10 / 8406), n=7: 707 / 327 (bounds 7.62e+11 / 52290)
```

`python -m pytest -q -m slow tests/test_opbounds.py -k operation_constant_all_n`
(about 2 minutes) checks the operation constants exactly for every `n <= 40`.

The optional slow tests (`-m slow`: instances whose residue survives thousands
of Core calls, `n = 5, 6, 7`) are not needed for any claim here and can take
hours.  The time goes into exact rational arithmetic, not queries: the cut
points' denominators grow with the number of Core calls (`≈ n^{nQ^2}`,
thousands of bits).

Core/SubCore alone were additionally stress-tested on ~3 100 sequences of up to
60 consecutive Core calls (`n = 4..7`, 14 families, 30 seeds, both cut rules,
fixed and rotating cutters, Bechler–Bitikofer's instance with every cutter):
every call satisfied Proposition 4.3 (i)–(ii) and the query bound, without
failures.

## Deviations and limitations

Everything below is either a choice the paper leaves open, a place where the
literal text had to be interpreted, or something this implementation does not
cover.  Nothing else deviates from Section 4 / Appendix E.

**Lemma 4.6 (`simultaneous.py`).**  The proof's procedure (every agent cuts
`R*` into pieces worth `γ V_i(R*)`, then a labeling of the cells is searched)
needs `≈ 10^{12}` cuts per agent at `n = 4` and an exhaustive search over the
labelings.  The lemma only asks for *a* deterministic bounded procedure with its
guarantee, so the protocol uses a *rounding* procedure with a proof (module
docstring of `simultaneous.py`, `PROOF.md` §6): the requests with `λ_j > δ` and a
label "unused" are the leaves of a balanced binary tree; every internal node splits
its cells between its children so that the left child gets the fraction
`p = λ(left)/λ(node)` of every positive agent's value up to `(r/2)γ V_i(R)`, by
keeping fractional weights whose `r` weighted sums are exact (a kernel step makes a
cell integral whenever `r + 1` are fractional) and halving cells that are not yet
fine; errors add up along the `D` levels to `δ`.  It makes at most
`m n^2 ⌈log2(n D/(2δ))⌉` cuts and polynomial work.  Requests with `λ_j ≤ δ` get
the empty piece (a valid output of the lemma); in particular an addition
`A_{s,k}` is empty unless some agent has a positive small bonus over `(s, k)`.  The
literal procedure is implemented for the tests only (`force_fallback`, with a
coarse `γ`).

**Core / SubCore (`subcore.py`).**  Ye–Bai use AM's SubCore with line 12
skipped and refer to AM's tie-breaking by "imaginary values", which is not
specified precisely enough to implement.  Bechler and Bitikofer (2017) showed
that the literal protocol can reach "Case 2" of AM's Claim 8 (an unallocated
contested piece without a non-winner trim) which AM's proof declares
impossible; random testing finds two more failure modes of the literal text:
(a) two agents' trims may coincide and the wrong winner breaks the benchmark
invariant of AM's induction (an agent ends below her required minimum), and
(b) an agent whose benchmark is 0 "trims to value 0" at the very end of every
contested piece and becomes the rightmost trimmer of pieces of which she receives an empty suffix.  The
realization used here (documented in the module docstring) passes *hints* to
the recursive calls — the piece an agent holds at the start of the current
step (she keeps preferring it among physically equal pieces, which is exactly
the property "ties broken in favour of `a`" that AM's Case-2 argument needs)
and the suffix she is guaranteed by the previous tentative allocation (an agent
reclaiming exactly her guaranteed suffix wins a tie on it).  Ties are otherwise
broken by the current holder, then by the lowest index.  Trim positions are
compared after moving them left over cake the trimmer is known (from stored
values) to value at zero, so the outcome does not depend on which admissible
point a `Cut` query returns (both the leftmost and the rightmost answer are
tested).  "Case 2" and every other guarantee are *detected*; when one fails,
`subcore` discards the attempt and computes its output by the equilibrium fallback
(`equilibrium.py`).
The cutter in Core takes the lowest-index unallocated piece; if she values the
residue at zero the fixed rule of C.2 makes piece 1 the whole residue.

**Deterministic choices in `Main`** (`TIE_BREAKING` in `main.py`): agents are
processed in increasing index order everywhere; the agent receiving `R*` when
nobody values it, the Stage-2 cutter and the pair whose witnesses are pruned
are the smallest ones; the shortcut uses the first snapshot whose graph is not
strongly connected and the sink component containing the smallest agent; the
witness for a pair is the earliest unused one, the label the smallest, and the
matching closes the shortest `k → h` path found by breadth-first search; `J`
is padded with the earliest unused snapshots; `E*` is the smallest profile
class; `R_1` lists the unused part of `R*` in the order of `R*` followed by
the prefixes `Y_h` in agent order.  Ordered pieces keep their components in
the order in which they were created (left to right within a component), as
in Appendix C.1; a trim keeps the remaining components in the same order.

**Base cases.**  Selfridge–Conway on ordered pieces uses at most 15 queries
here (five cuts, each with two Evals) against the paper's 14; the explicit
bound uses 15.

**SubCore fallback (`equilibrium.py`).**  When Aziz–Mackenzie's SubCore raises, the
output is computed from a competitive equilibrium of the market whose buyers are
the agents, whose goods are the nonempty pieces and in which a price is a trim
position: the prices of the highest equilibrium of the market without one agent
(the recursion of Alaei–Jain–Malekian, Theorem 2, memoized over pairs of buyer and
good sets), then a matching that supports them.  `CORRECTNESS.md` §8 proves the
theory and the code for positive densities.  It costs at most
`(m − 1) k C(m + k − 3, m − 2)` cuts on `m` agents and `k` pieces, about `4^m`;
Aziz–Mackenzie's procedure runs first, so a run in which it never fails costs the
same as without the fallback.

**Explicit bounds.**  `T(n)` instantiates the paper's accounting with the
constants of Props. 4.3/4.4/C.1 and Lemma 4.6 and the fallback's cuts; as a
monomial it is `n^{11} · 4^n / 4` (`bounds.MONOMIAL_C`, `bounds.MONOMIAL_K`); the
exponent cannot drop to 10.  The operation bound `U(n) ≤ 392 · n^{11} · 4^n` is in
`PROOF.md`.  Both are worst-case bounds: they charge every Core call of the
stage-1 elimination sequences, which dominate them and are rare in practice, with
a full run of the fallback, which is rarer still.

**What is not covered by the tests.**  Recursion of `Main` into a call with `≥ 4`
agents through Stage 3 requires two profile classes of size `≥ 4`, i.e. `n ≥ 8`,
which the tests do not reach; recursion into `Main` with 4 agents is exercised
through the sink-component shortcut (`n = 5`).  `CORRECTNESS.md` proves from the
code that the Stage-3 recursion is correct for every `n` (the recursive call gets
every value it needs, the envy-freeness and domination checks around it cannot
fail, and the combined allocation is complete and envy-free), and that a run
that returns always returns a complete envy-free allocation.

**SubCore failures and what remains open.**  `CORRECTNESS.md` §6 proves that every
check outside SubCore passes whenever every SubCore call returns.
Aziz–Mackenzie's SubCore alone does not always return: for the 5-agent input
`SUBCORE_COUNTEREXAMPLE` in `tests/test_main.py` (piecewise-constant densities with
zero regions, `cut_rule="right"`) its `while` loop finds no agent to add (Claim 8 of
Aziz–Mackenzie, on which the paper relies, fails); a random search found such
inputs at a rate of a few per 10^5, mostly with zero-density regions but also with
strictly positive densities (`CORRECTNESS.md` §7).  The fallback completes that
input with both cut rules, and every failing input of the search except one.  With
positive densities it cannot fail (Theorem D).  With zero-value stretches it can
(`CORRECTNESS.md` §9): neither this repository nor the literature has a procedure
for that case with a proof and a bound in `n`; Kurokawa, Lai and Procaccia (AAAI
2013, Theorem 1) show that bounded protocols for piecewise-uniform valuations,
which have such stretches, already give bounded protocols for all valuations, so
this is the general problem.  No run returns a wrong allocation (Theorem A).

For `n = 7` only instances whose
residue empties early are tested (1–6 s); an `n = 7` instance with a persistent
residue needs the full `7 · 42^2 = 12 348` snapshot Core calls on rationals with
tens of thousands of bits and did not finish within an hour.  Valuations in the
tests are piecewise constant with rational breakpoints and densities (the only
kind for which every `Cut` answer is an exact rational); the protocol itself
makes no such assumption.  Memory and time grow with the bit length of the cut
points, which is inherent to exact arithmetic.  The value table keeps every value
it has learned and never discards any: discarding values would only save memory,
its cost cannot be bounded per call, and it could change which trim positions
`normalize_left` finds.
