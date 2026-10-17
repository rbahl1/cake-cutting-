# Running time of the implementation: `U(n) <= k · n^11 · 4^n`

This document proves an explicit upper bound `U(n)` on the number of operations
executed by `run_protocol` on `n` agents, including the equilibrium fallback of
SubCore (`CORRECTNESS.md` §8), shows that `U(n) = O(n^10.5 log n · 4^n)`, and computes
the smallest constants with `U(n) <= k n^11 4^n`:

    k  = U(1) / 4 = 392                                          for every n >= 1,
    k2 = U(4) / (4^11 · 4^4) = 161700700527 / 1073741824 ≈ 150.60  for every n >= 2.

11 is the smallest integer exponent: `U(n) / (n^10 4^n)` is unbounded.  Without the
fallback (Aziz–Mackenzie's SubCore alone, which can fail) the same accounting gives
`6950.18 · n^10 · 2^n`; the fallback costs up to `(m-1) k C(m+k-3, m-2)` cuts per
SubCore call, which grows like `4^n`.

The bound is a function of `n` alone; it holds for every valuation profile and
both cut rules of the oracle.  `k` is the smallest constant for **this proved
bound** `U(n)`: `k` is attained by `U` at `n = 1`, `k2` at `n = 4`.  It is not
claimed that some instance makes the program execute `U(4)` operations: the traced
runs with `n = 4` executed at most about 2·10^7 instructions, against
`U(4) ≈ 1.61·10^11` (see §10); the gap comes mostly from the worst-case count of
stage-1 sequences and from charging every Core call with a full run of the fallback.

The bound is assembled by the code in `cake_cutting/opbounds.py`: every lemma below
corresponds to a function `c_*` (or a method of `Ctx` / `MainCost`) whose value is
the right-hand side of the lemma, built from the instruction counts of the program
(`cake_cutting/_opprofiles.py`).  The proof is the argument that each such function
bounds the corresponding program function; the arithmetic is the code's.

**Scope.**  The theorem is about runs of `run_protocol(oracle)` that return (the
protocol raises `ProtocolError` when a checked guarantee fails; `CORRECTNESS.md`
proves that a run that returns returns a complete envy-free allocation, and that
every run returns when all agents have positive density, Theorem D), with the
default arguments `trace=None` and
`check=True`, and an oracle whose agents are the integers `0, ..., n-1`
(`RobertsonWebbOracle` with integer keys, as everywhere in the repository).

## 1. Cost model (`cake_cutting/opcount.py`)

One **operation** is

* the execution of one CPython 3.13 bytecode instruction of the package
  (`dis.get_instructions`); an instruction counts one also when it handles a
  number of items fixed by the program text (building or unpacking a 3-tuple,
  comparing two pairs, one step of `zip` or `enumerate`);
* an arithmetic operation, comparison, hash, construction or conversion of an
  `int` or `fractions.Fraction`: the instruction that performs it (unit-cost
  arithmetic; the Python code of `fractions` is not counted);
* a dictionary or set lookup, insertion or deletion: the instruction that
  performs it, including the `__hash__` and `__eq__` methods it calls (the usual
  unit-cost assumption for hash tables);
* an oracle query (`eval`, `cut`) or access to the oracle's counters: one
  (what the agents do to answer is not the protocol's work).

In addition, a built-in whose work grows with the size of its data costs **one
per element it touches**: `len`-free container constructors (`list`, `tuple`,
`dict`, `set`, `frozenset` of a collection), `sorted` (`L·ceil(log2 L)`
comparisons, each costing the comparison of its keys, plus `2L + 1`), `sum`,
`max`, `min`, `any`, `all` over an iterable, `x in list`, list repetition
`[x] * L`, slicing, `list.extend`, `list.pop(0)`, comparison and algebra of sets and
frozensets, equality of tuples and lists of non-fixed length, `repr` and string
comparison (one per character).  Text of exception messages on error exits and
code executed only when `trace is not None` are not counted.

### Lemma 1.1 (instruction profiles)

For a function `f`, `opcount.profile(f)` attributes each instruction to the
innermost loop (`for`, `while`, inlined comprehension, `yield from`) whose source
span contains its position; the iterable of a `for` or of a comprehension's first
`for` clause is attributed to the enclosing level.  Three kinds of instructions
are placed by their offset instead, because CPython 3.13 gives them a position that
lies outside the loop they belong to: `FOR_ITER`, which carries the position of the
iterable and runs once per iteration plus once per entry, goes to its loop together
with its `EXTENDED_ARG` prefixes (a loop body longer than 255 code units); a
backward jump belongs to the loop that owns its target; and an instruction inside
the cycle of a loop but attributed to an enclosing level (for instance the entry of
an exception handler, which has no position) goes to that loop.  Within one level
the only backward jumps are the loop's own (`JUMP_BACKWARD*`), so:

* an instruction lying between the target of such a jump and the jump (the
  loop's **cycle**) executes at most once per iteration plus once per entry
  (the final `FOR_ITER` or loop test);
* every other instruction of the loop executes at most once per entry;
* an instruction outside every loop executes at most once per call.

Hence one call of `f` executes at most

    outside + Σ_loops [ cycle · (iterations + entries) + entry · entries ]

of its own instructions, plus the operations of the functions it calls.  `own(key,
t_1, ..., t_L)` in `opbounds.py` is this expression, with `t_j = (iterations,
entries)` or one number bounding `iterations + entries`.  A lambda or generator
expression is a separate code object; one call of a lambda, and the work of a
generator expression with one `for` clause per element of its iterable (plus once
for its exhaustion), execute each of its instructions at most once (`nested(key)`).
The profile records the number of loops of such a code object; the one generator
expression of the protocol with two `for` clauses (in `sink_component`) is charged
`nested(key)` once per execution of either clause's `FOR_ITER`.

`tests/test_opcount.py` checks the expression above against the instructions
executed (counted with `sys.monitoring`) for functions with known loop counts that
cover every construct concerned.

`tests/test_opbounds.py::test_profiles_match_the_code` regenerates the profiles and
compares them with the frozen copy, so the bound cannot silently diverge from the
code.

## 2. Persistent AVL sequences (`seq.py`)

A tree node stores its component, its value vector (length `n`, the number of
agents of the table), height, size, first and last component and the vector of
its subtree's values, computed in `Node.__init__` from the children's.

**Node.**  Creating a node executes `Node.__init__` with exactly one of its three
comprehensions over the `n` agents; the one for two children is the most
expensive: `c_node(n) = own(Node.__init__, 0, 0, (n, 1)) = 152 + 17n`.

**Lemma 2.1 (join).**  `join(L, x, R)` with `D = |h(L) - h(R)|` makes at most
`D - 1` invocations of `_join_right`/`_join_left` (each with three `height` calls),
at most `D` rotations and creates at most `3D + 1` nodes; the result is an AVL tree
of height `max(h(L), h(R))` or `max(h(L), h(R)) + 1`.

*Proof.*  If `D <= 1`, `join` creates one node.  Otherwise (say `h(L) > h(R) + 1`)
`_join_right` descends the right spine of `L` while `height(c) > height(R) + 1`; the
height of its first argument decreases by at least 1 per level and stays at least
`h(R) + 2`, so there are at most `D - 1` invocations.  A non-final invocation
creates one node and possibly one rotation (two nodes); the final one creates two
nodes, or six with a double rotation.  So at most `3(D - 2) + 6 = 3D` nodes and
`(D - 2) + 2 = D` rotations.  The height statement and the AVL invariant are
Lemma 1 of Blelloch, Ferizovic and Sun, *Just Join for Parallel Ordered Sets*
(SPAA 2016), whose AVL join this is; `tests/test_seq.py` checks both on random
trees.  ∎

`c_join(D, n) = own(join) + 2·HEIGHT + max(0, D-1)·JOINSTEP + D·ROT + (3D+1)·c_node(n)`.

**Lemma 2.2 (split).**  Let `h(T) <= h`.
`split(T, k)` makes at most `h + 1` invocations and at most `h` joins, whose height
differences sum to at most `3h`; `split3(T, k)` (split and extract element `k`)
makes at most `h` invocations and `h - 1` joins with the same sum bound;
`_split_last(T)` makes at most `h` invocations and `h - 1` joins with differences
summing to at most `2h`.

*Proof.*  `split` follows one root-to-leaf path `t_0, t_1, ...` with strictly
decreasing heights `h_0 > h_1 > ...`, so at most `h` nodes, plus the final call on
an empty tree.  Consider the joins building the right part, at the levels
`i_1 > i_2 > ... > i_q` (deepest first) where the path goes left: the `j`-th joins
the right part `R_{j-1}` built so far with the right child of `t_{i_j}`, of height
`r_j ∈ [h_{i_j} - 2, h_{i_j} - 1]`.  By Lemma 2.1, `h(R_j) <= max(h(R_{j-1}), r_j) + 1`
and `h(R_j) >= r_j`.  Induction gives `h(R_j) <= h_{i_j}` (`R_0` is empty, and
`h_{i_{j-1}} <= h_{i_j} - 1`).  The first difference is `r_1 <= h_{i_1}`.  For
`j >= 2`, `R_{j-1}` has height in `[r_{j-1}, h_{i_{j-1}}] ⊆ [h_{i_{j-1}} - 2, h_{i_j} - 1]`,
so `|h(R_{j-1}) - r_j| <= h_{i_j} - h_{i_{j-1}} + 1`.  The sum telescopes to at most
`h_{i_q} + q - 1`.  The joins on the left part give at most `h_{i'_{q'}} + q' - 1`
in the same way, and `q + q' <= h`, so the total is at most `3h - 2`.  For
`split3` the recursion stops at the extracted node `t*`; the parts start as
`t*.left`, `t*.right` of height at most `h(t*) - 1`, which only shortens the first
difference.  For `_split_last` (the path along the right spine, the parts start
with the left child of the last node, of height `<= 1`) only left-part joins occur,
and the same telescoping gives at most `g_0 + p - 2 <= 2h`.  ∎

With `_joins(count, dsum)` = `count` joins whose differences sum to `dsum`:
`c_split = (h+1)(own(split) + SIZE) + _joins(h, 3h)`,
`c_split3 = h(own(split3) + SIZE) + _joins(h-1, 3h)`,
`c_split_last = h·own(_split_last) + _joins(h-1, 2h)`,
`c_concat = own(concat) + c_split_last(h) + c_join(h)`.
`nth`, `find_ge`, `prefix_sum` run at most `h` iterations of their `while` loop
(one level of the tree each) with one `size` call each.

## 3. Suffix views (prepaid evaluation)

A suffix view (`OrderedPiece._view`, created only by `ValueTable.suffix_from`)
stands for components `k0, k0+1, ...` of a tree, the first possibly replaced by a
suffix `head`.  `ValueTable.value` of a view computes, the first time for each
agent, one prefix sum of the tree, and caches the result in the view.

**Lemma 3.1.**  Values are computed only for active agents, and a view created in a
call `Main(N)` is evaluated only during that call.  Hence a view is evaluated for at
most `m = |N|` agents, and charging `m · c_prefix_sum(h)` to its creation
(`c_view_prepaid`, part of `c_suffix_from`) pays for all its first evaluations;
every evaluation then costs `c_value(h) = own(value) + LEN + BASE`.

*Proof.*  Every `value(agent, ·)` of the protocol is for an agent of the current
`Main` call (`table.set_active(N)`; the agents of a recursive call are a subset).
A view is an allocated piece or a margin of SubCore.  After `Main(N)` returns,
its caller only concatenates the returned pieces (`X.oplus(...)`), which does not
evaluate them.  ∎

## 4. Pieces and sizes

**Lemma 4.1 (representations).**  (a) The only nonempty tuple-backed piece that
reaches the value table is the cake; its first evaluation builds its one-node tree
(`c_cake_tree`, charged once at the top level).  Every other nonempty piece is
tree-backed.  (b) Suffix views are created only by `suffix_from` (SubCore).
`prefix_of_value`, `suffix_of_value`, `prefix_to` and `prefix_at` receive trees,
except `unallocated_part(W_h, ·)` in stage 3, where `W_h` may be a single snapshot
piece, i.e. a view of a residue-derived tree (charged a `full_tree` conversion).
Every residue is a tree.

*Proof.*  Inspection of the call sites (listed in `opbounds.py`).  Pieces are made
by `OrderedPiece.cake()`, `OrderedPiece.empty()`, `_from_tree` (results of
`split`/`join`/`concat`) and `_view`.  Residues are `concat` results of leftovers,
which are trees (`prefix_at`, `prefix_to`, or division pieces); `R_1` concatenates
the Lemma 4.6 leftover and the `Y_h`, which are trees unless `Y_h = W_h`, which
happens only with an empty allocated suffix and then raises unless `D_h` is
empty (and then `W_h` is a concatenation of trees).  ∎

**Lemma 4.2 (sizes).**  Let `K(m) = mQ^2 + mQ^3 D_mid + m D_shortcut + m D_final`
(the Core calls of one `Main(m)` call, `Q = m(m-1)`), `H_46(m)` the Lemma 4.6 cut
bound (§6) and

    R_max(n) = 1 + Σ_{m=4}^{n} [ (m-1) K(m) + H_46(m) + m ],   C_res = R_max + n + 8,
    A_max(n) = 2 + Σ_{m=4}^{n} [ K(m) + 2Q + 2 ],              C_all = A_max · C_res.

Every residue has at most `R_max` components, every residue-derived tree (division
pieces, their prefixes, suffixes and views, Core allocations, snapshot pieces,
Lemma 4.6 cells and pieces, base-case pieces) at most `C_res` nodes, and every tree
at most `C_all` nodes.  Hence (AVL) every residue-derived tree has height at most
`h_res = max_height(C_res)` and every tree at most `h_all = max_height(C_all)`.

*Proof.*  A Core call on a residue with `C` components produces division pieces
with at most `C + m - 1` components in total (each of the `m - 1` prefix cuts
splits at most one component); the leftover of a piece with `L` components has at
most `L` (a suffix starting in component `k` leaves `k` components, the last
possibly shortened), so the new residue has at most `C + m - 1`.  The Lemma 4.6
cells are runs of `R*`, each halving adds at most one component, so all its pieces
together have at most `|R*| + H_46`; `R_1 = [leftover, Y_h ...]` with
`|Y_h| <= |P_h| + |A_h| + 1` (the trim does not enter `D_h`, checked at run time),
and the `A_h` use distinct addition pieces (`μ_s` is a permutation), so
`|R_1| <= |R*| + H_46 + m`.  Residues are passed down the recursion, so all residues
of the run are bounded by `R_max`.  Base cases add at most 5 cuts.  A tree that is
not residue-derived is a concatenation of residue-derived pieces: per `Main(m)` an
agent's piece collects at most one Core allocation per Core call (`K(m)`, snapshots
included), one `Z` suffix of `W_h = [P_h, A_h, D_h]` (at most `2Q + 1` atoms), and
`R*` in the zero-residue exit; over the recursion chain and the base case this is
at most `A_max` atoms.  An AVL tree with `C` nodes has height at most
`max_height(C)` (`F(h + 2) - 1 <= C`).  ∎

At `n = 4`: `R_max = 228137`, `h_res = 25`, `h_all = 48`.

## 5. Value table and SubCore

The functions of `bookkeeping.py` are bounded line by line in `opbounds.py`
(`c_value`, `c_split_by_cut`, `c_locate`, `c_position_of_value`,
`c_normalize_left`, `c_suffix_from`, `c_prefix_of_value`, `c_prefix_to`, ...):
each is its `own(...)` plus the bounds of the functions it calls, taking for
alternative paths the maximum (`pmax`).  A Cut or Eval costs one operation; a new
interval's value vector (`[None] * n`, `n` elements) is charged to the operation
that creates the interval (`split_by_cut`: two, `complement_prefix`: one).

**SubCore.**  `Ctx(n, m, h)` collects the costs of one SubCore family whose pieces
are views of trees of height `<= h`.  `A(w, P)` bounds one `SubCoreCall` with `w`
agents and `P` pieces (`__init__`, `run`, `favorite_unallocated` for every agent,
`contest(s)` for `s = 1, ..., w-1`, `verify`) **without** its trims and without its
recursive calls; `contest(s, w, P)` follows the 20 loops of `contest` with
`g = s + 1` agents in the group, `s` contested and `P - s` uncontested pieces, and
at most `s - 1` iterations of the `while` loop (each adds a winner).  Step 0 never
contests (nothing is allocated yet).

**Lemma 5.1 (SubCore recursion).**  In a call with `w` agents, the contest of step
`s` makes at most `s(s+1)` trims and recursive calls with distinct numbers of agents
`j ∈ {1, ..., s-1}`, each on `s` pieces.  Consequently the whole call tree makes at
most `S_w = 2^{w+1} - 2w - 2` trims, and costs at most

    R(w, P) + S_w · TRIM,    R(w, P) = A(w, P) + T(w),    T(w) = Σ_{s<w} Σ_{j<s} R(j, s),

where `TRIM` bounds `suffix_position_of_value + normalize_left` (one real trim).

*Proof.*  Each agent of the group places at most one trim per contested piece
(`place_trim` returns without a trim when the hint applies).  The recursive calls
are made in the `while` loop with the current winners, whose number increases by
one per iteration (`added`, else `ProtocolError`), on the margins of the `s`
contested pieces.  The trim count satisfies `t(w) <= Σ_{s=1}^{w-1} [s(s+1) +
Σ_{j<s} t(j)]`, which `S_w` satisfies with equality (`tests/test_bounds.py`).  ∎

**Lemma 5.2.**  With `α(u) = Σ_{j<u} A(j, u)` and `Z = Σ_{u>=1} α(u) / 2^u`,
`T(w) <= 2^{w-1} Z` for all `w >= 1`.

*Proof.*  `T(w+1) - T(w) = Σ_{j<w} R(j, w) = α(w) + W(w)` with `W(w) = Σ_{j<w} T(j)`,
and `W(w+1) - W(w) = T(w)`.  So `V = T + W` satisfies `V(1) = 0`,
`V(w+1) = 2V(w) + α(w)`, i.e. `V(w) = 2^{w-1} Σ_{u<w} α(u) / 2^u`, and `T <= V`.  ∎

For integer `w` the code evaluates `R` exactly; Lemma 5.2 is used in §8.

**Lemma 5.3 (the equilibrium fallback).**  When Aziz–Mackenzie's procedure raises,
`subcore` calls `equilibrium.allocate` with the same `w` agents and `P` pieces, `k` of
them nonempty (trees of height `<= h`).  With `a = w - 1`, its recursion
(`_Market.high`, `_Market.low`, starting at the high state of `a` buyers and all `k`
goods) has

* `HS = C(a+k, a)` high states and `LS = C(a+k, a+1)` low states,
* `HC = 1 + a C(a+k-1, a)` calls of `high` and `LC = k C(a+k-1, a)` calls of `low`
  (memo hits included),
* `VIEWS = k C(a+k-1, a)` suffixes (`suffix_from`), `LX = a C(a+k-1, a)` `utility`
  calls from `low` (and `w` more at the top),
* `HXY = a k C(a+k-2, a-1) = E(w, k)` iterations of the inner loop of `high`, each at
  most one `suffix_position_of_value`, and `LXY = a k C(a+k-2, a)` values read by
  the `utility` calls from `low`.

*Proof.*  The high states reached are `(X, Y) = (I − i_0 − A, J − B)` with
`|A| = |B| = t`, the low states those with `|A| = t`, `|B| = t + 1`; a computed high
state `(X, Y)` calls `low(X, Y − g)` for its `|Y|` goods and runs its inner loop over
the `|X|` buyers of each; a computed low state calls `high(X − b, Y)` for its `|X|`
buyers and reads `|Y|` values for each.  Each state is computed once (memo).  The
sums `Σ_t C(a,t) C(k,t) (…)` are Vandermonde convolutions, e.g.
`Σ_t C(a,t) C(k,t) (k−t) = k Σ_t C(a,t) C(k−1,t) = k C(a+k−1, a)`.  Every state of
these shapes is reached, so the counts are exact
(`tests/test_opbounds.py::test_equilibrium_counts_are_exact`).  ∎

`Ctx.fallback` charges: `_Market.__init__` (`P` length tests, `k` starts, `wk` whole
values); `own_calls` of `high` (`HC` calls, its two loops with `k·HS` and `HXY`
iterations), `HXY` suffix positions and `VIEWS` suffix views; `own_calls` of `low`
(`LC` calls, `w·LS` iterations); `LX + w` `utility` calls reading `LXY + wk` values;
the tight graph (`wk` values); `_cover_matching` (phase 1 at most `w` augmentations,
each dequeuing each buyer once and scanning each of at most `wk` edges once; phase 2
at most `k` searches, each dequeuing each good once and scanning at most `wk` edges;
paths of at most `k` goods); the free indices (`sorted` of at most `P` keys); and
`_verify` (`w + w^2 + wP` values).  `Ctx.subcore` adds this to the Aziz–Mackenzie
bound of Lemma 5.1, whose trim count `S_w` also holds for a failed attempt (its run
is a prefix of a complete one).

`core(m)` = `equal_division` (`m - 1` prefix cuts) + `subcore(m - 1, m)` with the
fallback on `k = m` pieces + at most `m - 1` `unallocated_part` + one concatenation of
`m - 1` trees + the final check (`c_core`).

## 6. Lemma 4.6: the rounding procedure (`simultaneous.py`)

With `K` requests (`K <= max_requests(m) = mQ + m - 1`), `L <= K + 1` labels, depth
`D = bit_length(K)`, `r <= m` agents of `A+`, `γ = 2δ/(rD)`:

**Lemma 6.1.**  (i) Along any chain of halvings an agent cuts at most
`ceil(log2(1/γ))` times, so a chain has at most `H_c = r ceil(log2(mD/(2δ)))`
halvings; (ii) a split makes at most `H_c` rounds, each halving at most `r` cells;
(iii) the procedure halves at most `H_46 = K m^2 ceil(log2(mD/(2δ)))` cells
(`bounds.lemma46_cuts`); (iv) the cells entering the splits number at most
`D(1 + H_46)` in total, and at most `1 + H_46` cells reach the leaves.

*Proof.*  (i) The cutter of a cell is the agent of largest share `V_i(K)/V_i(R)`;
a cell is halved only if it is not fine, so that share exceeds `γ`; both halves
have half the cutter's share, and shares never increase along a chain.  (ii)
Fine cells are never halved, so a cell that is coarse in round `k + 1` of a split
is a half made in round `k` of a cell coarse in round `k`; it lies on a chain with
`k` halvings in this split.  `|F| <= r` at the start of every round.  (iii) At most
`L - 1 <= K` splits.  (iv) The cells of one level of the label tree are disjoint
runs, at most `1 +` (halvings above).  ∎

**Lemma 6.2.**  Every `_reduce` removes at least one cell from `F`; hence the
reductions of all splits number at most `D(1 + H_46) + H_46`.

*Proof.*  The kernel vector `z` is nonzero and the step `t` is the smallest
distance of a coordinate to 0 or 1 along `z`, so at least one weight becomes
integral.  Cells enter `F` by insertion or as the extra half of a halving.  ∎

`c_lemma46` assembles `__init__` (sorting the request keys by `repr`, at most
`13 + 2·digits(n-1) + digits(Q^2)` characters each), the label tree, the splits
with these counts, the `kernel_vector` calls (`r` rows, `r + 1` columns), the
concatenation of the leaves and `verify`.

## 7. Main

**Lemma 7.1 (shrink).**  One iteration of `shrink` costs at most a value of the
residue, one Core call, `Allocation(X)` and `K ⊕ X`.  In `K ⊕ X` the piece `K[a]` is
a tree, except in at most one concatenation per agent and per sequence of
`shrink` calls (`H` in stage 1, `Y` in stage 2, `X` in stage 3): `K[a]` becomes a
tree at its first concatenation with a nonempty piece and stays one.  These at most
`3m` concatenations onto a view are charged once per `Main` call
(`MainCost.total`); otherwise appending `X[a]` costs the larger of the view path
(`split3` + join) and the tree path (`_split_last` + join).

**Lemma 7.2 (counts).**  In `Main(m)`: `mQ^2` snapshot Core calls; at most `mQ^3`
prescribed sequences of at most `D_mid` Core calls in stage 1 (checked at run time:
`steps > n Q^3` raises); the pruning loop runs at most `Q^2` passes (each removes
a snapshot; at least `Q` remain); at most `Q^2` sink tests; at most `m` shortcut
sequences of `D_shortcut` calls; at most `m` final sequences of `D_final` calls;
`Lemma 4.6` once; one SubCore call per profile class.

**Lemma 7.3 (class SubCores).**  The profile classes partition the `m` agents into
at least two classes; `Ctx.class_subcores` takes the maximum over all such
partitions (dynamic programming; a class of `e` agents has at most `e` nonempty
pieces).  For a symbolic `m` (§8): at most `m` classes, `A(e, e+1) <= A(m, m+1)`,
`Σ_E 2^{|E|} <= 2^{m-1} + 2` (the sum of `2^e` over a partition into at least two
parts is largest for `(m-1, 1)`), and for the fallback, whose bound for a class of
`e` agents is `F_0(e) + G_e F_1(e)` with `G_e = C(2e, e)` and polynomials `F_0`, `F_1`
with nonnegative coefficients: `Σ_E F_0(|E|) <= m F_0(m)` and
`Σ_E G_{|E|} F_1(|E|) <= F_1(m) · (m/(m−1)) G_{m−1} <= (8/21) G_m F_1(m)`, because
`G_e / e` is nondecreasing (`G_{e+1}/G_e = 2(2e+1)/(e+1) >= 3`) and every class has at
most `m − 1` agents.

**Lemma 7.4 (recursion).**  The `Main` calls of a run have strictly decreasing
numbers of agents (the recursion is on `N \ sink`, with a nonempty sink, or on the
smallest of at least two classes), and at most one base case (`m <= 3`) is
reached.  Hence

    U(n) = c_top(n) + max_{b<=min(n,3)} c_main_base(b) + Σ_{m=4}^{n} Λ_op(m; n),

where `Λ_op(m; n) = MainCost(m, n, h_res, h_all).total()` bounds one `Main(m)` call
without its recursive call (all branches of stage 2 and stage 3 added), and
`c_top` covers `MainProtocol.__init__`, `register_cake` and the cake's tree.

`MainCost` follows `main`, `preprocess`, `prune`, `sink_component`, `stage3`,
`matching_through`, `ranks`, `check_lemma_4_14` and `assert_domination` loop by
loop; every count used is the one stated in the comments of the code.

At `n = 4` the total is `U(4) = 160911634408`, most of it for the
`mQ^3 · D_mid = 69120` Core calls of stage 1, each charged with Aziz–Mackenzie's
SubCore and a full run of the fallback.

## 8. The constant k

**Polynomial evaluation.**  The cost functions accept, wherever they accept an
integer, an exact polynomial (`polybound.Poly`) in the nonnegative variables
`m`, `E = 2^m`, `B = log2 m`, `G = C(2m, m)`, `n`, `N = 2^n`, `M = C(2n, n)`,
`L = log2 n`, `h_r`, `h_a`.  Arithmetic
is exact; every non-polynomial step is replaced by a polynomial upper bound valid
for all values of the variables that occur:

* `max` → coefficientwise maximum (`pmax`); `max(0, x - 1)` → `x`;
* `ceil(log2 x) <= x`, `digits(x) <= x`, an agent's digits `<= L + 1`;
* `D_mid <= ln2 · m(1 + 2B) + 1`, `D_shortcut <= 2 ln2 · mB + 1`,
  `D_final <= ln2 · m(1 + 3.5B) + 1` (from `ceil(log_{m/(m-2)} x) <= (m/2) ln x + 1`,
  `ln(m/(m-2)) >= 2/m`, and `x <= 4m^4, m^4, 4m^7`), with `ln 2 < 0.6932`;
* Lemma 4.6: `D <= 3B + 1`, `ceil(log2(mD/(2δ))) <= 5 + 10B` (`1/δ <= 16 m^8`, `B >= 2`);
* `S_w <= 2^{w+1}`, `T(w) <= 2^{w-1} Z` (Lemma 5.2), with `Z` computed exactly for
  `u <= 40` and from the polynomial bound of `α(u)` beyond (`tail_moment`);
* the counts of Lemma 5.3 for the Core call of `Main(m)` (`w = m − 1`, `k = m`):
  `C(2m−2, m−2) <= G_{m−1}`, `C(2m−2, m−1) = G_{m−1}`, `C(2m−3, m−2) = G_{m−1}/2`,
  `C(2m−4, m−3) <= C(2m−4, m−2) = G_{m−2}`, with `G_{m−1} <= (2/7) G` and
  `G_{m−2} <= (3/35) G` for `m >= 4` (both equalities at `m = 4`;
  `eq_counts_core_sym`); for a class of `e` agents the counts in terms of `G_e`
  (`eq_counts_class_sym`) and Lemma 7.3;
* `Σ_{v=1}^{w-1} v^k <= w^{k+1}/(k+1)` after dropping negative monomials
  (`sum_below`);
* the bounds of Lemma 7.3 for the class SubCores.

**Lemma 8.1.**  For a polynomial `p` in `m, E, G, B` (and other variables), with
nonnegative coefficients after dropping negative monomials,
`Σ_{m=4}^{n} m^a B^c <= n^{a+1} L^c`, `Σ_{m=4}^{n} m^a 2^m B^c <= n^a L^c 2^{n+1}` and
`Σ_{m=4}^{n} m^a G_m B^c <= (7/5) n^a L^c G_n` (`_sum_over_m`; `G_j/G_{j−1} >= 7/2` for
`j >= 4`, so `Σ_{m<=n} G_m <= G_n Σ_i (2/7)^i = (7/5) G_n`).  No term with `E^2`, `G^2`
or `E·G` occurs.

**Lemma 8.2 (heights).**  For `n >= 16`, `h_res(n) <= HR(L)` and `h_all(n) <= HA(L)` with

    HR = 8.3655 + 15.1253 L,   HA = 14.8815 + 28.81 L   (exact rationals in the code),

from `max_height(C) <= log2(C + 1) / log2(φ) <= 1.4405 log2(C + 1)` (`F(k) >= φ^{k-2}`),
the polynomial bounds on `C_res` and `A_max` (Lemma 8.1), every monomial
`<= n^{a_max} L^{b_max}` and `log2 L <= L/2` (`L >= 4`).

**Lemma 8.3 (majorant).**  `tail_majorant()` — the sum over `m` of the symbolic
`Λ_op`, plus `c_top` and the base case, with `h_r`, `h_a` replaced by `HR`, `HA` — is a
polynomial in `n`, `N`, `M`, `L` with nonnegative coefficients that is at least `U(n)`
for every `n >= 16`.  Its terms containing `M` have degree at most 11 in `n` (the
largest are about `40.4 n^11 L M`, `20.2 n^11 M`, `1006 n^10 L M` and `915 n^10 L^2 M`),
its terms containing `N` degree at most 9: `U(n) = O(n^11 log n · C(2n, n))`, i.e.
`O(n^10.5 log n · 4^n)`.

**Lemma 8.4.**  `C(2n, n) <= 4^n / sqrt(3n + 1)` for `n >= 1` (induction: equality at
`n = 1`, and `C(2n+2, n+1)/C(2n, n) = 2(2n+1)/(n+1)` with
`(2n+1)^2 (3n+4) <= 4 (n+1)^2 (3n+1)`, i.e. `19n <= 20n`).  Hence a monomial `c n^a M L^b`
divided by `n^11 4^n` is at most `c n^{a−11−1/2} L^b / sqrt(3)`, nonincreasing for
`n >= n_0` if `a <= 11` and `ln n_0 >= b/(11.5 − a)`; a monomial `c n^a N L^b` gives
`c n^{a−11} L^b / 2^n` and one without `N`, `M` gives `c n^{a−11} L^b / 4^n`, both
nonincreasing once `max(0, a − 11) + b <= 0.69 n_0`.  `ratio_bound(majorant, n0, 11)`
checks these conditions at `n0` and sums the values (with
`1/sqrt(3 n0) <= 1/floor(sqrt(3 n0))`).

**Theorem 8.5.**  `U(n) <= 392 · n^11 · 4^n` for every `n >= 1`, and
`U(n) <= k2 · n^11 · 4^n` with `k2 = U(4)/(4^11 · 4^4) = 161700700527/1073741824 ≈ 150.60`
for every `n >= 2`; neither constant can be lowered for `U`, and no constant works
with `n^10 4^n`.

*Proof.*  For `1 <= n <= 40` the ratios `U(n)/(n^11 4^n)` are computed exactly
(`check_operation_constant`): 392 at `n = 1`, 0.87 at `n = 2`, 0.02 at `n = 3`, 150.60
at `n = 4`, then 95.9, 80.2, 77.6, 79.4, 79.8, 79.8, 80.8 (`n = 11`), and decreasing
from there (69.5 at `n = 20`, 57.7 at `n = 30`, 48.9 at `n = 40`).  For `n >= 41`,
Lemma 8.4 applied to the majorant of Lemma 8.3 gives a ratio of at most
`92.74 < k2` (`tail_ratio(41)`).  Equalities at `n = 1` and `n = 4` show minimality.  By
Lemma 5.3, one Core call of `Main(n)` is charged `E(n−1, n) = n(n−2) C(2n−4, n−3)`
suffix positions, each at least `n` operations (`split_by_cut` creates value vectors
of `n` entries), and stage 1 makes `n Q^3 D_mid(n) >= c n^8 log n` Core calls; this part
of `U(n)` alone is at least `c' n^{10.5} log n · 4^n`, so `U(n)/(n^10 4^n)` is unbounded
(`test_exponent_eleven_is_needed` shows the growth for `n <= 22`).  ∎

## 9. The query bound

`bounds.query_bound(n) = n + 15 + Σ_{k=4}^{n} Λ_k` with
`Λ_k = core_queries(k) · K(k) + lemma46_budget(k) + k (S_k + E(k, k))` bounds the
Robertson–Webb queries of a run: the only query sites are `register_cake` (`n`
Evals), `split_by_cut` (one Cut and at most `k - 1` Evals for `k` active agents) and
the base cases (at most 2 and 15 queries).  A Core call makes `k - 1` prefix cuts, at
most `S_{k-1}` trims (Lemma 5.1) and at most `E(k−1, k)` fallback cuts (Lemma 5.3), and
its query count is also checked at run time; the Lemma 4.6 call makes at most
`H_46(k)` cuts (Lemma 6.1, and its budget is checked at run time); the class
SubCores make at most `Σ_E (S_{|E|} + E(|E|, |E|)) <= S_k + E(k, k)` trims and cuts
(`S` and `e ↦ E(e, e) = e (2e−3)!/((e−2)!)^2` are superadditive, the latter because
`E(e, e)/e` is nondecreasing); the counts of Core calls are those of Lemma 7.2, and
the sizes of the recursive calls are distinct (Lemma 7.4).

`T(n) <= n^11 4^n / 4` for all `n >= 1` (equality at `n = 1`), and
`T(n) <= C2 · n^11 4^n` for `n >= 2` with
`C2 = T(34)/(34^11 4^34) ≈ 0.0475560` (`bounds.MONOMIAL_C2`, attained at `n = 34`):
exactly for `n <= 399`, and by `ratio_bound(query_majorant(), 400, 11) ≈ 0.04474 < C2`
beyond (`check_query_constant`; the majorant uses `E(k−1, k) <= (3/35) k^2 G_k` and
`E(k, k) <= (1/7) k^2 G_k`).  The exponent 11 cannot be lowered: the stage-1 Core
calls alone give at least `c n^{10.5} ln(n) 4^n` queries in this accounting.

The Lemma 4.6 term is that of the rounding procedure of §6; the procedure of the
paper's proof (`n^2/γ` queries) is not used by the protocol and does not enter `T(n)`.

## 10. Verification aids (not part of the proof)

* `tests/test_opbounds.py` regenerates the instruction profiles, checks the
  constants, the soundness of the polynomial arithmetic on sample values, the counts
  of Lemma 5.3 against the code, and runs instances under an instruction-counting
  tracer (`tests/opmeter.py`) asserting that every measured call of the main
  functions stays below its bound, including a run in which SubCore falls back to
  the equilibrium computation (`test_fallback_runs_respect_the_bounds`; on that run
  the fallback executed 131731 instructions against its bound 889776, ratio 0.15).  Over
  the traced runs (Core sequences with 4-6 agents, complete runs with up to 5
  agents) the largest measured/bound ratios are about 0.75 (`value`), 0.67
  (`split_by_cut`), 0.31 (`subcore`), 0.14 (`core`) and 0.11 (`main`, for the
  small base-case calls; complete runs with `n = 4` stay below `2·10^-4` of `U(4)`).
