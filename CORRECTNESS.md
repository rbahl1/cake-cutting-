# Correctness of the protocol, from the code

This document proves from the code which properties of `Main` hold for every `n`,
shows where Aziz–Mackenzie's SubCore fails, and proves the fallback that replaces
it in that case.

* §1–§5: the recursion of stage 3, `X ⊕ Main(E*, R_2)`, is correct for every `n`.
  It has 4 or more agents only when `n ≥ 8`, which the tests do not run; the proof
  does not depend on `|E*|`.  Every call that returns returns a complete
  envy-free allocation.
* §6: every check of the protocol outside SubCore passes whenever every SubCore
  call returns.
* §7: Aziz–Mackenzie's SubCore does not always return; an explicit 5-agent input
  makes it raise.
* §8: when it raises, `subcore` computes its output from a competitive
  equilibrium.  For valuations with positive density this always succeeds
  (Theorem D), so every run returns a complete envy-free allocation.
* §9: what remains open for valuations with zero-density stretches.

**Scope.**  Runs of `run_protocol` with `check=True` (the default, and the scope of
`PROOF.md`), valuations that are nonnegative, additive and nonatomic (the
oracle's contract).  An allocation `A` of a piece `C` among agents `N` is
*complete* if its pieces are pairwise disjoint and their union is `C`, and
*envy-free* if `Δ_ij(A) = V_i(A_i) − V_i(A_j) ≥ 0` for all `i, j ∈ N`, with the
true valuations `V_i`.  Line references are to functions of `cake_cutting/`.

## Results

**Theorem A (no wrong output).**  For every call `main(N, C)` in a run: if the call
returns, the returned allocation is a complete envy-free allocation of `C` among `N`.

**Theorem B (the recursive call of stage 3).**  Let a call `main(N, C)` with
`n = |N| ≥ 4` reach the last line of `stage3`.  Then

1. `1 ≤ |E*| ≤ ⌊n/2⌋`, so the recursive call has fewer agents;
2. the check `X is not envy-free (Lemma 4.19)` and the check
   `assert_domination(..., "Proposition 4.20")` cannot fail: their conditions
   follow from checks that already passed in the same call (Lemmas 8 and 10);
3. every value that the recursive call needs from its caller's table is
   recorded: every component of `R_2` and every interval recorded so far is
   recorded for every agent of `E*`, as at the start of a run (Lemma 2);
4. if the recursive call returns, it returns a complete envy-free allocation of
   `R_2` among `E*` (Theorem A), and the caller then returns a complete
   envy-free allocation of `C` among `N` (Lemma 11).

**Theorem C (everything outside SubCore, §6).**  If every `subcore` call of a run
returns, and every result of the fallback among them has properties (P5) and (P6)
of §8, then no other check raises and the run returns a complete envy-free
allocation.

**Theorem D (§8).**  If every agent values every nonempty interval positively,
every `subcore` call returns a result with the properties used in §6.2 and §6.6.
With Theorem C: every run of `run_protocol` returns a complete envy-free allocation.

Theorem A does not say that a call returns.  A call raises `ProtocolError` when a
check fails.  By Theorem C only the checks inside SubCore can fail.  §7 gives an
input on which Aziz–Mackenzie's procedure fails; the fallback of §8 handles it, and
by Theorem D the fallback cannot fail when densities are positive.

## 1. The value table

**Lemma 1 (exact values).**  Every value recorded by the `ValueTable` equals the
true value, and `table.value(i, P)` equals `V_i(P)`.

*Proof.*  Values are recorded only by `ValueTable._set`, which ignores a key that
is already present, so a recorded value never changes.  `_set` is called by
`register_cake` and `register_interval` (an `Eval` answer), `split_by_cut` (for the
cutter, `amount` for `[c.a, y)` where `y` is the `Cut` answer, so
`V_cutter([c.a, y)) = amount`, and `total − amount` for `[y, c.b)`),
`_learn_split` (an `Eval` answer and a difference, or a difference),
`complement_prefix` (`V(outer) − V(suffix)`) and `record_union` (a sum).  Each
difference or sum combines recorded values of intervals `I_1, I_2` with
`I = I_1 ⊔ I_2`; by induction over the order of the `_set` calls and additivity,
every recorded value is exact.  `table.value` combines recorded component values
by sums (the subtree sums of `seq`, `prefix_sum`) and, for a suffix view, by
`total − prefix − V(component k_0) + V(head)`; each equals the sum of `V_i` over
the components of the piece, which are pairwise disjoint (every piece is built
from disjoint parts, §2–§3), so it equals `V_i(P)`.  Endpoints have value 0
because the valuations are nonatomic.  ∎

So every comparison of values in the code compares true values.

**Lemma 2 (active sets and recorded values).**

(a) `set_active` is called only at the start of `main`.  `main` is called by `run`
with all agents and by `main` itself in exactly two places,
`return Y.oplus(self.main(rest, U))` (stage 2) and
`return X.oplus(self.main(E_star, U))` (stage 3), with `rest, E* ⊊ N`.  After a
recursive call returns, its caller only evaluates `oplus` (`OrderedPiece.concat`,
which joins trees and copies value vectors with `vals_of` but neither reads a
value through `ValueTable.value` nor records one) and returns.  Hence, if
`N_0 ⊋ N_1 ⊋ ...` are the agent sets of the `Main` calls of a run in the order
they start, all table operations of call `t` happen while `N_t` is active, and
the active sets over time form this decreasing chain.

(b) Each function that records a value for an interval `I` records `I` for every
agent active at that moment before it returns: `register_cake` and
`register_interval` loop over `self.active` (the protocol never passes `agents`
to `register_interval`); `split_by_cut` records both halves for the cutter and,
through `_learn_split`, for every other active agent; `split_at`,
`complement_prefix` and `record_union` loop over `self.active`.

(c) Consequently, at every moment, an interval recorded for some agent is
recorded for every agent that is active then: it was recorded for every agent
active when it was first recorded (b), and the active set then contains the
current one (a).  Tree nodes copy the value vector of their interval
(`vals_of`) when they are built, and they are built only for intervals that are
already recorded: components of existing pieces, the intervals returned by the
functions in (b), the part `[c.a, y)` in `prefix_to` (called only by
`unallocated_part`, right after `complement_prefix` records it), and suffix heads
`[y, c.b)`, which are looked up in the table when evaluated and are recorded
intervals (the right half of a `split_by_cut`, or a start taken from
`_min_start`, which lists recorded intervals only).  So a copied vector has
entries for every agent active at any later moment.  ∎

For the recursive call `main(E*, R_2)` this gives Theorem B (3).  The order in (a)
is necessary: running `main` on a set `F` of agents and afterwards on a set
`F' ⊄ F` over the same table gives `F'` intervals recorded only for `F`, and the
second call fails with `KeyError`.  The protocol never does this.

## 2. Building blocks

**Lemma 3 (splits).**  For a piece `P` and `0 ≤ x ≤ V_a(P)`,
`prefix_of_value(P, a, x)` returns `(A, B, pos)` where `A` consists of the
components of `P` before `pos` and `B` of the rest (if `pos` lies inside a
component `c`, it is split by `split_by_cut` into `[c.a, y) ∈ A` and
`[y, c.b) ∈ B`), so `P = A ⊔ B`, and `V_a(A) = x`.  `suffix_of_value(P, a, x)`
calls `prefix_of_value(P, a, V_a(P) − x)`, so its suffix is worth `x`.  For a
suffix `σ` of `W`, `unallocated_part(W, σ)` is the prefix of `W` before `σ`
(`W` if `σ` is empty; the first `k` components if `σ` starts at a component
boundary; otherwise also `[c.a, σ.first.a)`), so `W = unallocated_part(W, σ) ⊔ σ`.

*Proof.*  Read off the code of `prefix_of_value` (`_locate` finds the first
boundary whose cumulative value is `x`, or the component that straddles `x`;
`split_by_cut` cuts it at value `x − cum`), `suffix_of_value` and
`unallocated_part`.  ∎

**Lemma 4 (Core).**  If `core(table, N, R, c)` returns `(X, R', _)`, then
`R = ⊔_{i∈N} X_i ⊔ R'`, `X` is envy-free among `N`, and
`V_c(R') ≤ ((n−2)/n) V_c(R)`.

*Proof.*  `equal_division` returns `R, ∅, ..., ∅` if `V_c(R) = 0`; otherwise it
takes `n − 1` prefixes of value `V_c(R)/n` and the rest.  By Lemma 3 the pieces
`P_0, ..., P_{n−1}` partition `R` and `V_c(P_u) = V_c(R)/n` for all `u` (the last
one is worth `V_c(R) − (n−1)V_c(R)/n`).  SubCore runs on the `n − 1` other
agents with minimum 0; `SubCoreCall.run` always ends with `verify`, and a
returning call gives each `j ≠ c` a pair `(u_j, σ_j)` with distinct `u_j` (else
"assigned one piece to two agents"), `σ_j` a suffix of `P_{u_j}` (`_is_suffix`),
`V_j(σ_j) ≥ V_j(σ_k)` for all `k ≠ c`, and `V_j(σ_j) ≥ V_j(P_u)` for every `u`
that no agent holds.  The cutter receives `P_{u_c}`, `u_c = free[0]`; exactly one
index is free because `n − 1` agents hold distinct indices.  Envy-freeness: an
agent `j ≠ c` does not envy `k ≠ c` by `verify`, nor `c` because `P_{u_c}` is not
held; `c` has `V_c(X_c) = V_c(R)/n ≥ V_c(σ_k)` because `σ_k ⊆ P_{u_k}`.
Partition: `R'` concatenates `unallocated_part(P_{u_j}, σ_j)` over the held
pieces, so by Lemma 3 `R = ⊔_j σ_j ⊔ P_{u_c} ⊔ R'`.  The factor is checked at
the end of `core` ("the cutter's residue did not shrink"), so a returning call
satisfies it.  ∎

**Lemma 5 (the Lemma 4.6 procedure).**  `simultaneous_cut(table, N, R*, requests, δ)`
returns a piece `cut[key]` for every key of `requests` and a piece `leftover` with
`R* = ⊔_key cut[key] ⊔ leftover`.

*Proof.*  The protocol does not set `force_fallback`, so `run` calls `_rounding`.
If no request exceeds `δ` or no agent values `R*` positively, every piece is
empty and `leftover = R*`.  Otherwise the single initial cell is `R*`.
`_assign` distributes a list of cells along the label tree by `_split`, which
places every cell in exactly one of `side1`, `side2` and `F`: `_reduce` moves a
cell to `side1` (weight 1) or `side2` (weight 0) or keeps it in `F`; `_halve`
replaces a cell of `F` by the two parts of a `prefix_of_value` split (Lemma 3);
at the end every cell of `F` goes to one side.  By induction over the label
tree, the cell lists of the leaves partition `R*`.  The piece of a request is
the concatenation of its leaf's cells (empty if `λ ≤ δ`), and `leftover` that of
the label `None`.  ∎

**Lemma 6 (sums of allocations).**  If `A^1, ..., A^r` are allocations among the
same agents whose pieces are pairwise disjoint, then
`Δ_ij(A^1 ⊕ ... ⊕ A^r) = Σ_t Δ_ij(A^t)`.  So a sum of envy-free allocations is
envy-free, and `Δ_ij(A ⊕ B) ≥ Δ_ij(A)` if `B` is envy-free.

*Proof.*  Additivity of `V_i`.  ∎

## 3. The allocation `X` of stage 3

Fix a call `main(N, C)` that reaches the last line of `stage3`; `n = |N|`,
`Q = n(n−1)`.  Notation, with the code's names:

* `H, S, R*`: the output of `preprocess`.  A snapshot `s ∈ S` is the allocation
  of one Core call; `s_k = s.pieces[k]`, `b_s(i) = V_i(s_i)`,
  `d_{s,i,k} = V_i(s_i) − V_i(s_k)`.
* `VR_i = V_i(R*)`, `η_i = VR_i / (4n(Q+1)Q^2)` (`eta` in `check_lemma_4_14`),
  `Γ_i = VR_i / Q^2`, `ε_i = eps[i] = VR_i / (4n^2(Q+1)Q^2)`, `L = 1/(nQ^2)`.
  Then `L·VR_i = 4(Q+1)η_i` and `Γ_i = 4n(Q+1)η_i`.
* `P`: the pruned snapshots of the cutter `c`; `marks_s(i) = {k : d_{s,i,k} > ε_i}`;
  the profile `F_i = (marks_s(i))_{s∈P}`; `F_i ⪯ F_h` iff
  `marks_s(i) ⊆ marks_s(h)` for all `s ∈ P` (`preceq`); the profile classes; `ℓ_i`,
  the rank of the class of `i`.
* `J` (`|J| = Q`, checked) and the maps `μ_s` (`mu[s.sid]`), `s ∈ J`.
* `A(s, k)`, `P_h` (empty if `ℓ_h = 0`) and `leftover`: the output of
  `simultaneous_cut`.
* `D_h = ⊔_{s∈J} s_{μ_s(h)}`, `A_h = ⊔_{s∈J} A(s, μ_s(h))`,
  `W_h = [P_h, A_h, D_h]`, `B_i = Σ_{s∈J} b_s(i)`,
  `q_i = B_i + max(0, L(ℓ_i − 1/2)) VR_i`.
* `Z_i = σ_i`, the suffix that agent `i` receives in the SubCore of its class;
  `Y_h = unallocated_part(W_h, σ)`; `R_1 = [leftover, Y_h for h ∈ N]`;
  `X = H ⊕ Id(S∖J) ⊕ Z`.
* `X'` and `R_2`: `X` and the residue after the prescribed sequences of
  stage 3 (`X` and `U` in the code at the recursive call).

**Lemma 7 (partition).**
(a) `C = ⊔ H ⊔ ⊔_{s∈S, k∈N} s_k ⊔ R*`.
(b) Each `μ_s` is a permutation of `N`.
(c) In each class `E`, SubCore gives the agents of `E` the pieces `W_h`, `h ∈ E`,
by a bijection `E → E`; `σ_i` is a suffix of `W_{h(i)}` and
`W_h = Y_h ⊔ σ_{i(h)}`.
(d) `C = ⊔_{i∈N} X_i ⊔ R_1` and `R_1 ⊆ R*`.
(e) `C = ⊔_{i∈N} X'_i ⊔ R_2` and `R_2 ⊆ R_1`.

*Proof.*  (a) `preprocess` applies `core` to successive residues starting from
`C` (snapshot calls, then the `shrink` calls that build `H`); apply Lemma 4 to
each.  (b) `μ_s` is the identity or the result of `matching_through`, which
raises unless `sorted(matching.values()) == sorted(N)`, and its keys are `N`.
(c) `stage3` raises if SubCore returns an index `≥ |E|`; SubCore raises if a piece
has two holders; `verify` checks the suffix property.  `|E|` agents with distinct
indices in `{0, ..., |E|−1}` form a bijection.  `W_h = Y_h ⊔ σ` by Lemma 3.
(d) The keys of `requests` are `("a", s, k)` for `s ∈ J`, `k ∈ N`, and `("p", h)`
for `ℓ_h > 0`, and `stage3` uses each key exactly once.  By Lemma 5,
`R* = ⊔_{s,k} A(s,k) ⊔ ⊔_h P_h ⊔ leftover`.  By (b),
`⊔_h D_h = ⊔_{s∈J} ⊔_k s_k` and `⊔_h A_h = ⊔_{s,k} A(s,k)`; by (c),
`⊔_h W_h = ⊔_i Z_i ⊔ ⊔_h Y_h`.  `used` is the set of identifiers of `J`.
Therefore
`⊔ X_i ⊔ R_1 = ⊔H ⊔ (S∖J) ⊔ ⊔_h W_h ⊔ leftover = ⊔H ⊔ (S∖J) ⊔ (S∩J) ⊔ R* = C`
by (a).  For `R_1 ⊆ R*`: the check "a trim entered D_h" passes only if `D_h` is
empty, or `σ` has more components than `D_h`, or the same number and the same
first component.  Since `σ` is a suffix of `W_h` and `D_h` is the last part of
`W_h`, in each case `σ ⊇ D_h`, so `Y_h ⊆ P_h ⊔ A_h ⊆ R*`.  (e) `shrink` applies
`core` to successive residues starting from `R_1` (Lemma 4).  ∎

**Lemma 8 (Lemma 4.19 for the code).**  `X` is envy-free, and `Δ_ij(X) ≥ η_i`
whenever `i` and `j` are in different classes.

*Proof.*  The facts used, each with its source in the code:

* (F1) Every snapshot is envy-free (Lemma 4), so
  `V_i(s_k) = b_s(i) − d_{s,i,k} ≤ b_s(i)`.
* (F2) For every `s ∈ S`: `d_{s,i,k} > ε_i` implies `d_{s,i,k} ≥ Γ_i`.  The loop of
  `preprocess` ends only when, for every `i` with `V_i(U) > 0` (and `U = R*` at
  the end), the smallest positive bonus above `eps_i = Gamma/(4n(Q+1))/n = ε_i`
  (found by `bisect_right` in the list of all positive `d_{s,i,k}`, which the loop
  does not change) is not below `Gamma = Γ_i`.  If `VR_i = 0` then
  `ε_i = Γ_i = 0` and (F2) is immediate.
* (F3) `V_i(A(s,k)) ≤ η_i` for `s ∈ J`, `k ∈ N` (`check_lemma_4_14`, (ii));
  `V_i(P_h) ≤ L ℓ_h VR_i` (`check_lemma_4_14`, (iii)); `0 ≤ ℓ_h ≤ n−1` (checked
  after `ranks`).
* (F4) If `F_i ⪯ F_j` and `F_i ≠ F_j`, then `ℓ_i ≥ ℓ_j + 1`: in `ranks`,
  `longest(a) ≥ 1 + longest(b)` whenever `strictly_below(a, b)`.
* (F5) If `F_i ⋠ F_h`, then `J` contains a snapshot `s` with
  `μ_s(h) ∈ marks_s(i)`.  The witness loop visits every pair `(i, h)` with
  `not preceq(i, h)` (then `i ≠ h`), picks an unused `s` with
  `marks_s(i) ⊄ marks_s(h)` (it raises if there is none), sets
  `k = min(marks_s(i) − marks_s(h))` and `μ_s = matching_through(N, marks_s, h, k)`,
  which maps `h` to `k`, and adds `s` to `used` and `J`; `μ_s` is not changed
  afterwards.
* (F6) `σ_i` is a suffix of `W_{h(i)}` with `h(i)` in the class of `i`,
  `V_i(σ_i) ≥ q_i`, and `V_i(σ_i) ≥ V_i(σ_k)` for `k` in the class of `i`
  (`verify` with the minima `q`, and the index check of `stage3`).

Let `i` and `j` be in different classes, and `h = h(j)`; then `F_h = F_j` and
`ℓ_h = ℓ_j`.  By Lemma 7 the parts of `W_h` are disjoint, so
`V_i(Z_j) ≤ V_i(W_h) = V_i(P_h) + V_i(A_h) + V_i(D_h)`, with
`V_i(A_h) = Σ_{s∈J} V_i(A(s, μ_s(h))) ≤ Qη_i` (F3) and `V_i(D_h) ≤ B_i` (F1).

*Case `F_i ⪯ F_j`.*  Then `ℓ_i ≥ ℓ_j + 1 ≥ 1` (F4), so
`q_i = B_i + L(ℓ_i − 1/2) VR_i`, and

    Δ_ij(Z) ≥ q_i − (L ℓ_j VR_i + Qη_i + B_i) = L(ℓ_i − ℓ_j − 1/2) VR_i − Qη_i
            ≥ (L/2) VR_i − Qη_i = 2(Q+1)η_i − Qη_i = (Q+2)η_i ≥ η_i.

*Case `F_i ⋠ F_j`.*  By (F5) there is `s* ∈ J` with `k* = μ_{s*}(h) ∈ marks_{s*}(i)`,
so `d_{s*,i,k*} > ε_i` and, by (F2), `d_{s*,i,k*} ≥ Γ_i`.  With (F1) for the other
snapshots, `V_i(D_h) ≤ B_i − Γ_i`.  Also `q_i ≥ B_i` and `V_i(P_h) ≤ L(n−1) VR_i`, so

    Δ_ij(Z) ≥ B_i − (L(n−1) VR_i + Qη_i + B_i − Γ_i) = Γ_i − Qη_i − 4(n−1)(Q+1)η_i
            = (4n(Q+1) − Q − 4(n−1)(Q+1)) η_i = (3Q+4)η_i ≥ η_i.

Within a class `Δ_ij(Z) ≥ 0` by (F6).  So `Z` is envy-free with `Δ_ij(Z) ≥ η_i`
across classes.  `H` is a sum of Core allocations and `Id(S∖J)` a sum of
snapshots, all envy-free among `N` (Lemma 4), and all pieces are disjoint
(Lemma 7); by Lemma 6, `Δ_ij(X) ≥ Δ_ij(Z)`.  ∎

## 4. The recursive call

**Lemma 9 (the prescribed sequences).**  `V_i(R_2) ≤ η_i` for every `i ∉ E*`.

*Proof.*  If `V_i(R_1) = 0`, stage 3 skips `i`, and `V_i(R_2) = 0` because
`R_2 ⊆ R_1`.  Otherwise it runs `shrink(N, U, X, i, D)` with
`D = D_final(n) = ceil_log(n, β)`, `β = 4n(Q+1)Q^2`; `ceil_log` returns the least `d`
with `(n/(n−2))^d ≥ β` (here `n ≥ 4` and `β > 1`).  `shrink` stops when
`V_i(U) = 0`, or after `D` Core calls with cutter `i`, each of which returns
`V_i(U_new) ≤ ((n−2)/n) V_i(U_old)` (Lemma 4).  If `U_i ⊆ R_1` is the residue when
the sequence of `i` starts, then afterwards
`V_i(U) ≤ ((n−2)/n)^D V_i(U_i) ≤ V_i(R_1)/β`, and the later sequences only shrink
the residue.  Since `R_1 ⊆ R*` (Lemma 7),
`V_i(R_2) ≤ V_i(R_1)/β ≤ VR_i/β = η_i`.  ∎

**Lemma 10 (domination).**  `X'` is envy-free, and `Δ_ij(X') ≥ V_i(R_2)` for all
`i ∉ E*`, `j ∈ E*`.  Hence the `X` check before the sequences and
`assert_domination(X', R_2, N∖E*, E*)` do not raise (they compare exactly these
quantities, Lemma 1).

*Proof.*  `X'` is `X` plus Core allocations among `N` on disjoint pieces
(Lemma 7 (e)), all envy-free (Lemma 4); by Lemma 6, `X'` is envy-free and
`Δ_ij(X') ≥ Δ_ij(X)`.  `E*` is a class, so `i ∉ E*` and `j ∈ E*` are in different
classes, and `Δ_ij(X) ≥ η_i ≥ V_i(R_2)` by Lemmas 8 and 9.  ∎

**Lemma 11 (combination; Corollary 4.2 of the paper).**  Let `X'` be an envy-free
allocation among `N`, `R_2` disjoint from its pieces, `E ⊆ N` with
`Δ_ij(X') ≥ V_i(R_2)` for all `i ∉ E`, `j ∈ E`, and `Y` a complete envy-free
allocation of `R_2` among `E`.  Then `X' ⊕ Y` is envy-free among `N`, and its
pieces partition `⊔_i X'_i ⊔ R_2`.

*Proof.*  `(X' ⊕ Y)_i = X'_i ⊔ Y_i` for `i ∈ E` and `X'_i` otherwise.
`i, j ∈ E`: `V_i(X'_i) + V_i(Y_i) ≥ V_i(X'_j) + V_i(Y_j)`.  `i ∈ E`, `j ∉ E`:
`V_i(X'_i) + V_i(Y_i) ≥ V_i(X'_i) ≥ V_i(X'_j)`.  `i ∉ E`, `j ∈ E`:
`V_i(X'_j) + V_i(Y_j) ≤ V_i(X'_j) + V_i(R_2) ≤ V_i(X'_i)`, because `Y_j ⊆ R_2`.
`i, j ∉ E`: unchanged.  The partition follows from that of `Y`.  ∎

**Proof of Theorem B.**  (1) `stage3` raises unless there are at least two
classes; every agent is placed in exactly one class; `E*` is a smallest one, so
`1 ≤ |E*| ≤ n/2`.  (2) Lemmas 8 and 10.  (3) Lemma 2.  (4) Theorem A for the
recursive call (it has fewer agents, §5), then Lemma 11 with Lemma 10, and
`⊔_i (X'_i ⊔ Y_i) = ⊔_i X'_i ⊔ R_2 = C` by Lemma 7 (e).  ∎

## 5. Proof of Theorem A

By strong induction on `n = |N|`; let `main(N, C)` return.  The induction
hypothesis is about every call with fewer agents, whatever its piece and the
state of the table, so it applies to the recursive calls, which run on residues
inside their caller's table.

* `n ≤ 3` (`base_case`): one agent receives `C`; `cut_and_choose` gives the two
  parts of `_equal_pieces`, a partition by Lemma 3; `selfridge_conway` gives the
  three pieces of `_equal_pieces`, the trimmed piece and its trimmings
  (`suffix_of_value`, Lemma 3) and the three parts of the trimmings to three
  distinct agents, a partition of `C`.  `base_case` raises unless the result is
  envy-free (Lemma 1: exact comparison).
* Stage 2, nobody values `R*`: the result is `H ⊕ Id(S)` plus `R*` for `N[0]`.
  It partitions `C` (Lemma 7 (a)); `H ⊕ Id(S)` is a sum of envy-free Core
  allocations (Lemmas 4, 6), and adding `R*`, which every agent values at 0
  (the condition of the branch), changes no value.
* Stage 2, shortcut: `sink_component` returns a sink component only if the
  graph has at least two components, so `rest = N ∖ sink` is nonempty and
  smaller than `N`.  `Y` is `H ⊕ Id(S)` plus the Core allocations of the
  sequences: envy-free (Lemmas 4, 6), and with the residue `U` it partitions `C`
  (Lemma 7 (a), then Lemma 4 for each call of the sequences).  `assert_domination(Y, U, sink, rest)` passed.  By induction
  `main(rest, U)` returned a complete envy-free allocation of `U`; Lemma 11
  with `E = rest` gives the claim.
* Stage 3: Theorem B (4), using the induction hypothesis for `main(E*, R_2)`.  ∎

## 6. The checks before the recursion

**Theorem C (everything outside SubCore).**  In a run with `check=True`: if every
call of `subcore` made during the run returns (raises nothing), and every result
computed by the equilibrium fallback (§8) has properties (P5) and (P6), then no
other check of the protocol raises, and every call of `Main` returns.  Hence, by
Theorem A, the run returns a complete envy-free allocation.  (Results of
Aziz–Mackenzie's procedure have the corresponding properties by §6.2 and §6.6;
fallback results have them when densities are positive, Theorem D.)

The proof is §6.1–§6.7.  Every loop of the protocol is bounded (the `while` loops
of SubCore add a winner per iteration or raise; stage 1 by §6.3; pruning removes a
snapshot per pass; the Lemma 4.6 procedure by `PROOF.md` Lemma 6.1; `Main` and
SubCore recurse on fewer agents), so a call that raises nothing returns.

### 6.1 Base cases (`base_cases.py`)

`cut_and_choose`: `a` cuts two halves of equal value for `a` (Lemma 3), `b` takes
one of largest value; `a` is indifferent.  `selfridge_conway` (agents `a1 < a2 < a3`):
`a1` cuts three pieces worth `V_1(C)/3` each.  If `a2`'s largest value exceeds her
second largest, she trims her favourite piece to her second-largest value
(`suffix_of_value`), so afterwards she values at least two of the three pieces at
her maximum.  `a3` takes a piece of largest value; `a2` takes the trimmed piece if
`a3` left it (it has her maximum value), otherwise a piece of largest value among
the two left, one of which has her maximum; `a1` gets the last piece, which is never
the trimmed one, so she values it at `V_1(C)/3` and every piece at most that.  So
nobody envies after the first round.  If the trimmings are nonempty, the
non-`a1` agent who did not get the trimmed piece (the *divider*) cuts them into
three parts she values equally; the holder of the trimmed piece chooses first,
then `a1`, then the divider.  The holder and the divider envy nobody in either
round; `a1` does not envy the divider (she chooses before the divider), and she does not
envy the holder because the holder's two parts together lie in the original
favourite piece, which `a1` values at `V_1(C)/3`.  All parts partition `C`.  So the
check at the end of `base_case` passes.

### 6.2 Core, given that its SubCore call returns

Let `core(table, N, R, c)` run its SubCore call to the end.  At the last step of
SubCore, either the newcomer takes an unallocated piece whole, or the loser of the
contest takes an uncontested piece whole; that holding is not changed afterwards.
So a non-cutter receives a whole piece `P_u` (for a result of the fallback this is
(P5)), and the cutter receives the whole piece `P_{u_c}`; both are worth `V_c(R)/n`
to `c` (Lemma 4), so
`V_c(R') ≤ V_c(R) − 2V_c(R)/n`: the factor check of `core` passes.  The queries of
the call are the `n − 1` prefix cuts of `equal_division` and the trims of SubCore,
each at most one `split_by_cut` (one Cut and at most `n − 1` Evals); hint
placements, `normalize_left`, `unallocated_part` and `suffix_from` make no queries.
SubCore on `n − 1` agents places at most `S_{n−1}` trims (`PROOF.md` Lemma 5.1,
whose proof uses only that each `while` iteration adds a winner; a failed attempt
places fewer), and the fallback, if it runs, at most `E(n−1, n)` (§8.5).  So the
query check of `MainProtocol._core` (`used ≤ n((n−1) + S_{n−1} + E(n−1, n))`,
`bounds.core_queries`) passes.

### 6.3 Stage 1: at most `nQ^3` prescribed sequences (Lemma 4.7)

For an agent `i`, `positive[i]` is the set of positive bonuses `d_{s,i,k}`
(`s ∈ S`, `k ≠ i`), so `|positive[i]| ≤ |S|(n−1) = nQ^2(n−1) = Q^3`.  An iteration of
the loop chooses an agent `i` with `V_i(U) > 0` and a bonus `d* ∈ positive[i]` with
`ε_i < d* < Γ_i` (`ε_i, Γ_i` computed from the current `U`), and runs `shrink` with
cutter `i` and `D_mid(n) = ceil_log(n, 4n^2(Q+1))` calls.  Afterwards either
`V_i(U') = 0`, or `V_i(U') ≤ ((n−2)/n)^{D_mid} V_i(U) ≤ V_i(U)/(4n^2(Q+1))` (§6.2), so
`Γ'_i = V_i(U')/Q^2 ≤ Γ_i/(4n^2(Q+1)) = ε_i < d*`.  The residue only shrinks, so
`Γ_i` never increases and `d* ≥ Γ_i` from then on: the pair `(i, d*)` is never
chosen again.  Hence the loop makes at most `Σ_i |positive[i]| ≤ nQ^3` iterations,
and the check `steps > nQ^3` never fires.  The envy-freeness check of the snapshots
holds by Lemma 4.

### 6.4 Stage 2 (Lemmas 4.9 and 4.10)

*Pruning leaves at least `Q` snapshots.*  `P` starts with the `Q^2` snapshots of the
cutter `c` (the assertion `len(P) == Q*Q`).  A removal step takes an ordered pair
`(i, h)`, `i ≠ h`, with between 1 and `Q − 1` witnesses and removes all of them;
removing snapshots never creates a witness, so that pair never triggers again.
There are `n(n−1) = Q` ordered pairs, so at most `Q(Q−1)` snapshots are removed and
`|P| ≥ Q^2 − Q(Q−1) = Q`.

*The shortcut (Lemma 4.10).*  Let `G_s` (`s ∈ P`) have a sink component `K`; then
`rest = N∖K` is nonempty, because `sink_component` returns a component only if
there are at least two.  For `i ∈ K` and `j ∉ K` there is no edge `i → j`, i.e.
`j ∈ marks_s(i)`, so `d_{s,i,j} > ε_i` and, by (F2) of Lemma 8 (which only uses the
end of `preprocess`), `d_{s,i,j} ≥ Γ_i = V_i(R*)/Q^2`.  `Y = H ⊕ Id(S) ⊕` (Core
allocations) is a sum of envy-free allocations containing `s`, so by Lemma 6
`Δ_ij(Y) ≥ d_{s,i,j} ≥ V_i(R*)/Q^2`.  The sequence of `i` (if `V_i(R*) > 0`) has
`D_shortcut(n) = ceil_log(n, Q^2)` calls, so the final residue satisfies
`V_i(U) ≤ V_i(R*)/Q^2` (as in Lemma 9); if `V_i(R*) = 0` then `V_i(U) = 0`.  So
`assert_domination(Y, U, K, rest)` passes.

### 6.5 Stage 3 before SubCore

*At least two classes (Lemma 4.9).*  Every `s ∈ P` is a snapshot of the cutter `c`,
chosen with `V_c(R*) > 0`, made by a Core call on a residue `R_s ⊇ R*` that leaves
a residue `R'_s ⊇ R*`.  By Lemma 4, `b_s(c) = V_c(R_s)/n` and
`R_s = ⊔_k s_k ⊔ R'_s`, so `V_c(R'_s) = V_c(R_s) − Σ_k V_c(s_k) = Σ_k d_{s,c,k}`.  If
every `d_{s,c,k} ≤ ε_c`, then `V_c(R*) ≤ V_c(R'_s) ≤ (n−1)ε_c < V_c(R*)`, which is
impossible; so `marks_s(c) ≠ ∅`.  If all agents had the same profile, then for every
`s ∈ P` all `marks_s(i)` would be one set `M_s`, and `k ∈ M_s` would give
`k ∈ marks_s(k)`, i.e. `d_{s,k,k} = 0 > ε_k`, which is false; so `M_s = ∅`,
contradicting `marks_s(c) ≠ ∅`.

*Witnesses and `|J| = Q` (Lemma 4.12).*  After pruning, every ordered pair has no
witness or at least `Q` witnesses in `P`.  The witness loop treats each ordered pair
`(i, h)` with `F_i ⋠ F_h` (at least one witness, hence at least `Q`) and uses one
snapshot per pair; at most `Q − 1` were used before, so an unused witness exists.
After the loop `|J| ≤ Q`, and since `|P| ≥ Q` the filling loop reaches `|J| = Q`.

*Matchings (Lemma 4.11).*  Stage 3 runs only if no `G_s`, `s ∈ P`, has a sink
component, i.e. every `G_s` is strongly connected.  So for `k ≠ h` the breadth-first
search of `matching_through` (edges `u → v`, `v ∉ marks_s(u)`, `v ≠ u`) reaches `h`
from `k`, and the returned map is the cyclic shift along the path `k → ... → h`
closed by `h → k`, the identity elsewhere: a permutation (its check passes).  Every
agent `u` is mapped to `μ_s(u) ∉ marks_s(u)`: along path edges by definition, `h → k`
because `k ∉ marks_s(h)`, and `u → u` because `d_{s,u,u} = 0 ≤ ε_u`.

*Ranks (Lemma 4.13).*  A chain of strictly increasing profiles has at most
`#classes − 1 ≤ n − 1` steps, so `0 ≤ ℓ ≤ n − 1`, and `Σ_h ℓ_h ≤ n(n−1) = Q`.

*Requested fractions (Lemma 4.14).*  With `e = 1/(4n^2(Q+1)Q^2) = L/(4n(Q+1))` and
`δ = L/(16n(Q+1))`: every addition requests `r_{s,k} + δ ≤ e + δ` (the maximum in
`r_{s,k}` is over `d/VR_i` with `d ≤ ε_i`, i.e. `d/VR_i ≤ e`).  The sum of all
requests is at most `nQ(e + δ) + LQ + nδ = L(Q + (5Q+1)/(16(Q+1))) < L(Q+1) < 1`.
So `simultaneous_cut`'s preconditions hold (`0 < δ < 1`, requests `≥ 0`, sum `≤ 1`).

*The Lemma 4.6 procedure.*  Let `A+` be the `r` agents with `V_i(R*) > 0`.  At the
root the cells are `R*` with weight `p`; kernel steps keep the `r` sums
`Σ_K y_K V_i(K)` exactly (the step is along a kernel vector of the `r × (r+1)`
value matrix, and `t` is the distance to the nearest boundary, so weights stay in
`[0, 1]` and at least one becomes 0 or 1); halving keeps the sums (both halves keep
the weight); at the end at most `r` fractional cells remain, all fine
(`V_i(K) ≤ γV_i(R*)`), and rounding each to the nearer of 0, 1 moves each sum by at
most `(r/2)γV_i(R*)`.  A node's cells are worth `λ_ν V_i(R*) + e_ν`; its left child
receives `p(λ_ν V_i(R*) + e_ν) ± (r/2)γV_i(R*)` with `p = λ_left/λ_ν ≤ 1`, so errors
grow by at most `(r/2)γV_i(R*)` per level and a leaf at depth `≤ D` has error at
most `D(r/2)γV_i(R*) = δV_i(R*)` (`γ = 2δ/(rD)`).  Requests with `λ ≤ δ` get the
empty piece, within `δV_i(R*)` of `λV_i(R*)`.  Agents outside `A+` value every piece
at 0.  So `verify` passes.  The cut count is at most
`K r^2 ceil(log2(rD/(2δ)))` with `K ≤` the number of requests (`PROOF.md`
Lemma 6.1), and `r ≤ n` and the depth `≤ bit_length(#requests)`, so the budget
`n · lemma46_cuts(n, #requests, δ)` is not exceeded.

*`check_lemma_4_14` (Lemma 4.14).*  For `VR_i > 0`: if `d_{s,i,k} ≤ ε_i`, the
request `λ = r_{s,k} + δ ≥ d_{s,i,k}/VR_i + δ` gives `V_i(A(s,k)) ≥ (λ − δ)VR_i ≥ d_{s,i,k}`;
always `V_i(A(s,k)) ≤ (e + 2δ)VR_i = 3L·VR_i/(8n(Q+1)) ≤ L·VR_i/(4(Q+1)) = η_i`
(as `n ≥ 2`); and for `ℓ_h > 0`,
`L(ℓ_h − 1/2)VR_i ≤ V_i(P_h) ≤ (L(ℓ_h − 1/2) + 2δ)VR_i ≤ Lℓ_h VR_i` (as `2δ ≤ L/2`).
For `VR_i = 0` all values are 0 and `d ≤ ε_i = 0`.  For `ℓ_h = 0`, `P_h = ∅` and the
bounds are `0 ≤ 0 ≤ 0`.

*`V_i(W_h) ≥ q_i` (Lemma 4.17 (ii)).*  Let `i, h` be in one class `E`.  For `s ∈ J`,
`μ_s(h) ∉ marks_s(h) = marks_s(i)`, so `d_{s,i,μ_s(h)} ≤ ε_i` and
`V_i(A(s, μ_s(h))) ≥ d_{s,i,μ_s(h)}`.  Hence
`V_i(A_h) + V_i(D_h) ≥ Σ_s d_{s,i,μ_s(h)} + Σ_s (b_s(i) − d_{s,i,μ_s(h)}) = B_i`, and
`V_i(P_h) ≥ max(0, L(ℓ_h − 1/2))VR_i` with `ℓ_h = ℓ_i`.  So `V_i(W_h) ≥ q_i`.

### 6.6 The SubCore call of each class, given that it returns

The two properties below are proved here for a result of Aziz–Mackenzie's
procedure; for a result of the fallback they are (P6) (§8.4).

*The empty piece is not allocated (Proposition 4.5).*  The direct call has hints
`None`.  The empty piece has the last index and value 0.  At a step that takes a
favourite, at most `|E| − 1` of the pieces `W_h` are held, so some `W_h` is
unallocated, has value `≥ 0`, and precedes the empty piece among the candidates of
largest value whenever the empty piece is one.  The loser of a contest chooses
among the uncontested pieces, which include a `W_h` (at most `|E| − 1` pieces are
contested), by largest value and then lowest index, so again not the empty piece.
The empty piece is therefore never held, hence never contested, hence never part of
a recursive call.

*Trims never enter `D_h` (Lemma 4.17 (iv)).*  Every agent `j` of `E` has
`V_j(D_h) = Σ_{s∈J} V_j(s_{μ_s(h)}) ≤ B_j ≤ q_j` (F1), and every level `β` used for
`j` anywhere in the call tree is at least `q_j` (levels are `max(b_j, ·)` and
benchmarks are inherited from the parent's levels, starting from `q_j`).  By
induction over the order in which positions are produced: every piece of the call
tree that is a suffix of `W_h` contains `D_h`, and every position produced on such
a piece `P` lies at or before the start of `D_h`.  A trim of `j` at level
`β ≤ V_j(P)` is `position_of_value(P, j, V_j(P) − β)`, the first position where the
cumulative value reaches `V_j(P) − β ≤ V_j(P) − V_j(D_h)`, which is the cumulative
value at the start of `D_h`; so it is at or before that start, and `normalize_left`
only moves positions to the left.  Hint placements are starts of earlier holdings
or hint suffixes, margins are trim positions, and allocated suffixes start at such
positions or are whole pieces.  So every allocated suffix of `W_h` contains `D_h`
and the check of `stage3` passes.

### 6.7 Preconditions of the table functions

`split_by_cut` is called by `prefix_of_value` / `position_of_value` only in the
"inside" case of `_locate`, where `cum < amount < cum + V(c)`, so its amount lies
strictly between 0 and `V(c)`; then `V([c.a, y)) = amount > 0` and
`V([y, c.b)) > 0` give `c.a < y < c.b`.  `prefix_of_value`, `suffix_of_value` and
`suffix_position_of_value` are called with amounts in `[0, V]`: equal division
(`share ≤ V`), trims (`β ≤ V_j(P)` by the trim condition), halving (half of a
positive value).  Positions passed to `suffix_from` were produced by the table on the
same piece.  Missing values cannot occur (Lemma 2).

## 7. SubCore

Theorem C reduces the correctness of the protocol to one statement: **every
`subcore` call returns.**  This is the SubCore guarantee of Aziz–Mackenzie (proof of
Lemma 4.5 in arXiv:1604.03655v12, cited by Ye–Bai as the proof of Lemma 4.13 of the
extended version, Appendix D), which Ye–Bai use for Propositions 4.3 (i) and 4.5.
The step of that argument that fails below is Claim 8: in the `while` loop, some
unallocated contested piece always carries a trim of a non-winner.  The statement
is false for the SubCore of this repository.

**Counterexample.**  Five agents with piecewise-constant densities
(`tests/test_main.py`, `SUBCORE_COUNTEREXAMPLE`):

| agent | breakpoints | densities |
|---|---|---|
| 0 | 0, 1/3, 1/2, 5/6, 1 | 1, 1, 0, 2 |
| 1 | 0, 1/6, 1/2, 2/3, 1 | 2, 2, 0, 1 |
| 2 | 0, 1/8, 13/48, 17/48, 7/16, 13/24, 9/16, 3/4, 5/6, 1 | 2, 0, 0, 0, 2, 0, 0, 3, 0 |
| 3 | 0, 1/24, 1/6, 3/16, 5/12, 23/48, 25/48, 7/12, 11/16, 37/48, 47/48, 1 | 1, 0, 2, 0, 3, 1, 2, 1, 2, 3, 0 |
| 4 | 0, 1/8, 1/4, 1/2, 3/4, 1 | 0, 1, 1, 1, 1 |

With `cut_rule="right"`, `run_protocol` raises `ProtocolError` in the first Core
call of `Main` (cutter 0, after 58 queries).  Cutter 0 divides the cake into
`[0, 1/6), [1/6, 1/3), [1/3, 5/6), [5/6, 11/12), [11/12, 1)`; SubCore runs on agents
1–4.  At its last step (newcomer 4) agent 2 values both uncontested pieces at 0, so
her level is 0: she trims every contested piece she values positively to the empty
suffix at its end and becomes the rightmost trimmer of the pieces `[0, 1/6)` and `[1/3, 5/6)`.  The winners
are agents 2 and 3; the recursive call on them gives agent 3 a suffix of the
margin of `[1/3, 5/6)` and agent 2 the piece `[0, 1/6)`, and the remaining contested piece
`[5/6, 11/12)` carries trims of winners only.  The `while` loop has no agent to
add and raises.

The input does admit an allocation with the properties SubCore promises: agent 1
gets `[0, 1/6)`, agent 2 gets `[71/96, 5/6)`, agent 3 gets `[5/6, 11/12)`, agent 4
gets `[1/6, 1/3)`, and `[11/12, 1)` stays unallocated
(`test_subcore_counterexample_has_a_valid_allocation` checks it exactly).  The
procedure does not reach it.  AM break ties with "imaginary values", a scheme not
specified precisely enough to implement (docstring of `subcore.py`); the zero
values in this example are values of the valuations, not ties between positive
values.

**Random search.**  A randomized search (not part of the test suite) ran (a) `4·10^5`
instances, each a sequence of up to six Core calls with 4–7 agents followed by one
SubCore call of the stage-3 kind, and (b) SubCore calls on 3–5 agents with
single-interval pieces and many exact ties.  (a) gave 19 failing instances; (b) gave
33 failures in about `4.7·10^5` calls with zero densities allowed and 3 failures in
about `1.45·10^6` calls with strictly positive densities.  The failures are of two
kinds: the situation above (no agent to add in the `while` loop), and an agent of
a recursive call receiving less than her required minimum.  The failure with
positive densities shows that zero densities are not necessary.

So Aziz–Mackenzie's procedure alone does not always return.  `subcore` therefore
falls back to an exact computation when it raises (§8).  With the fallback, the
input above returns a complete envy-free allocation with both cut rules
(`test_subcore_counterexample`), and so do all failing instances of the random
search except one with zero-density stretches (§9).

## 8. The equilibrium fallback (`equilibrium.py`)

When Aziz–Mackenzie's procedure raises `ProtocolError`, `subcore` calls
`equilibrium.allocate(table, pieces, agents, q)` and returns its result, or raises
if it returns `None`.  The allocation is computed from a competitive equilibrium
of the market whose buyers are the agents, whose goods are the nonempty pieces, and
in which the price of a piece is a trim position.  The theory is that of Alaei,
Jain and Malekian, *Competitive equilibria in two sided matching markets with
non-transferable utilities* (arXiv:1006.4696, "AJM"); §8.2–§8.3 give complete
proofs, filling in the steps AJM leave out.

**Positive density.**  An agent has *positive density* if she values every
nonempty interval positively.  Then she values every nonempty piece positively,
and the value of the suffix `P[x:]` of a nonempty piece is a strictly decreasing
function of the trim position `x`.

**Theorem D (the fallback).**  If every agent has positive density, then every call
`allocate(table, pieces, I, q)` made by `subcore` returns pairs `(u_j, σ_j)`,
`j ∈ I`, with

* (P1) distinct indices `u_j`, and `σ_j` a suffix of `pieces[u_j]`;
* (P2) `V_j(σ_j) ≥ q_j`;
* (P3) `V_j(σ_j) ≥ V_j(σ_k)` for all `j, k`;
* (P4) `V_j(σ_j) ≥ V_j(pieces[u])` for every index `u` that no agent holds;
* (P5) in a Core call (`n − 1` agents, the cutter's `n` pieces, `q = 0`): some `σ_j`
  is a whole piece;
* (P6) in a stage-3 call (pieces `W_h`, `h ∈ E`, and one empty piece, with
  `V_i(W_h) ≥ q_i` and `V_i(D_h) ≤ q_i` for all `i, h ∈ E`): no agent holds the empty
  piece, and every `σ_j` with `u_j = h` contains `D_h`;

and it makes at most `E(m, k) = (m − 1) k C(m + k − 3, m − 2)` calls of
`suffix_position_of_value` (each at most one Cut), for `m = |I|` agents and `k`
nonempty pieces (`bounds.E`).

Consequently, with positive densities every `subcore` call returns a result with
the properties that §6.2 and §6.6 use, so Theorem C applies, and every run of
`run_protocol` returns a complete envy-free allocation.

(P1)–(P4) are also checked by `_verify` before `allocate` returns, and (P5), (P6)
by `core` and `stage3`; the proof shows that these checks cannot fail.

### 8.1 Markets

A *market* `M = (X, Y, f)` consists of finite sets `X` (buyers) and `Y` (goods) and,
for `i ∈ X`, `j ∈ Y`, a continuous strictly decreasing function `f_i^j` from `ℝ`
onto `ℝ` (the utility of good `j` to buyer `i` at price `x`); `g_i^j` denotes its
inverse, also continuous, strictly decreasing and onto.  An *outcome*
`W = (u, p, μ)` has utilities `u ∈ ℝ^X`, prices `p ∈ ℝ^Y` and a matching `μ` (a
partial injective map `X → Y`).  It is an *equilibrium* (CE) if

* (E0) `u ≥ 0` and `p ≥ 0`;
* (E1) `u_i = f_i^j(p_j)` whenever `μ(i) = j`;
* (E2) `u_i ≥ f_i^j(p_j)` for all `i ∈ X`, `j ∈ Y`;
* (E3) `p_j = 0` for every good `j` that `μ` leaves unmatched;
* (E4) `u_i = 0` for every buyer `i` that `μ` leaves unmatched.

`μ` is a *supporting matching* of `(u, p)`.  For `p ∈ ℝ^Y` and `u ∈ ℝ^X` let
`U_i(p) = max(0, max_{j∈Y} f_i^j(p_j))` and `P^j(u) = max(0, max_{i∈X} g_i^j(u_i))`
(an empty maximum is omitted).  `M_{−i}` and `M^{−j}` remove buyer `i` or good `j`;
the *size* of `M` is `|X| + |Y|`.  The *dual* market `M*` has buyers `Y`, goods `X`
and functions `f*_j^i = g_i^j`.

*The market of a call.*  For `allocate(table, pieces, I, q)` let `J` be the indices of
the nonempty pieces, `L_j` the total length of `P_j = pieces[j]`, and identify a
price `x ∈ [0, L_j]` with the position of `P_j` before which the components of
`P_j` have total length `x`.  Put `f_i^j(x) = V_i(P_j[x:])` for `x ∈ [0, L_j]`,
`f_i^j(x) = V_i(P_j) − x` for `x < 0` and `f_i^j(x) = L_j − x` for `x > L_j`.
`V_i` is nonatomic, so `f_i^j` is continuous; with positive density it is strictly
decreasing on `[0, L_j]` from `V_i(P_j) > 0` to `0`, hence a continuous strictly
decreasing map onto `ℝ`.  In a CE every matched good has `p_j = g_i^j(u_i) ≤ g_i^j(0) = L_j`
and every unmatched good has `p_j = 0`, so prices are trim positions.

*Correspondence with the code.*  (C1) The positions `(k, y)` that the table
produces on `P_j` (boundaries `(k, c_k.a)`, interior points `(k, y)` with
`c_k.a < y < c_k.b`, the end `(len, 0)`) are ordered lexicographically, and this
order is the order of the removed length; the start of `P_j` is
`start[g] = (0, P_j.first.a)`, price 0.  (C2) For `0 ≤ v ≤ V_i(P_j)`,
`suffix_position_of_value(P_j, i, v)` returns a position whose suffix is worth `v`
to `i` (Lemma 3), which with positive density is the unique one, `g_i^j(v)`.  So
the loop of `_Market.high` over `g` computes `max(start, max_{b: u_b ≤ V_b(P_g)} g_b^g(u_b))`,
which is `P^g(u)`: a buyer with `u_b > V_b(P_g)` has `g_b^g(u_b) < 0`.  (C3)
`_Market.utility(b, views)` returns `max(0, max_g V_b(P_g[p_g:]))`, which is `U_b(p)`
for the prices `p` that produced the views (Lemma 1: the table's values are
exact).

### 8.2 General facts

**Lemma A.**  If `(u, p, μ)` is a CE, then `u = U(p)` and `p = P(u)`.

*Proof.*  A matched buyer has `u_i = f_i^{μ(i)}(p_{μ(i)})`, at least every
`f_i^j(p_j)` (E2) and at least 0 (E0), so `u_i = U_i(p)`.  An unmatched buyer has
`u_i = 0 ≥ f_i^j(p_j)` for all `j`, so `U_i(p) = 0`.  `u_i ≥ f_i^j(p_j)` is equivalent
to `g_i^j(u_i) ≤ p_j` (apply the decreasing map `g_i^j`), and `u_i = f_i^j(p_j)` to
`g_i^j(u_i) = p_j`; so `(p, u, μ^{−1})` is a CE of `M*`, and the first part applied
to `M*` gives `p = P(u)`.  ∎

The proof also shows: **(duality)** `(u, p, μ)` is a CE of `M` iff `(p, u, μ^{−1})`
is a CE of `M*`; `M** = M`, and removing a buyer of `M*` is removing a good of `M`.

**Lemma B.**  Let `W = (u, p, μ)` and `W' = (u', p', μ')` be CEs.  Partition the
buyers into `S = {u > u'}`, `S' = {u < u'}`, `S_0 = {u = u'}` and the goods into
`T = {p > p'}`, `T' = {p < p'}`, `T_0 = {p = p'}`.  Then `μ` maps `S` onto `T'`,
`μ'` maps `T'` back onto `S` (as `μ'^{−1}`), `μ'` maps `S'` onto `T`, `μ` maps `T`
back onto `S'`, and `μ` maps the matched buyers of `S_0` into `T_0`.  Consequently:

1. (*lattice*) `W ∧ W' = (max(u, u'), min(p, p'), μ'')` with `μ''(i) = μ(i)` if
   `u_i ≥ u'_i` and `μ'(i)` otherwise is a CE, and so is
   `W ∨ W' = (min(u, u'), max(p, p'), ·)`;
2. (*conservation*) a buyer with positive utility in some CE is matched in every CE,
   and a good with positive price in some CE is matched in every CE.

*Proof.*  `i ∈ S` has `u_i > u'_i ≥ 0`, so it is matched (E4); for `j = μ(i)`,
`f_i^j(p_j) = u_i > u'_i ≥ f_i^j(p'_j)`, so `p_j < p'_j`: `μ(S) ⊆ T'`.  `j ∈ T'` has
`p'_j > p_j ≥ 0`, so it is matched by `μ'` (E3) to some `i`, and
`u'_i = f_i^j(p'_j) < f_i^j(p_j) ≤ u_i`: `μ'^{−1}(T') ⊆ S`.  Both maps are
injective, so `|S| = |T'|` and both are bijections.  The statements for `S'`, `T`
follow by exchanging `W` and `W'`.  A matched `i ∈ S_0` holds in `W` a good that is
not in `T'` (held by `S`) nor in `T` (held by `S'`), so in `T_0`.

1. `μ''` maps `S` onto `T'`, `S_0` into `T_0`, `S'` onto `T`, so it is injective.
   (E1): for `i ∈ S ∪ S_0` matched, `j = μ(i)` has `min(p, p')_j = p_j` (as `j ∈ T' ∪ T_0`)
   and `max(u, u')_i = u_i = f_i^j(p_j)`; for `i ∈ S'` symmetrically with `W'`.
   (E2): `min(p,p')_j` is `p_j` or `p'_j`, and `f_i^j` of it is at most `u_i` or `u'_i`.
   (E3): a good unmatched by `μ''` is not in `T ∪ T'`, so it is in `T_0`; it is not
   held in `W` by `S_0` (else `μ''` matches it), nor by `S` or `S'` (whose
   `W`-goods lie in `T' ∪ T`), so it is unmatched in `W` and `p_j = p'_j = 0`.  (E4):
   a buyer unmatched by `μ''` lies in `S_0` (buyers of `S` and `S'` are matched) and
   is unmatched in `W`, so `u_i = u'_i = 0`.  (E0) is clear.  `W ∨ W'` is `∧` in `M*`
   (duality).
2. If `u_i > 0` in `W` and `u'_i = 0`, then `i ∈ S` and `i = μ'^{−1}(μ(i))` is
   matched in `W'`; if `u'_i > 0`, `i` is matched in `W'` by (E4).  Goods: by duality.
∎

**Lemma C (extremes).**  If `M` has a CE, it has a *lowest* CE `W_low` and a
*highest* CE `W_high`: `p(W_low) ≤ p(W) ≤ p(W_high)` and
`u(W_low) ≥ u(W) ≥ u(W_high)` for every CE `W` (the pairs `(u, p)` are unique, the
matchings need not be).  The same holds for the CEs satisfying lower bounds
`u ≥ u̲`, `p ≥ p̲` (*bounded* CEs), if there is one.

*Proof.*  For a fixed `μ`, the set of `(u, p)` satisfying (E0)–(E4) is closed (the
`f_i^j` are continuous) and bounded: a matched buyer has `u_i = f_i^j(p_j) ≤ f_i^j(0)`,
a matched good `p_j = g_i^j(u_i) ≤ g_i^j(0)`, the others are 0.  So the CEs form a
compact set (a finite union over `μ`), and so do the bounded ones.  Let `W*`
minimize `Σ_j p_j`.  For any CE `W`, `W* ∧ W` is a CE (Lemma B; it satisfies the
bounds if both do) with prices `min(p*, p) ≤ p*` and hence equal sum, so
`min(p*, p) = p*`, i.e. `p* ≤ p`.  Then `u* = U(p*) ≥ U(p) = u` (Lemma A; `U` is
nonincreasing).  The highest CE is the lowest CE of `M*`.  ∎

### 8.3 AJM's characterization

For `s ≥ 0` let `H(s)` be the statement: every market of size at most `s` has a CE,
and Theorems 2 and 5 below hold for every market of size at most `s`.  `H(s)` is
proved by induction on `s`; within the step, Theorem 2 for size `s` uses `H(s − 1)`,
Theorem 3 uses Theorem 2 for size `s`, Theorem 4 uses Theorem 3, and Theorem 5
uses Theorem 4.  All statements are closed under duality (the dual of a market
of size `s` has size `s`; the two halves of each theorem are dual to each other),
so it suffices to prove one half; the other follows in the dual market.

**Theorem 2.**  Let `M = (X, Y, f)`, `i ∈ X`, `j ∈ Y`, and let `W̄_{−i}` be the highest
CE of `M_{−i}` and `W^{−j}` the lowest CE of `M^{−j}`.  Then `M` has a CE, and

* (I) `u_i(W_low) = U_i(p(W̄_{−i}))`;
* (II) `p_j(W_high) = P^j(u(W^{−j}))`, with `P^j` over the buyers `X`;
* (III) `p(W_low) ≤ p(W̄_{−i})`, with equality at `j` if `i` and `j` are matched in
  some supporting matching of `W_low`.

*Proof.*  `M_{−i}` is smaller, so it has a CE and `W̄_{−i}` exists (Lemma C); let
`p = p(W̄_{−i})`.

*Lemma E (a CE of `M` from `W̄_{−i}`).*  There is a CE `W` of `M` with prices `p`,
`u_k(W) = u_k(W̄_{−i})` for `k ≠ i` and `u_i(W) = U_i(p)`.  If `U_i(p) = 0`, keep a
supporting matching of `W̄_{−i}` and leave `i` unmatched: (E2) for `i` is
`0 ≥ f_i^{j'}(p_{j'})`, true as `U_i(p) = 0`.  Otherwise let `j*` maximize
`f_i^{j*}(p_{j*})`.  By Theorem 5 for `M_{−i}` (smaller), `(u(W̄_{−i}), p_{−j*})` is a CE
of `M_{−i}^{−j*}` with a matching `μ'`; add `i ↦ j*`.  (E1) for `i` holds by the
choice of `j*`; (E2) for `i` because `U_i(p)` is the largest utility, for `k ≠ i` at
`j*` because `W̄_{−i}` is a CE with the same prices; (E3) and (E4) hold because `μ'`
satisfies them and `j*`, `i` are matched.  So `M` has a CE, and `W_low`, `W_high`
exist.

(III) `p(W_low) ≤ p(W) = p`.  Let `i` and `j` be matched in `W_low`.  Removing both
leaves a CE of `M_{−i}^{−j}` (the remaining buyers keep their partners, the
remaining unmatched goods were unmatched).  Let `W°` be the lowest CE of
`M_{−i}^{−j}`; then `p(W°) ≤ p(W_low)` on `Y − j`, so `u(W°) ≥ u(W_low)` on `X − i`
(Lemma A in `M_{−i}^{−j}`).  By (II) for `M_{−i}` (smaller),
`p_j = P^j_{X−i}(u(W°)) ≤ P^j_{X−i}(u(W_low)) ≤ P^j_X(u(W_low)) = p_j(W_low)` (`P^j` is
nonincreasing in `u`; Lemma A for `W_low`).  With the first part, `p_j(W_low) = p_j`.

(I) The lowest CE has the highest utilities (Lemma C), so
`u_i(W_low) ≥ u_i(W) = U_i(p)`.  If `i` is unmatched in `W_low`, `u_i(W_low) = 0`, so
both are 0.  If `i` is matched to `j`, then by (III)
`u_i(W_low) = f_i^j(p_j(W_low)) = f_i^j(p_j) ≤ U_i(p)`.

(II) is (I) in `M*`: the lowest CE of `M*` is the highest CE of `M`, the highest CE
of `M*_{−j}` is the lowest CE of `M^{−j}`, and `U*_j = P^j`.  ∎

(AJM conclude (I) in the matched case from the choice of `j*`, which need not be
the good matched to `i` in `W_low`; the argument above avoids this.)

**Theorem 3.**  Let `|X| = |Y| ≥ 1` and let `M` have a CE with `u ≥ u̲`, `p ≥ p̲`.  At the
lowest such CE some good `j*` has price `p̲_{j*}`; at the highest such CE some buyer
has utility `u̲_i`.

*Proof.*  Let `W_b` be the lowest bounded CE (Lemma C) and suppose `p(W_b) > p̲`
everywhere.  Let `M'` have the functions `f'^j_i(x) = f_i^j(x + p̲_j) − u̲_i`
(continuous, strictly decreasing, onto).  Then `(u − u̲, p − p̲, μ)` is a CE of `M'`
for every bounded CE `(u, p, μ)` of `M`: (E1), (E2) are the same equations,
(E0) is the bounds, and an unmatched good has `0 = p_j ≥ p̲_j ≥ 0`, an unmatched buyer
`0 = u_i ≥ u̲_i ≥ 0`.  So `W'_b`, the image of `W_b`, is a CE of `M'` with all prices
positive, hence with all goods and (square) all buyers matched; by Lemma B (2) every
CE of `M'` matches all goods and all buyers, and therefore maps back to a bounded CE
of `M` by `(u' + u̲, p' + p̲, μ')` ((E3), (E4) are vacuous).  `M'` has the size of
`M`, at most `s`, so Theorem 2 (III) applies to it: for a buyer `i`, `M'_{−i}` has more goods than buyers,
so its highest CE leaves a good `j*` unmatched, with price 0, and the lowest CE of
`M'` has `p'_{j*} = 0`.  Its image is a bounded CE of `M` with price `p̲_{j*}` at `j*`,
so `p_{j*}(W_b) ≤ p̲_{j*}`, a contradiction.  The second claim is the first in `M*`.  ∎

For `T ⊆ Y` let `D_T(W) = {i ∈ X : u_i = f_i^j(p_j) for some j ∈ T}` and for
`S ⊆ X` let `D_S(W) = {j ∈ Y : u_i = f_i^j(p_j) for some i ∈ S}`.

**Theorem 4 (tightness).**  If `W` is the lowest CE, then `|D_T(W)| ≥ |T| + 1` for every
nonempty set `T` of goods with positive prices.  If `W` is the highest CE, then
`|D_S(W)| ≥ |S| + 1` for every nonempty set `S` of buyers with positive utilities.

*Proof.*  Let `W = (u, p, μ)` be the lowest CE and `T` as stated.  The goods of `T`
are matched (E3); let `S = μ^{−1}(T)`, so `|S| = |T|` and `S ⊆ D_T(W)`.  Put
`p̲_j = max(0, max_{i ∈ X−S} g_i^j(u_i))` for `j ∈ T` and
`u̲_i = max(0, max_{j ∈ Y−T} f_i^j(p_j))` for `i ∈ S`.  The restriction of `W` to the
square market `(S, T)` is a `(u̲, p̲)`-bounded CE: `p_j = P^j(u) ≥ p̲_j` and
`u_i = U_i(p) ≥ u̲_i` (Lemma A).  Let `W°` be the lowest bounded CE of `(S, T)` and
replace the part of `W` on `S` and `T` by `W°`.  The result is a CE of `M`: (E2) for
`i ∈ S`, `j ∉ T` holds as `u°_i ≥ u̲_i ≥ f_i^j(p_j)`; for `i ∉ S`, `j ∈ T` as
`p°_j ≥ p̲_j ≥ g_i^j(u_i)`; the other cases of (E2), and (E0), (E1), (E3), (E4), are
inherited from `W` on `X − S`, `Y − T` (`μ` maps `X − S` into `Y − T`) and from `W°`
on `S`, `T`.  The new prices are at most `p`, and `W` is lowest, so `p° = p` on `T`:
the restriction of `W` has the prices of the lowest bounded CE.  By Theorem 3 (the market `(S, T)` has size at most `s`), some
`j* ∈ T` has `p_{j*} = p̲_{j*}`.  As `p_{j*} > 0`, `p̲_{j*} = g_{i*}^{j*}(u_{i*})` for some
`i* ∉ S`, i.e. `u_{i*} = f_{i*}^{j*}(p_{j*})`: `i* ∈ D_T(W) − S`.  The second claim is
the first in `M*`.  ∎

**Theorem 5.**  If `W` is the highest CE of `M`, then for every good `j`,
`(u, p_{−j})` is a CE of `M^{−j}`.  If `W` is the lowest CE, then for every buyer `i`,
`(u_{−i}, p)` is a CE of `M_{−i}`.

*Proof.*  Let `W = (u, p, μ)` be highest and `j ∈ Y`.  If `j` is unmatched, `μ` is a
supporting matching in `M^{−j}`.  If its holder `h` has `u_h = 0`, unmatch `h`
((E4) holds for `h`).  Otherwise grow a set `S` of buyers with positive utilities,
each `i ∈ S` with an alternating sequence `j = j_1, i_1, j_2, i_2, ..., j_r, i_r = i` of
distinct goods and buyers with `μ(i_t) = j_t` and `u_{i_t} = f_{i_t}^{j_{t+1}}(p_{j_{t+1}})`
for `t < r`; start with `S = {h}`.  By Theorem 4, `|D_S(W)| ≥ |S| + 1`, and `μ(S)`
has `|S|` elements, so some `i* ∈ S` has `u_{i*} = f_{i*}^{j*}(p_{j*})` for a good
`j* ∉ μ(S)`.  If `j*` is matched to `b` with `u_b > 0`, then `b ∉ S`; add `b` with
the sequence of `i*` extended by `j*, b`, and repeat.  Otherwise rematch along the
sequence of `i*`: `i_t ↦ j_{t+1}` for `t < r`, `i* ↦ j*`, unmatch the holder `b` of `j*`
if any (`u_b = 0`), and drop `j`.  (E1) holds by the indifferences, (E2) is
unchanged, the unmatched goods are among the previously unmatched ones (price 0),
and the unmatched buyers have utility 0.  `S` grows, so this ends.  The second
claim is the first in `M*`.  ∎

*Base and induction.*  A market without buyers has the CE with all prices 0, and
it is both lowest and highest; one without goods has the CE with all utilities 0.
In these cases Theorems 2 and 5 hold directly (Lemma E with `U_i = 0`; removing a
good or buyer leaves the same trivial CE), so `H(0)` and the markets with `X` or `Y`
empty are covered.  For the step, the proofs above use only `H(s − 1)` and the
theorems for size `s` proved before them.  ∎

**Corollary (the recursion).**  For buyers `X ⊆ I` and goods `Y ⊆ J` of the market of
a call, `_Market.low(X, Y)` returns `u(W_low(X, Y))` and `_Market.high(X, Y)` returns
`p(W_high(X, Y))` with its suffixes.

*Proof.*  Induction on `|X| + |Y|`, following the code: `low(∅, Y)` is empty;
`low(X, Y)_b = utility(b, high(X − b, Y))`, which is `U_b(p(W_high(X − b, Y)))` (C3),
which is `u_b(W_low(X, Y))` by Theorem 2 (I); `high(X, ∅)` is empty;
`high(X, Y)_g = P^g(low(X, Y − g))` (C2), which is `p_g(W_high(X, Y))` by Theorem 2 (II).
The memo tables store each result under its arguments.  ∎

### 8.4 The result of `allocate`

Let `m = |I| ≥ 1`, `i_0 = I[0]`, `k = |J|`.  `allocate` computes
`p = p(W̄_{−i_0})` (`mk.high` of all buyers but `i_0` and all goods; Corollary),
`u = U(p)` (C3), the graph of the pairs `(b, g)` with `V_b(P_g[p_g:]) = u_b`, and a
matching of it covering every buyer with `u_b > 0` and every good with
`p_g ≠ start` (`_cover_matching`).

By Lemma E there is a CE `W` of `M` with prices `p`, and by Lemma A its utilities
are `U(p) = u`.  A supporting matching of `W` lies in the graph (E1) and covers
those buyers (E4) and goods (E3).  `_cover_matching` finds such a matching when one
exists: phase 1 (augmenting paths from each needed buyer, Kuhn) covers the needed
buyers, since a matching covering them exists; phase 2 keeps every matched buyer
matched and, for each uncovered needed good `g`, flips an alternating path from `g`
ending at an unmatched buyer or at a buyer whose good is not needed.  Such a path
exists: in the symmetric difference of the current matching and a matching that
covers both sets, the component of `g` is a path starting at `g` with an edge of
the latter, and it ends at a buyer with no current edge or at a good with no edge
of the latter (not needed), and its buyer-to-good steps are current edges.  The
breadth-first search explores all such alternating paths.  So `allocate` gets a
matching `μ̂`, and `(u, p, μ̂)` is a CE: (E1) by the graph, (E2) as `u = U(p)`, (E3)
and (E4) by the covering.

Each matched agent `b` receives `σ_b = P_g[p_g:]` (`views[g]`), an unmatched one the
empty suffix of a free index (empty pieces first).

*(P1)–(P4).*  The indices are distinct (a matching, then distinct free indices).
(P3) and (P4) are (E2) at the matched goods and at the unmatched goods (price 0,
whole pieces); empty pieces are worth 0.  (P2) and the existence of the free
indices are shown below for the two kinds of calls; `_verify` then passes.

*(P5), Core calls.*  `M_{−i_0}` has `n − 2` buyers and `n` goods, so `W̄_{−i_0}`
leaves at least two goods unmatched, with price 0.  With positive density, a good
at price 0 is a whole nonempty piece worth `V_b(P_g) > 0` to every agent, so every
`u_b > 0` and every agent is matched; then exactly one good is unmatched, and one of
the two goods at price 0 is matched: its holder receives a whole piece.  (P2) is
trivial (`q = 0`).  In `core`, `free[0]` is the unmatched good, a whole piece, and
the cutter's residue loses two whole pieces worth `V_c(R)/n` each, so the factor check
passes.  (If the cutter values the residue at 0, all pieces are empty under positive
density, `J = ∅`, every agent is unmatched with utility 0 and receives an empty
piece, and the check holds trivially.)

*(P2), (P6), stage-3 calls.*  If every `W_h` is nonempty, the market is square:
`M_{−i_0}` has `|E| − 1` buyers and `|E|` goods, so some good has price 0 in
`W̄_{−i_0}` and hence in `(u, p, μ̂)`; every agent has `u_b ≥ V_b(W_h) ≥ q_b`, which
gives (P2), and `u_b > 0`, so all agents are matched, all `|E|` goods are matched, and
the empty piece is held by nobody.  If some `W_h` are empty, then `q_b = 0` for every
`b` (as `0 = V_b(W_h) ≥ q_b`), and some good is unmatched only if every agent is
matched, which is impossible with fewer goods than agents; so all goods are matched
and the `|E| − |J|` unmatched agents receive the empty pieces `W_h`, which come
before the extra empty piece among the free indices.  For `D_h`: a good `h` with
positive price has `p_h = P^h(u) = max_b g_b^h(u_b)` over the agents with
`u_b ≤ V_b(W_h)` (Lemma A, (C2)); each has `u_b ≥ q_b ≥ V_b(D_h)` (F1), and
`g_b^h(V_b(D_h))` is the start of `D_h` (the suffix from there is `D_h`), so
`g_b^h(u_b)` is at or before the start of `D_h`.  So every allocated suffix of `W_h`
contains `D_h`, and the test in `stage3` (more components than `D_h`, or as many
and the same first component) passes.

### 8.5 The number of cuts

Only `_Market.high` calls `suffix_position_of_value`: for a state `(X, Y)` at most
`|X|` times per good of `Y`.  Each state is computed once (memo).  Starting from
`(I − i_0, J)`, `high(X, Y)` calls `low(X, Y − g)`, which calls `high(X − b, Y − g)`; so
the states reached are `(I − i_0 − A, J − B)` with `|A| = |B| = t`, `0 ≤ t ≤ m − 1`,
and the calls number at most

    Σ_t C(m−1, t) C(k, t) (m−1−t)(k−t) = (m−1) k Σ_t C(m−2, t) C(k−1, t) = (m−1) k C(m+k−3, m−2)

(Vandermonde's identity).  `low`, `utility`, the matching and `_verify` make no
queries.  For a Core call on `n` agents (`m = n − 1`, `k ≤ n`) this is at most
`E(n−1, n) = (n−2) n C(2n−4, n−3)`; for the stage-3 calls of one `Main` call, the
classes partition the agents, and `e ↦ E(e, e) = e (2e−3)!/((e−2)!)^2` satisfies
`E(e, e)/e` nondecreasing, so `Σ_E E(|E|, |E|) ≤ E(n, n)` (superadditivity).  The
attempt of Aziz–Mackenzie's procedure before a fallback places at most `S_m` trims
(its run is a prefix of a complete run, `PROOF.md` Lemma 5.1).  These counts are
`bounds.subcore_cuts` and `bounds.core_queries`; the check of `MainProtocol._core`
uses the latter, and the query bound of `PROOF.md` §9 uses both.

## 9. Zero densities

Without positive density the fallback still runs and its result is still checked,
so Theorem A holds, but Theorem D is not claimed, and its proof fails at a
specific point.  If an agent values a stretch of `P_j` at zero, `f_i^j` is constant
there, and the equation `V_i(P_j[x:]) = v` has an interval `[l, r]` of solutions.
The equilibrium theory of §8 needs a definite choice in this interval, and which
choice is right depends on the market: in a random search over markets with
zero-density stretches (single-interval pieces, 2–4 agents), taking always the left
end `l` gave a non-equilibrium in 1 of 1500 markets, taking always the right end
`r` in 278, and the mixed rule (left end in the lowest, right end in the highest
equilibria of the recursion) in 15.  The Robertson–Webb oracle decides the choice:
a Cut answer may be any point of `[l, r]` (the oracle of this repository returns
consistently the left or the right end), and no finite sequence of queries locates
the other end exactly.  A standard device for ties, perturbing every valuation to
`V_i + ελ` (Lebesgue measure `λ`) and letting `ε → 0`, does not help in this model:
the limit positions depend on the ratio of two agents' densities at a point, which
Robertson–Webb queries do not determine.  Neither this repository nor the
literature has a procedure for zero-density inputs with a proof and a bound in `n`:

* Aziz–Mackenzie's last version (arXiv:1604.03655v12, August 2017, after
  Bechler–Bitikofer) adds "imaginary values" for ties (§3.1.1), but states that
  "all the cuts that are actually made on the cake are in reality only dependent on
  the physical part of the value", so where a trim lands inside a zero-value stretch
  is not determined; the CACM version (2020) does not give SubCore's details;
* Dehghani, Farhadi, HajiAghayi and Yami (SODA 2018) state a simpler cake Core
  (try all orderings of agents and pieces) without proof; it assumes the existence of
  a Core allocation, taken from Aziz–Mackenzie, and Yami's thesis (UMD 2020) drops it;
* none of the 195 works citing Aziz–Mackenzie in Semantic Scholar (titles screened,
  the 14 closest read in full) mentions the gap; Bechler–Bitikofer is not indexed by
  Google Scholar or Semantic Scholar;
* Kurokawa, Lai and Procaccia (AAAI 2013, Theorem 1) show that a bounded envy-free
  protocol for piecewise-uniform valuations (densities in `{0, c}`) is one for all
  valuations: zero-value stretches are the general problem, not a special case.  No
  lower bound beyond `Ω(n^2)` (Procaccia 2009) is known for pieces that may be
  unions of intervals.

Measured: the fallback returns a valid result on 55 of the 56 failing inputs of the
random search of §7 (the exception has two agents who value all relevant cake at
0, and the run raises `ProtocolError`), and on every instance of a direct test of
the fallback with positive densities (`tests/test_core_subcore.py`).
