"""``Main(N, C)`` -- Algorithms 1-3 of Ye–Bai (Sections 4.3-4.6, Appendix E).

One call ``Main(N, C)`` with ``n = |N| >= 4`` agents proceeds in three stages
(Section 3.1); ``n <= 3`` is handled by :mod:`cake_cutting.base_cases`.

Stage 1 -- pre-processing (Section 4.3, Algorithm 2, Lemma 4.7)
    ``Q^2`` consecutive Core calls per agent as the cutter produce the
    ``nQ^2`` snapshots; bonuses ``d_{s,i,k}`` are classified small /
    intermediate / large w.r.t. the current residue; while an intermediate
    bonus exists its owner runs a prescribed sequence of ``D_mid`` Core calls
    (``Shrink``).  The final residue is ``R*``.

Stage 2 -- pruning (Section 4.4, Lemmas 4.8-4.10)
    If every agent values ``R*`` at zero the protocol finishes.  Otherwise the
    marks ``F_{s,i}``, the witness sets of the ``Q^2`` snapshots of the chosen
    cutter ``c``, the pruning loop (pairs with ``1..Q-1`` witnesses lose all
    of them), the graphs ``G_s`` and the sink-SCC shortcut.

Stage 3 -- the partial allocation (Section 4.5, Algorithm 3, Lemmas 4.11-4.19)
    Witness snapshots and matchings (Lemma 4.12), ranks (Lemma 4.13),
    additions and rank pieces via Lemma 4.6 (Lemma 4.14), ``W_h = [P_h, A_h,
    D_h]`` and one SubCore call per profile class with required minima
    (Lemma 4.17), the allocation ``X`` with residue ``R_1`` (Lemma 4.18),
    prescribed sequences for the agents outside the chosen class ``E_*`` and
    the recursion ``Main(E_*, R_2)`` (Section 4.6, Proposition 4.20).

Every "choose" of the pseudocode is resolved by a fixed rule (Appendix C.2);
the rules used here are listed in :data:`TIE_BREAKING`.  The protocol touches
the agents only through the value table (hence the counting oracle).  Several
guarantees proved in the paper are re-checked at run time from stored values
(no queries); a violation raises :class:`ProtocolError` rather than returning
a wrong allocation.
"""

from __future__ import annotations

import bisect
from fractions import Fraction
from typing import Callable, Dict, FrozenSet, Hashable, Iterable, List, Optional, Sequence, Set, Tuple

from . import bounds
from .allocation import Allocation
from .base_cases import base_case
from .bookkeeping import ValueTable
from .oracle import RobertsonWebbOracle
from .pieces import OrderedPiece, ZERO
from .simultaneous import SimultaneousCutError, simultaneous_cut
from .subcore import ProtocolError, core, subcore, unallocated_part

Agent = Hashable
SnapshotId = Tuple[Agent, int]

TIE_BREAKING = """
Deterministic choices (Appendix C.2 leaves them to a fixed rule):
* agents are processed in increasing index order everywhere (cutters in Stage 1,
  the agent whose intermediate bonus is treated, the agents of a prescribed
  sequence, the agents of a SubCore call, the pairs (i, h));
* the agent receiving R* when nobody values it: the smallest agent;
* the cutter c of Stage 2: the smallest agent with V_c(R*) > 0;
* pruning removes the witnesses of the lexicographically first pair with 1..Q-1 witnesses;
* the shortcut uses the first snapshot (in order) whose graph is not strongly
  connected and the sink component containing the smallest agent;
* for a pair (i, h) the witness s is the earliest unused one, the label k is the
  smallest in F_{s,i} \\ F_{s,h}, and the matching closes the shortest k -> h
  path found by breadth-first search with neighbours in increasing order;
* J is padded with the earliest unused snapshots of P;
* the class E_* to recurse on is the smallest profile class (ties: smallest agent);
* R_1 lists the unused part of R* in the order of R* followed by Y_h in agent order.
"""


class Snapshot:
    """One Core allocation stored during pre-processing (Section 4.3)."""

    __slots__ = ("sid", "pieces", "value")

    def __init__(self, sid: SnapshotId, pieces: Dict[Agent, OrderedPiece]) -> None:
        self.sid = sid
        self.pieces = pieces              # label k -> c_{s,k}
        self.value: Dict[Tuple[Agent, Agent], Fraction] = {}   # (i, k) -> V_i(c_{s,k})

    def record(self, table: ValueTable, agents: Sequence[Agent]) -> None:
        for i in agents:
            for k in agents:
                self.value[(i, k)] = table.value(i, self.pieces[k])

    def b(self, i: Agent) -> Fraction:
        return self.value[(i, i)]

    def d(self, i: Agent, k: Agent) -> Fraction:
        return self.value[(i, i)] - self.value[(i, k)]

    def allocation(self) -> Allocation:
        return Allocation(self.pieces)


def identity(snapshots: Iterable[Snapshot]) -> Allocation:
    """``Id(T)``: agent ``i`` receives the label-``i`` piece of every snapshot of ``T``."""
    out: Dict[Agent, List[OrderedPiece]] = {}
    for s in snapshots:
        for k, p in s.pieces.items():
            out.setdefault(k, []).append(p)
    return Allocation({k: OrderedPiece.concat(ps) for k, ps in out.items()})


class MainProtocol:
    """Runs ``Main(A, [0, 1])`` against a Robertson–Webb oracle."""

    def __init__(self, oracle: RobertsonWebbOracle, agents: Optional[Sequence[Agent]] = None, *,
                 trace: Optional[Callable[[str], None]] = None, check: bool = True) -> None:
        self.oracle = oracle
        self.all_agents = sorted(oracle.agents if agents is None else agents)  # type: ignore[type-var]
        self.table = ValueTable(oracle, self.all_agents)
        self.trace = trace
        self.check = check
        self.stats: Dict[str, object] = {
            "core_calls": 0, "core_calls_snapshots": 0, "core_calls_stage1_shrink": 0,
            "core_calls_stage2_shrink": 0, "core_calls_stage3_shrink": 0,
            "stage1_shrink_steps": 0, "max_core_queries": 0, "core_query_bound_ok": True,
            "lemma46_queries": 0, "lemma46_phase": [], "subcore_direct_trims": 0,
            "shortcuts": 0, "recursion": [], "zero_residue_exits": 0,
            "additions_nonempty": 0, "rank_pieces_nonempty": 0, "witness_pairs": 0,
            "profile_classes": [],
        }

    # -- utilities ----------------------------------------------------------
    def log(self, msg: str) -> None:
        if self.trace is not None:
            self.trace(msg)

    def V(self, i: Agent, piece: OrderedPiece) -> Fraction:
        return self.table.value(i, piece)

    def run(self) -> Allocation:
        self.table.register_cake()
        return self.main(self.all_agents, OrderedPiece.cake())

    def _core(self, N: Sequence[Agent], U: OrderedPiece, c: Agent, stage: str
              ) -> Tuple[Dict[Agent, OrderedPiece], OrderedPiece]:
        q0 = self.oracle.queries
        X, U2, _ = core(self.table, N, U, c)
        used = self.oracle.queries - q0
        self.stats["core_calls"] += 1          # type: ignore[operator]
        self.stats[stage] += 1                 # type: ignore[operator]
        if used > self.stats["max_core_queries"]:   # type: ignore[operator]
            self.stats["max_core_queries"] = used
        if used > bounds.core_queries(len(N)):
            self.stats["core_query_bound_ok"] = False
            raise ProtocolError(f"Core used {used} queries > bound {bounds.core_queries(len(N))}")
        return X, U2

    def shrink(self, N: Sequence[Agent], U: OrderedPiece, K: Allocation, c: Agent, D: int,
               stage: str) -> Tuple[Allocation, OrderedPiece]:
        """``Shrink(N, U, K, c, D)`` of Appendix E: at most ``D`` Core calls with cutter ``c``
        on successive residues, stopping once ``V_c(U) = 0`` (Proposition 4.3 (iii))."""
        for _ in range(D):
            if self.V(c, U) == 0:
                break
            X, U = self._core(N, U, c, stage)
            K = K.oplus(Allocation(X))
        return K, U

    # -- Algorithm 1 --------------------------------------------------------
    def main(self, N: Sequence[Agent], C: OrderedPiece) -> Allocation:
        N = sorted(N)  # type: ignore[type-var]
        n = len(N)
        self.table.set_active(N)
        self.stats["recursion"].append(n)      # type: ignore[union-attr]
        if self.trace is not None:
            self.log(f"Main(n={n}, |C|={len(C)} components)")
        if n <= 3:
            return Allocation(base_case(self.table, N, C))
        Q = n * (n - 1)

        # ---- stage 1 ----
        H, S, R_star = self.preprocess(N, C)
        if self.trace is not None:
            self.log(f"  stage 1 done: {len(S)} snapshots, residue has {len(R_star)} components")

        # ---- stage 2 ----
        if all(self.V(i, R_star) == 0 for i in N):
            self.stats["zero_residue_exits"] += 1   # type: ignore[operator]
            if self.trace is not None:
                self.log("  nobody values R*: finishing with H + Id(S)")
            return H.oplus(identity(S)).give(N[0], R_star)

        eps = {i: self.V(i, R_star) / (4 * n * n * (Q + 1) * Q * Q) for i in N}
        c = next(i for i in N if self.V(i, R_star) > 0)
        P: List[Snapshot] = [s for s in S if s.sid[0] == c]
        assert len(P) == Q * Q
        marks: Dict[SnapshotId, Dict[Agent, FrozenSet[Agent]]] = {
            s.sid: {i: frozenset(k for k in N if s.d(i, k) > eps[i]) for i in N} for s in P}
        P = self.prune(N, P, marks, Q)
        if self.trace is not None:
            self.log(f"  stage 2: cutter {c!r}, {len(P)} snapshots survive pruning")

        for s in P:
            sink = self.sink_component(N, marks[s.sid])
            if sink is not None:
                self.stats["shortcuts"] += 1   # type: ignore[operator]
                if self.trace is not None:
                    self.log(f"  shortcut: G_{s.sid} not strongly connected, sink component {sorted(sink)}")
                Y = H.oplus(identity(S))
                U = R_star
                for i in sorted(sink):  # type: ignore[type-var]
                    if self.V(i, R_star) > 0:
                        Y, U = self.shrink(N, U, Y, i, bounds.D_shortcut(n), "core_calls_stage2_shrink")
                rest = [j for j in N if j not in sink]
                if self.check:
                    self.assert_domination(Y, U, dominators=sorted(sink), dominated=rest, where="Lemma 4.10")
                return Y.oplus(self.main(rest, U))

        # ---- stage 3 ----
        return self.stage3(N, Q, H, S, P, marks, R_star, eps)

    # -- Algorithm 2 --------------------------------------------------------
    def preprocess(self, N: Sequence[Agent], C: OrderedPiece
                   ) -> Tuple[Allocation, List[Snapshot], OrderedPiece]:
        n = len(N)
        Q = n * (n - 1)
        U = C
        H = Allocation()
        S: List[Snapshot] = []
        for c in N:
            for t in range(1, Q * Q + 1):
                X, U = self._core(N, U, c, "core_calls_snapshots")
                snap = Snapshot((c, t), X)
                snap.record(self.table, N)
                S.append(snap)
        # positive bonuses of every agent, sorted, for the intermediate-bonus test
        positive: Dict[Agent, List[Fraction]] = {}
        for i in N:
            vals = set()
            for s in S:
                for k in N:
                    d = s.d(i, k)
                    if d < 0:
                        raise ProtocolError(f"snapshot {s.sid} is not envy-free: d_{{s,{i!r},{k!r}}} = {d}")
                    if d > 0:
                        vals.add(d)
            positive[i] = sorted(vals)
        D_mid = bounds.D_mid(n)
        steps = 0
        while True:
            chosen = None
            for i in N:
                v = self.V(i, U)
                if v == 0:
                    continue  # Gamma_i = eps_i = 0: every bonus is small (= 0) or large (> 0)
                Gamma = v / (Q * Q)
                eps_i = Gamma / (4 * n * (Q + 1)) / n
                idx = bisect.bisect_right(positive[i], eps_i)
                if idx < len(positive[i]) and positive[i][idx] < Gamma:
                    chosen = i
                    break
            if chosen is None:
                break
            steps += 1
            if steps > n * Q ** 3:
                raise ProtocolError("Stage 1 made more than nQ^3 prescribed sequences (Lemma 4.7)")
            H, U = self.shrink(N, U, H, chosen, D_mid, "core_calls_stage1_shrink")
        self.stats["stage1_shrink_steps"] += steps   # type: ignore[operator]
        return H, S, U

    # -- Section 4.4 --------------------------------------------------------
    @staticmethod
    def witnesses(P: Sequence[Snapshot], marks, i: Agent, h: Agent) -> List[Snapshot]:
        return [s for s in P if not marks[s.sid][i] <= marks[s.sid][h]]

    def prune(self, N: Sequence[Agent], P: List[Snapshot], marks, Q: int) -> List[Snapshot]:
        """Remove the witnesses of pairs with between 1 and Q-1 witnesses until none remain."""
        while True:
            removed = False
            for i in N:
                for h in N:
                    if i == h:
                        continue
                    W = self.witnesses(P, marks, i, h)
                    if 1 <= len(W) <= Q - 1:
                        drop = {s.sid for s in W}
                        P = [s for s in P if s.sid not in drop]
                        removed = True
                        break
                if removed:
                    break
            if not removed:
                break
        if len(P) < Q:
            raise ProtocolError(f"pruning left {len(P)} < Q snapshots (Lemma 4.9)")
        return P

    @staticmethod
    def sink_component(N: Sequence[Agent], F: Dict[Agent, FrozenSet[Agent]]) -> Optional[Set[Agent]]:
        """A sink strongly connected component of ``G_s`` (edges ``i -> k`` iff ``k`` not in
        ``F_{s,i}``), or ``None`` if the graph is strongly connected."""
        succ = {i: [k for k in N if k not in F[i]] for i in N}

        def reach(start: Agent, adj) -> Set[Agent]:
            seen = {start}
            stack = [start]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            return seen

        pred = {i: [k for k in N if i in succ[k]] for i in N}
        comps: List[Set[Agent]] = []
        assigned: Set[Agent] = set()
        for i in N:
            if i in assigned:
                continue
            comp = reach(i, succ) & reach(i, pred)
            comps.append(comp)
            assigned |= comp
        if len(comps) == 1:
            return None
        sinks = [comp for comp in comps if all(v in comp for u in comp for v in succ[u])]
        return min(sinks, key=lambda comp: min(comp))  # type: ignore[type-var]

    # -- Section 4.5 / Algorithm 3 ------------------------------------------
    def stage3(self, N, Q, H, S, P, marks, R_star, eps) -> Allocation:
        n = len(N)
        # profiles and classes
        profile = {i: tuple(marks[s.sid][i] for s in P) for i in N}
        classes: List[List[Agent]] = []
        for i in N:
            for E in classes:
                if profile[E[0]] == profile[i]:
                    E.append(i)
                    break
            else:
                classes.append([i])
        if len(classes) < 2:
            raise ProtocolError("fewer than two profile classes (Lemma 4.9)")
        self.stats["profile_classes"].append([list(E) for E in classes])   # type: ignore[union-attr]

        def preceq(i: Agent, h: Agent) -> bool:
            return all(fi <= fh for fi, fh in zip(profile[i], profile[h]))

        # witnesses and matchings (Lemma 4.12)
        J: List[Snapshot] = []
        used: Set[SnapshotId] = set()
        mu: Dict[SnapshotId, Dict[Agent, Agent]] = {}
        for i in N:
            for h in N:
                if i == h or preceq(i, h):
                    continue
                cands = [s for s in self.witnesses(P, marks, i, h) if s.sid not in used]
                if not cands:
                    raise ProtocolError(f"no unused witness for the pair ({i!r}, {h!r}) (Lemma 4.12)")
                s = cands[0]
                k = min(marks[s.sid][i] - marks[s.sid][h])  # type: ignore[type-var]
                mu[s.sid] = self.matching_through(N, marks[s.sid], h, k)
                used.add(s.sid)
                J.append(s)
        for s in P:
            if len(J) >= Q:
                break
            if s.sid not in used:
                used.add(s.sid)
                mu[s.sid] = {i: i for i in N}
                J.append(s)
        if len(J) != Q:
            raise ProtocolError("could not select Q snapshots (Lemma 4.12)")
        J.sort(key=lambda s: s.sid[1])
        # ranks (Lemma 4.13)
        rank_of_class = self.ranks(classes, profile)
        rank = {h: rank_of_class[ci] for ci, E in enumerate(classes) for h in E}
        if sum(rank.values()) > Q or any(not 0 <= r <= n - 1 for r in rank.values()):
            raise ProtocolError("ranks violate Lemma 4.13")
        if self.trace is not None:
            self.log(f"  stage 3: classes {classes}, ranks {rank}, {len(J)} matched snapshots")

        # Algorithm 3: additions and rank pieces
        L = Fraction(1, n * Q * Q)
        delta = L / (16 * n * (Q + 1))
        VR = {i: self.V(i, R_star) for i in N}
        requests: Dict[Hashable, Fraction] = {}
        for s in J:
            for k in N:
                r = ZERO
                for i in N:
                    if VR[i] > 0:
                        d = s.d(i, k)
                        if d <= eps[i] and d / VR[i] > r:
                            r = d / VR[i]
                requests[("a", s.sid, k)] = r + delta
        for h in N:
            if rank[h] > 0:
                requests[("p", h)] = L * (rank[h] - Fraction(1, 2)) + delta
        if sum(requests.values(), ZERO) > 1:
            raise ProtocolError("requested fractions exceed 1 (Lemma 4.14)")
        budget = bounds.lemma46_queries(n, len(requests), delta)
        q0 = self.oracle.queries
        try:
            cut, leftover, cutter = simultaneous_cut(self.table, N, R_star, requests, delta, budget=budget)
        except SimultaneousCutError as exc:
            raise ProtocolError(f"Lemma 4.6 failed: {exc}") from exc
        self.stats["lemma46_queries"] += self.oracle.queries - q0   # type: ignore[operator]
        self.stats["lemma46_phase"].append(cutter.phase_used)      # type: ignore[union-attr]
        A = {(s.sid, k): cut[("a", s.sid, k)] for s in J for k in N}
        Ppiece = {h: (cut[("p", h)] if rank[h] > 0 else OrderedPiece.empty()) for h in N}
        self.stats["additions_nonempty"] += sum(1 for p in A.values() if not p.is_empty)  # type: ignore[operator]
        self.stats["rank_pieces_nonempty"] += sum(1 for p in Ppiece.values() if not p.is_empty)  # type: ignore[operator]
        self.stats["witness_pairs"] += len(used) - sum(1 for s in J if all(mu[s.sid][i] == i for i in N))  # type: ignore[operator]
        if self.check:
            self.check_lemma_4_14(N, J, A, Ppiece, eps, VR, L, rank, delta)
        D = {h: OrderedPiece.concat(s.pieces[mu[s.sid][h]] for s in J) for h in N}
        Ah = {h: OrderedPiece.concat(A[(s.sid, mu[s.sid][h])] for s in J) for h in N}
        B = {i: sum((s.b(i) for s in J), ZERO) for i in N}

        # SubCore within each profile class (Lemma 4.17)
        Z = Allocation()
        Y: Dict[Agent, OrderedPiece] = {}
        for ci, E in enumerate(classes):
            ell = rank_of_class[ci]
            W = {h: OrderedPiece.concat([Ppiece[h], Ah[h], D[h]]) for h in E}
            q = {i: B[i] + max(ZERO, L * (ell - Fraction(1, 2))) * VR[i] for i in E}
            for i in E:
                for h in E:
                    if self.V(i, W[h]) < q[i]:
                        raise ProtocolError(f"V_{i!r}(W_{h!r}) < q_{i!r} (Lemma 4.17 (ii))")
            pieces = [W[h] for h in E] + [OrderedPiece.empty()]
            result, trims = subcore(self.table, pieces, E, q)
            self.stats["subcore_direct_trims"] += trims   # type: ignore[operator]
            for i in E:
                idx, suffix = result[i]
                if idx >= len(E):
                    raise ProtocolError("SubCore allocated the empty piece (Proposition 4.5)")
                h = E[idx]
                # suffix is a suffix of W_h = [P_h, A_h, D_h]; it contains D_h entirely iff it
                # starts no later than D_h does (O(1) instead of comparing components)
                if len(D[h]) > 0 and not (len(suffix) > len(D[h])
                                          or (len(suffix) == len(D[h]) and suffix.first == D[h].first)):
                    raise ProtocolError("a trim entered D_h (Lemma 4.17 (iv))")
                Z = Z.give(i, suffix)
                Y[h] = unallocated_part(self.table, W[h], suffix)
        R1 = OrderedPiece.concat([leftover] + [Y[h] for h in N])
        X = H.oplus(identity(s for s in S if s.sid not in used)).oplus(Z)
        if self.check:
            for i in N:
                for j in N:
                    if self.V(i, X[j]) > self.V(i, X[i]):
                        raise ProtocolError(f"X is not envy-free: {i!r} envies {j!r} (Lemma 4.19)")

        # Section 4.6: domination and recursion
        E_star = min(classes, key=lambda E: (len(E), E[0]))
        U = R1
        values_R1 = {i: self.V(i, R1) for i in N}
        for i in N:
            if i in E_star or values_R1[i] == 0:
                continue
            X, U = self.shrink(N, U, X, i, bounds.D_final(n), "core_calls_stage3_shrink")
        if self.check:
            self.assert_domination(X, U, dominators=[i for i in N if i not in E_star],
                                   dominated=E_star, where="Proposition 4.20")
        return X.oplus(self.main(E_star, U))

    @staticmethod
    def matching_through(N: Sequence[Agent], F: Dict[Agent, FrozenSet[Agent]], h: Agent, k: Agent
                         ) -> Dict[Agent, Agent]:
        """Lemma 4.11: a perfect matching of ``G_s`` using the edge ``h -> k``."""
        if h == k:
            return {i: i for i in N}
        succ = {i: [v for v in N if v not in F[i] and v != i] for i in N}
        parent: Dict[Agent, Optional[Agent]] = {k: None}
        queue = [k]
        while queue and h not in parent:
            u = queue.pop(0)
            for v in succ[u]:
                if v not in parent:
                    parent[v] = u
                    queue.append(v)
        if h not in parent:
            raise ProtocolError("G_s is not strongly connected (no path k -> h for Lemma 4.11)")
        matching = {i: i for i in N}
        v = h
        while parent[v] is not None:
            u = parent[v]
            matching[u] = v
            v = u
        matching[h] = k
        if sorted(matching.values(), key=repr) != sorted(N, key=repr):
            raise ProtocolError("constructed matching is not perfect")
        return matching

    @staticmethod
    def ranks(classes: List[List[Agent]], profile) -> List[int]:
        """``ell_E``: the longest chain ``F_E = F_{E_0} < F_{E_1} < ...`` (Lemma 4.13)."""
        m = len(classes)
        prof = [profile[E[0]] for E in classes]

        def strictly_below(a: int, b: int) -> bool:
            return prof[a] != prof[b] and all(x <= y for x, y in zip(prof[a], prof[b]))

        memo: Dict[int, int] = {}

        def longest(a: int) -> int:
            if a in memo:
                return memo[a]
            best = 0
            for b in range(m):
                if strictly_below(a, b):
                    best = max(best, 1 + longest(b))
            memo[a] = best
            return best

        return [longest(a) for a in range(m)]

    # -- run-time checks of proved guarantees (no queries) -------------------
    def check_lemma_4_14(self, N, J, A, Ppiece, eps, VR, L, rank, delta) -> None:
        n = len(N)
        Q = n * (n - 1)
        for s in J:
            for k in N:
                for i in N:
                    v = self.V(i, A[(s.sid, k)])
                    d = s.d(i, k)
                    eta = VR[i] / (4 * n * (Q + 1) * Q * Q)
                    if d <= eps[i] and v < d:
                        raise ProtocolError("Lemma 4.14 (i) violated")
                    if v > eta:
                        raise ProtocolError("Lemma 4.14 (ii) violated")
        for h in N:
            for i in N:
                v = self.V(i, Ppiece[h])
                lo = max(ZERO, L * (rank[h] - Fraction(1, 2))) * VR[i]
                hi = L * rank[h] * VR[i]
                if not (lo <= v <= hi):
                    raise ProtocolError("Lemma 4.14 (iii) violated")

    def assert_domination(self, X: Allocation, U: OrderedPiece, dominators, dominated, where: str) -> None:
        for i in dominators:
            vi = self.V(i, X[i])
            vu = self.V(i, U)
            for j in dominated:
                if vi - self.V(i, X[j]) < vu:
                    raise ProtocolError(f"{i!r} does not dominate {j!r} ({where})")


def run_protocol(oracle: RobertsonWebbOracle, agents: Optional[Sequence[Agent]] = None, *,
                 trace: Optional[Callable[[str], None]] = None, check: bool = True
                 ) -> Tuple[Allocation, Dict[str, object]]:
    """Run ``Main(A, [0, 1])``; returns the complete allocation and run statistics."""
    protocol = MainProtocol(oracle, agents, trace=trace, check=check)
    allocation = protocol.run()
    return allocation, protocol.stats
