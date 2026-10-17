"""Functions checked by the meter, with their parameters read from the call's arguments
and the bound of :mod:`cake_cutting.opbounds` they must respect."""

from __future__ import annotations

from cake_cutting import base_cases, bookkeeping, equilibrium, main, pieces, seq, simultaneous, subcore
from cake_cutting import opbounds as O
from cake_cutting.seq import height, max_height


def _n(table):
    return len(table.agents_all)


def _m(table):
    return len(table.active)


def _base_h(piece):
    """Height of the tree a (tree-backed) piece is a view of."""
    return height(piece._tree) if piece._tree is not None else 0


def _tree_params(t):
    if t is None:
        return 0, 1
    return t.h, len(t.vals)


def targets():
    T = {}

    def add(fn, name, params, bound):
        code = fn.fget.__code__ if isinstance(fn, property) else fn.__code__
        T[code] = (name, params, bound)

    add(seq.split, "seq.split",
        lambda L: (max(1, _tree_params(L["t"])[0]), _tree_params(L["t"])[1]), O.c_split)
    add(seq.split3, "seq.split3",
        lambda L: (max(1, _tree_params(L["t"])[0]), _tree_params(L["t"])[1]), O.c_split3)
    add(seq.join, "seq.join",
        lambda L: (abs(height(L["tl"]) - height(L["tr"])), len(L["vals"])), O.c_join)
    add(seq.concat, "seq.concat",
        lambda L: (max(height(L["tl"]), height(L["tr"]), 1),
                   len((L["tl"] or L["tr"]).vals)) if (L["tl"] or L["tr"]) else None,
        O.c_concat)
    add(seq.find_ge, "seq.find_ge", lambda L: (height(L["t"]),), O.c_find_ge)
    add(seq.prefix_sum, "seq.prefix_sum", lambda L: (height(L["t"]),), O.c_prefix_sum)
    add(seq.nth, "seq.nth", lambda L: (height(L["t"]),), O.c_nth)

    VT = bookkeeping.ValueTable
    # a nonempty tuple-backed piece (the cake, before its first evaluation) builds its
    # tree on first use; that one-time cost is charged to the top level (c_cake_tree)
    add(VT.value, "value",
        lambda L: None if (L["piece"]._tree is None and len(L["piece"]) > 0) else (_base_h(L["piece"]),),
        O.c_value)
    add(VT.split_by_cut, "split_by_cut", lambda L: (_m(L["self"]), _n(L["self"])), O.c_split_by_cut)
    add(VT.complement_prefix, "complement_prefix", lambda L: (_m(L["self"]), _n(L["self"])),
        O.c_complement_prefix)
    add(VT.position_of_value, "position_of_value",
        lambda L: (_m(L["self"]), _n(L["self"]), _base_h(L["piece"])), O.c_position_of_value)
    add(VT.suffix_position_of_value, "suffix_position_of_value",
        lambda L: (_m(L["self"]), _n(L["self"]), _base_h(L["piece"])), O.c_suffix_position_of_value)
    add(VT.normalize_left, "normalize_left", lambda L: (_n(L["self"]), _base_h(L["piece"])),
        O.c_normalize_left)
    add(VT.suffix_from, "suffix_from", lambda L: (_m(L["self"]), _n(L["self"]), _base_h(L["piece"])),
        O.c_suffix_from)
    add(VT.prefix_of_value, "prefix_of_value",
        lambda L: (_m(L["self"]), _n(L["self"]), _base_h(L["piece"])) if not L["piece"].is_view else None,
        O.c_prefix_of_value)
    add(VT.prefix_to, "prefix_to",
        lambda L: (_n(L["self"]), _base_h(L["piece"])) if not L["piece"].is_view else None,
        O.c_prefix_to)

    def concat_params(L):
        ps = L["pieces"]
        if not isinstance(ps, list):
            return None                 # a generator: its work is the caller's
        nonempty = [p for p in ps if len(p) > 0]
        if not nonempty:
            return None
        owner = nonempty[0]._owner
        n = len(owner.agents_all)
        m = len(owner.active)
        total = sum(len(p) for p in nonempty)
        h_cat = max_height(max(total, 1))
        views = [(_base_h(p), 1) for p in nonempty if p.is_view]
        return (len(ps), h_cat, n, views)

    add(pieces.OrderedPiece.concat, "OrderedPiece.concat", concat_params,
        lambda Kin, h, n, views: O.c_piece_concat(Kin, h, n, views))

    def core_params(L):
        table = L["table"]
        m = len(L["agents"])
        return (m, _n(table), max_height(len(L["residue"]) + m + 1))

    add(subcore.core, "core", core_params, O.c_core)

    def subcore_params(L):
        table = L["table"]
        h = max([_base_h(p) for p in L["pieces"]] + [1])
        k = sum(1 for p in L["pieces"] if len(p) > 0)
        return (_n(table), _m(table), h, len(L["agents"]), len(L["pieces"]), k)

    add(subcore.subcore, "subcore", subcore_params,
        lambda n, m, h, w, P, k: O.Ctx(n, m, h).subcore(w, P, k=k))
    add(equilibrium.allocate, "allocate", subcore_params,
        lambda n, m, h, w, P, k: O.Ctx(n, m, h).fallback(w, k, P))

    def l46_params(L):
        table = L["table"]
        m = len(L["agents"])
        c = O.lemma46_counts(m)
        if len(L["requests"]) > c["K"] or L["delta"] < __import__("cake_cutting").bounds.stage3_delta(m):
            return None
        return (m, _n(table), max_height(len(L["residue"]) + c["H"] + 1))

    add(simultaneous.simultaneous_cut, "simultaneous_cut", l46_params, O.c_lemma46)

    def base_params(L):
        table = L["table"]
        m = len(L["agents"])
        return (m, _n(table), max_height(2 * (len(L["cake"]) + 8)))

    add(base_cases.base_case, "base_case", base_params, O.c_base_case)

    def main_params(L):
        self = L["self"]
        n = len(self.table.agents_all)
        return (len(L["N"]), n)

    def main_bound(m, n):
        if m <= 3:
            return O.c_main_base(m, n)
        return (sum(O.lambda_op(k, n) for k in range(4, m + 1))
                + max(O.c_main_base(b, n) for b in (1, 2, 3)))

    add(main.MainProtocol.main, "main", main_params, main_bound)
    add(main.run_protocol, "run_protocol",
        lambda L: (len(L["oracle"].agents if L["agents"] is None else L["agents"]),), O.operation_bound)
    return T
