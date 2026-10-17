"""Instruction profiles of the protocol's functions (the cost model of the operation bound).

The operation bound (``cake_cutting.opbounds``, proof in ``PROOF.md``) counts
operations in the following model:

* one **operation** is the execution of one CPython 3.13 bytecode instruction of
  this package's code (as listed by :func:`dis.get_instructions`);
* an instruction that performs an arithmetic operation or comparison on numbers
  counts as one operation whatever the size of the numbers -- in particular a
  ``fractions.Fraction`` operation is one operation, although ``fractions`` is
  written in Python ("an arithmetic operation is one operation");
* a dictionary or set lookup / insertion counts as one operation (the instruction
  performing it), including the hashing of its key;
* a call of a built-in (``len``, ``max``, ``sorted``, ``list.append``, ...) costs
  one operation per element it touches (at least one) in addition to its call
  instruction;
* an oracle query (``RobertsonWebbOracle.eval`` / ``cut``) costs one operation:
  what the agent does to answer it is not the protocol's work;
* the text of an exception message on an error exit and the ``trace`` callback
  (``None`` by default) are not counted.

This module derives, for a function, the number of its instructions that run at
most once per call and, for every loop of its source, the number that run at most
once per iteration and at most once per entry of the loop.  Instructions are
attributed to the innermost loop (``for``, ``while``, inlined comprehension, or
``yield from``, which compiles to a loop over the delegated values) whose source
span contains their position; the iterable of a ``for`` loop or of a
comprehension's first generator is attributed to the enclosing level, since it is
evaluated once per execution of that level -- except ``FOR_ITER``, which carries the
position of the iterable but runs once per iteration plus once per entry, and is
attributed to its loop.  Within a level the only backward jumps are the loop's own
(a backward jump belongs to the loop that owns its target), so an instruction
between the target of such a jump and the jump (the *cycle*) runs at most once per
iteration plus once per entry (the final ``FOR_ITER`` or loop test), and any other
instruction of the level at most once per entry.  An instruction inside the cycle
of a loop but attributed to an enclosing level (for instance one without a source
position, such as the entry point of an exception handler) is attributed to that
loop.  ``tests/test_opcount.py`` checks the resulting bound against executions
measured with ``sys.monitoring``.  A call therefore costs at most

    outside + sum over loops (cycle x (iterations + entries) + entry x entries)
            + callees.

Generator expressions are separate code objects; :func:`profile` lists them as
nested functions.  The profiles are recorded in ``PROFILES`` of
``cake_cutting.opbounds`` and ``tests/test_opbounds.py`` checks that they match
the code, so that a change of the code cannot silently invalidate the bound.
"""

from __future__ import annotations

import ast
import dis
import inspect
import textwrap
import types
from typing import Dict, List, Optional, Tuple

LOOP_NODES = (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.YieldFrom)


def _span(node: ast.AST) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return (node.lineno, node.col_offset), (node.end_lineno, node.end_col_offset)


def _contains(span, pos) -> bool:
    (l0, c0), (l1, c1) = span
    return (l0, c0) <= pos <= (l1, c1)


class _Loop:
    def __init__(self, node: ast.AST, exclude: Optional[ast.AST]) -> None:
        self.node = node
        self.span = _span(node)
        self.exclude = _span(exclude) if exclude is not None else None
        self.kind = type(node).__name__
        self.line = node.lineno
        self.cycle = 0
        self.entry = 0

    def owns(self, pos) -> bool:
        return _contains(self.span, pos) and not (self.exclude is not None and _contains(self.exclude, pos))


def _loops_of(tree: ast.AST, line_offset: int, col_offset: int) -> List[_Loop]:
    found: List[_Loop] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, LOOP_NODES):
            if isinstance(node, ast.For):
                exclude = node.iter
            elif isinstance(node, (ast.While, ast.YieldFrom)):
                exclude = None
            else:
                exclude = node.generators[0].iter
            found.append(_Loop(node, exclude))
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.GeneratorExp)):
                continue   # separate code objects
            visit(child)

    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for s in stmt.body:
                if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    visit(s)
            # default values and decorators are evaluated at definition time, not per call
    for lp in found:
        (a, b), (c, d) = lp.span
        lp.span = ((a + line_offset, b + col_offset), (c + line_offset, d + col_offset))
        if lp.exclude is not None:
            (a, b), (c, d) = lp.exclude
            lp.exclude = ((a + line_offset, b + col_offset), (c + line_offset, d + col_offset))
        lp.line += line_offset
    return found


def _source_tree(obj):
    src, first = inspect.getsourcelines(obj)
    indent = len(src[0]) - len(src[0].lstrip())
    return ast.parse(textwrap.dedent("".join(src))), first, indent


def profile(fn) -> Dict[str, object]:
    """``{"outside": n0, "loops": [(line, kind, cycle, entry), ...], "nested": [names]}``
    for a function, method, property or the code object of an inner ``def``.

    ``cycle + entry`` is the number of instructions attributed to that loop level
    itself (not to loops nested in it): ``cycle`` of them lie on the loop's cycle and
    run at most once per iteration plus once per entry, ``entry`` of them at most
    once per entry."""
    if isinstance(fn, property):
        fn = fn.fget
    if isinstance(fn, (staticmethod, classmethod)):
        fn = fn.__func__
    if isinstance(fn, types.CodeType):
        code = fn
    else:
        fn = inspect.unwrap(fn)
        code = fn.__code__
    tree, first, indent = _source_tree(code)
    loops = _loops_of(tree, first - 1, indent)
    outside = 0
    nested: List[str] = []
    instrs = list(dis.get_instructions(code))
    owner_of = []
    for ins in instrs:
        pos = ins.positions
        owner = None
        if pos is not None and pos.lineno is not None:
            p = (pos.lineno, pos.col_offset if pos.col_offset is not None else 0)
            for lp in loops:
                if lp.owns(p) and (owner is None or _contains(owner.span, (lp.span[0]))):
                    owner = lp
        owner_of.append(owner)
        if isinstance(ins.argval, types.CodeType):
            nested.append(ins.argval.co_name)
    # FOR_ITER carries the position of the iterable, which is excluded from the loop's
    # span; it belongs to the loop (it runs once per iteration plus once per entry):
    # the loop whose excluded iterable span is the smallest one containing it
    for idx, ins in enumerate(instrs):
        if ins.opname != "FOR_ITER" or ins.positions is None or ins.positions.lineno is None:
            continue
        p = (ins.positions.lineno, ins.positions.col_offset or 0)
        best = None
        for lp in loops:
            if lp.exclude is not None and _contains(lp.exclude, p):
                (l0, c0), (l1, c1) = lp.exclude
                size = (l1 - l0, c1 - c0)
                if best is None or size < best[0]:
                    best = (size, lp)
        if best is not None:
            owner_of[idx] = best[1]
            # EXTENDED_ARG prefixes of the FOR_ITER (long loop bodies) go with it
            j = idx - 1
            while j >= 0 and instrs[j].opname == "EXTENDED_ARG":
                owner_of[j] = best[1]
                j -= 1
    # the cycle of a loop: the instructions between the target of one of its backward
    # jumps and that jump; a backward jump belongs to the loop that owns its target
    # (the loop's FOR_ITER, while test or SEND; EXTENDED_ARG prefixes of the target are
    # skipped), or else to its own owner
    offset_index = {ins.offset: i for i, ins in enumerate(instrs)}
    regions: Dict[int, List[Tuple[int, int]]] = {}
    for idx, ins in enumerate(instrs):
        if not ins.opname.startswith("JUMP_BACKWARD"):
            continue
        tgt = offset_index.get(ins.argval)
        while tgt is not None and tgt < len(instrs) and instrs[tgt].opname == "EXTENDED_ARG":
            tgt += 1
        lp = owner_of[tgt] if tgt is not None and owner_of[tgt] is not None else owner_of[idx]
        if lp is not None:
            regions.setdefault(id(lp), []).append((ins.argval, ins.offset))
    # an instruction inside a cycle of a loop L but attributed to a level enclosing L
    # (or to no loop), e.g. one without a source position, runs at most once per
    # iteration of L: it is attributed to the innermost such L
    for idx, ins in enumerate(instrs):
        cur = owner_of[idx]
        best = None
        for lp in loops:
            if lp is cur:
                continue
            if cur is not None and not (_contains(cur.span, lp.span[0]) and _contains(cur.span, lp.span[1])):
                continue          # lp is not nested in the current owner
            for a, b in regions.get(id(lp), ()):
                if a <= ins.offset <= b and (best is None or b - a < best[0]):
                    best = (b - a, lp)
        if best is not None:
            owner_of[idx] = best[1]
    for ins, owner in zip(instrs, owner_of):
        if owner is None:
            outside += 1
            continue
        if any(a <= ins.offset <= b for a, b in regions.get(id(owner), ())):
            owner.cycle += 1
        else:
            owner.entry += 1
    return {"outside": outside,
            "loops": [(lp.line, lp.kind, lp.cycle, lp.entry) for lp in sorted(loops, key=lambda l: l.span)],
            "nested": nested}


def _nested_codes(code: types.CodeType):
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield const


def _add_code(out: Dict[str, dict], key: str, code: types.CodeType, prof: Dict[str, object]) -> None:
    out[key] = prof
    for sub in _nested_codes(code):
        skey = f"{key}.<{sub.co_name.strip('<>')}@{sub.co_firstlineno}>"
        if sub.co_name.startswith("<"):
            # a lambda or generator expression: one call (lambda) or one element of the
            # iterable (generator expression with one ``for`` clause) executes every
            # instruction at most once; ``loops`` counts the loops it contains
            # (a lambda with a comprehension, a generator expression with several
            # ``for`` clauses), for which this does not hold and the bound is by hand
            _add_code(out, skey, sub, {"total": instruction_count(sub), "loops": loop_heads(sub)})
        else:
            _add_code(out, skey, sub, profile(sub))


MODULES = ("seq", "pieces", "bookkeeping", "equilibrium", "subcore", "allocation", "base_cases",
           "simultaneous", "main", "bounds")


def generate() -> Dict[str, dict]:
    """Profiles of every function and method defined in :data:`MODULES`."""
    import importlib
    out: Dict[str, dict] = {}
    for name in MODULES:
        mod = importlib.import_module(f"cake_cutting.{name}")
        items = []
        for attr, obj in vars(mod).items():
            if inspect.isfunction(obj) and obj.__module__ == mod.__name__:
                items.append((obj.__code__.co_firstlineno, obj.__qualname__, obj))
            elif inspect.isclass(obj) and obj.__module__ == mod.__name__:
                for mname, mobj in vars(obj).items():
                    f = mobj
                    if isinstance(f, property):
                        f = f.fget
                    if isinstance(f, (staticmethod, classmethod)):
                        f = f.__func__
                    if inspect.isfunction(f):
                        items.append((f.__code__.co_firstlineno, f.__qualname__, f))
        for _, qual, f in sorted(items, key=lambda t: t[0]):
            _add_code(out, f"{name}.{qual}", f.__code__, profile(f))
    return out


def nested_code(fn, name: str) -> types.CodeType:
    """The nested code object ``name`` (a lambda, a generator expression, an inner def)."""
    fn = inspect.unwrap(fn.fget if isinstance(fn, property) else fn)
    for const in fn.__code__.co_consts:
        if isinstance(const, types.CodeType) and const.co_name == name:
            return const
    raise KeyError(name)


def loop_heads(code: types.CodeType) -> int:
    """The number of distinct targets of backward jumps (EXTENDED_ARG prefixes skipped):
    the number of loops of a code object without loop statements of its own."""
    instrs = list(dis.get_instructions(code))
    index = {ins.offset: i for i, ins in enumerate(instrs)}
    heads = set()
    for ins in instrs:
        if ins.opname == "JUMP_BACKWARD":
            i = index[ins.argval]
            while instrs[i].opname == "EXTENDED_ARG":
                i += 1
            heads.add(instrs[i].offset)
    return len(heads)


def instruction_count(code: types.CodeType) -> int:
    return sum(1 for _ in dis.get_instructions(code))


HEADER = '''"""Frozen instruction profiles of the package (generated by ``python -m
cake_cutting.opcount``; checked against the code by ``tests/test_opbounds.py``).
Keys: ``module.qualname``; nested code objects as ``...<name@line>``: inner functions
with their own profile, lambdas and generator expressions with their total
instruction count."""

'''


def write_profiles(path=None) -> None:
    import pathlib
    import pprint
    if path is None:
        path = pathlib.Path(__file__).with_name("_opprofiles.py")
    text = HEADER + "PROFILES = " + pprint.pformat(generate(), width=100, sort_dicts=False) + "\n"
    pathlib.Path(path).write_text(text)


if __name__ == "__main__":
    write_profiles()
