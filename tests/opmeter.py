"""Counting executed bytecode instructions of the package, per call of chosen functions.

Used by ``tests/test_opbounds.py`` to compare the operation bounds of
:mod:`cake_cutting.opbounds` with what the code actually executes.  The meter
counts what the cost model counts as instructions (``cake_cutting.opcount``), with
the exclusions the proof makes:

* ``Interval.__hash__`` / ``Interval.__eq__`` (part of a dictionary operation, or an
  explicit comparison that the bounds charge separately),
* the first evaluation of a suffix view inside ``ValueTable.value`` (the prefix sum,
  prepaid by the view),
* the oracle and the valuations (a query costs one operation).

Built-in work per element is not measured, so a measured count below the bound is a
necessary condition for the bound, not a proof of it (the proof is ``PROOF.md``).
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Callable, Dict, Optional

import cake_cutting
from cake_cutting import bookkeeping, pieces, seq

PKG = os.path.dirname(os.path.abspath(cake_cutting.__file__))
SKIP_FILES = {"oracle.py", "valuations.py", "opcount.py", "opbounds.py", "_opprofiles.py"}

_IV_HASH = pieces.Interval.__hash__.__code__
_IV_EQ = pieces.Interval.__eq__.__code__
_PREFIX_SUM = seq.prefix_sum.__code__
_VALUE = bookkeeping.ValueTable.value.__code__


class Meter:
    """``with Meter({code: (name, params_fn, bound_fn)}) as m: ...``; afterwards
    ``m.results[name]`` lists ``(measured, bound, params)`` of every call."""

    def __init__(self, targets: Optional[Dict[object, tuple]] = None) -> None:
        self.targets = targets or {}
        self.count = 0
        self.excluded = 0
        self.active: Dict[object, tuple] = {}
        self.results = defaultdict(list)

    # -- tracing ------------------------------------------------------------
    def _is_pkg(self, code) -> bool:
        fn = code.co_filename
        return fn.startswith(PKG) and os.path.basename(fn) not in SKIP_FILES

    def _exclude(self, frame) -> bool:
        code = frame.f_code
        if code is _IV_HASH or code is _IV_EQ:
            return True
        back = frame.f_back
        if code is _PREFIX_SUM and back is not None and back.f_code is _VALUE:
            loc = back.f_locals
            if loc.get("k0", 0) or loc.get("head") is not None:
                return True
        return False

    def _global(self, frame, event, arg):
        code = frame.f_code
        if not self._is_pkg(code):
            return None
        if self.excluded or self._exclude(frame):
            self.excluded += 1
            frame.f_trace_opcodes = False
            return self._local_excluded
        frame.f_trace_opcodes = True
        target = self.targets.get(code)
        if target is not None and frame not in self.active:
            name, params_fn, bound_fn = target
            try:
                params = params_fn(frame.f_locals)
            except Exception:          # parameters not applicable: do not check this call
                params = None
            if params is not None:
                self.active[frame] = (name, self.count, params, bound_fn)
        return self._local

    def _local(self, frame, event, arg):
        if event == "opcode":
            self.count += 1
        elif event == "return":
            rec = self.active.get(frame)
            if rec is not None and not (frame.f_code.co_flags & 0x20 and arg is not None):
                del self.active[frame]
                name, start, params, bound_fn = rec
                self.results[name].append((self.count - start, bound_fn(*params), params))
        return self._local

    def _local_excluded(self, frame, event, arg):
        if event == "return":
            self.excluded -= 1
        return self._local_excluded

    def __enter__(self) -> "Meter":
        self._old = sys.gettrace()
        sys.settrace(self._global)
        return self

    def __exit__(self, *exc) -> None:
        sys.settrace(self._old)

    # -- reporting ----------------------------------------------------------
    def violations(self):
        return [(name, meas, bnd, params) for name, rows in self.results.items()
                for meas, bnd, params in rows if meas > bnd]

    def summary(self) -> Dict[str, tuple]:
        """``name -> (calls, max measured/bound)``."""
        out = {}
        for name, rows in self.results.items():
            out[name] = (len(rows), max(meas / bnd for meas, bnd, _ in rows) if rows else 0.0)
        return out
