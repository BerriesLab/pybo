"""Constraint evaluation shared by the campaign plots.

A constraint is a boolean expression over an observation's named columns - the labels the
problem definition gives its parameters, objectives, constraints and trackers - passed
whole on the CLI (repeatable via ``--constraint``). Because the expression is evaluated
with asteval, all of the following work in one mechanism:

  - single-key bound      ``Branin <= 50``
  - general linear        ``Branin + 2*Currin <= 80``
  - nonlinear             ``Branin**2 + Currin**2 <= 100``
  - equality as tolerance ``abs(Currin - 5) <= 1``

An observation is *feasible* iff every constraint evaluates truthy against its columns. A
missing key, a None value, or any evaluation error makes the observation infeasible.

Expressions run in a sandbox: ``minimal`` asteval (arithmetic, comparisons and the math
helpers - abs, min, max, sqrt, sin, log, ...), with statement constructs (import,
comprehensions, lambda, assignment, loops, with, ...) and the ``open`` builtin removed.
No attribute access to dunders, no imports, no I/O.
"""
from asteval import Interpreter

# Statement-level constructs we never want in a constraint expression.
_DISABLED = (
    "import", "importfrom", "with", "functiondef", "lambda", "delete",
    "augassign", "while", "for", "try", "raise", "print", "assert",
    "listcomp", "dictcomp", "setcomp",
)


class ConstraintError(ValueError):
    """A constraint expression failed to parse."""


class CompiledConstraints:
    """A set of pre-parsed constraint expressions sharing one sandboxed interpreter.

    Column values are bound into the symbol table per observation; everything injected
    for one observation is cleared before the next, so a key present in one cannot leak
    into another that lacks it.
    """

    def __init__(self):
        self._aeval = Interpreter(minimal=True, use_numpy=False)
        self._aeval.symtable.pop("open", None)
        for key in _DISABLED:
            if key in self._aeval.config:
                self._aeval.config[key] = False
        # Snapshot the safe builtins so per-observation injections can be cleared
        # without disturbing them.
        self._base_keys = set(self._aeval.symtable)
        self._items = []  # list of (expr_text, parsed_node)

    def add(self, expr_text: str) -> None:
        try:
            node = self._aeval.parse(expr_text)
        except SyntaxError as exc:
            self._aeval.error = []
            raise ConstraintError(f"Invalid constraint {expr_text!r}: {exc}") from None
        if self._aeval.error:
            err = self._aeval.error[0]
            self._aeval.error = []
            raise ConstraintError(f"Invalid constraint {expr_text!r}: {err.msg}")
        self._items.append((expr_text, node))

    def is_feasible(self, values: dict) -> bool:
        if not self._items:
            return True
        aeval = self._aeval
        table = aeval.symtable
        for key in [k for k in table if k not in self._base_keys]:
            del table[key]
        table.update(values)
        for _expr_text, node in self._items:
            aeval.error = []
            try:
                result = aeval.run(node)
            except Exception:  # noqa: BLE001 - any failure means infeasible, not a crash
                return False
            if aeval.error or not bool(result):
                return False
        return True


def parse_constraints(specs) -> CompiledConstraints:
    """Compile `specs` (boolean expression strings) into a reusable CompiledConstraints.

    Blank entries are skipped; a syntactically invalid expression raises ConstraintError.
    """
    compiled = CompiledConstraints()
    for spec in specs or []:
        spec = spec.strip()
        if spec:
            compiled.add(spec)
    return compiled


def feasible_mask(df, specs) -> "list[bool]":
    """One boolean per row of `df`, True where every constraint holds.

    Takes the frame rather than a dict per caller so the campaign scripts share both the
    parsing and the row iteration.
    """
    compiled = parse_constraints(specs)
    if not specs:
        return [True] * len(df)
    return [compiled.is_feasible(row) for row in df.to_dict("records")]
