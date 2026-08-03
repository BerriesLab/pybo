"""Shared constraint evaluation for the Pareto and hypervolume plots.

A constraint is a boolean expression over an experiment's result keys, passed
whole on the CLI (repeatable via ``--constraint``). Because the expression is
evaluated with asteval, all of the following work in one mechanism:

  - single-key bound      ``wear_microns <= 50``
  - general linear        ``wear_microns + 2*down_time_minutes <= 80``
  - nonlinear             ``wear_microns**2 + orbiting_time_minutes**2 <= 100``
  - equality as tolerance ``abs(wear_microns - 50) <= 1``

An experiment is *feasible* iff every constraint evaluates truthy against its
results. A missing key, a None value, or any evaluation error makes the
experiment infeasible.

Expressions run in a sandbox: ``minimal`` asteval (arithmetic, comparisons and
the math helpers — abs, min, max, sqrt, sin, log, …), with statement
constructs (import, comprehensions, lambda, assignment, loops, with, …) and the
``open`` builtin removed. No attribute access to dunders, no imports, no I/O."""
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
    """A set of pre-parsed constraint expressions sharing one sandboxed
    interpreter. Result values are bound into the symbol table per experiment;
    everything injected for one experiment is cleared before the next so a key
    present in one experiment cannot leak into another that lacks it."""

    def __init__(self):
        self._aeval = Interpreter(minimal=True, use_numpy=False)
        self._aeval.symtable.pop("open", None)
        for key in _DISABLED:
            if key in self._aeval.config:
                self._aeval.config[key] = False
        # Snapshot the safe builtins so per-experiment injections can be cleared
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

    def is_feasible(self, results: dict) -> bool:
        if not self._items:
            return True
        aeval = self._aeval
        st = aeval.symtable
        for k in [k for k in st if k not in self._base_keys]:
            del st[k]
        st.update(results)
        for _expr_text, node in self._items:
            aeval.error = []
            try:
                val = aeval.run(node)
            except Exception:
                return False
            if aeval.error or not bool(val):
                return False
        return True


def parse_constraints(specs) -> CompiledConstraints:
    """Compile ``specs`` (a list of boolean expression strings) into a reusable
    :class:`CompiledConstraints`. Blank entries are skipped; a syntactically
    invalid expression raises :class:`ConstraintError`."""
    compiled = CompiledConstraints()
    for spec in specs or []:
        spec = spec.strip()
        if spec:
            compiled.add(spec)
    return compiled


def is_feasible(results, constraints: CompiledConstraints) -> bool:
    """True iff `results` satisfies every compiled constraint."""
    return constraints.is_feasible(results)
