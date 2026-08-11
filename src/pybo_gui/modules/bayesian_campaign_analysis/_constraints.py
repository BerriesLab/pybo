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

A result key has to be a Python name to be referenced at all, and most of them are
not: a problem names its channels "Tool Wear", "Material Removal Rate". So a run of
words is read as one name, both in the expression and when the results are bound -
``Tool Wear <= 100`` works, and so does ``abs(Tool Wear - 50) <= 1``. The words asteval
reads as operators are left alone, so ``Tool Wear <= 100 and Orbiting Time >= 19`` still
parses as two comparisons. A key that is already a name ("con_00") is unaffected.

A key carrying anything else non-alphanumeric - "Orbiting Time Deviation (min)" - stays
unreferenceable, because no run of words can produce it.

Expressions run in a sandbox: ``minimal`` asteval (arithmetic, comparisons and
the math helpers — abs, min, max, sqrt, sin, log, …), with statement
constructs (import, comprehensions, lambda, assignment, loops, with, …) and the
``open`` builtin removed. No attribute access to dunders, no imports, no I/O."""
import re

from asteval import Interpreter

# Words asteval reads as operators or literals. A run of words is broken on these rather
# than swallowing them, or "100 and Orbiting Time" would become one name.
_RESERVED = frozenset(("and", "or", "not", "in", "is", "if", "else",
                       "True", "False", "None"))
_WORD_RUN = re.compile(r"[A-Za-z_]\w*(?:[ \t]+[A-Za-z_]\w*)+")


def _as_name(key: str) -> str:
    """A result key as the name an expression can reach it by."""
    return re.sub(r"[ \t]+", "_", key)


def _join_names(text: str) -> str:
    """The same joining, applied to what the user wrote."""
    def _run(match):
        joined, pending = [], []
        for word in match.group(0).split():
            if word in _RESERVED:
                if pending:
                    joined.append("_".join(pending))
                    pending = []
                joined.append(word)
            else:
                pending.append(word)
        if pending:
            joined.append("_".join(pending))
        return " ".join(joined)

    return _WORD_RUN.sub(_run, text)

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
        # Parsed from the joined form, reported from what the user actually typed.
        parsed_text = _join_names(expr_text)
        try:
            node = self._aeval.parse(parsed_text)
        except SyntaxError as exc:
            self._aeval.error = []
            raise ConstraintError(f"Invalid constraint {expr_text!r}: {exc}") from None
        if self._aeval.error:
            err = self._aeval.error[0]
            self._aeval.error = []
            raise ConstraintError(f"Invalid constraint {expr_text!r}: {err.msg}")
        self._items.append((expr_text, node))

    def __bool__(self) -> bool:
        """True if anything was actually compiled: blank specs leave nothing to enforce."""
        return bool(self._items)

    def is_feasible(self, results: dict) -> bool:
        if not self._items:
            return True
        aeval = self._aeval
        st = aeval.symtable
        for k in [k for k in st if k not in self._base_keys]:
            del st[k]
        # Names first, then the keys themselves, so a key that is already a name beats
        # another key's joined form if the two ever collide.
        st.update({_as_name(key): value for key, value in results.items()})
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
