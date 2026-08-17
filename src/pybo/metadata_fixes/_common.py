import json
import math
from pathlib import Path


def find_experiments(roots: str | Path | list[str | Path]) -> list[Path]:
    """Every experiment.json at or under one root or several, deduplicated and sorted.

    Recursive per root, so one step folder and a whole campaign are the same command,
    and any step_NNN/ or step_NNN_repYY/ layout works — the same glob
    build_polynomial_gt uses. A single root is accepted directly, not just a
    one-element list, so existing single-root callers pass what they always have.
    """
    if isinstance(roots, (str, Path)):
        roots = [roots]
    found = {path for root in roots for path in Path(root).glob("**/*experiment.json")}
    if not found:
        where = roots[0] if len(roots) == 1 else roots
        raise SystemExit(f"No experiment.json found under {where} (searched recursively)")
    return sorted(found)


def split_roots(value: str) -> list[str]:
    """One CLI argument, comma-separated, as the list find_experiments wants.

    A bare path with no comma is a one-element list, so a single root needs no
    special-casing at the call site - "data/iformac" and "data/iformac,data/vformac"
    are the same code path.
    """
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve(node, parts: list[str], scope: dict, create: bool, strict: bool = False):
    """(parent, key, scope) for every place a dot path lands.

    "*" fans out over a list, which is how the observations inside "data" are
    reached, and the element it lands on becomes the scope an --expr is evaluated
    against. A path that never crosses a list keeps the whole file as its scope.

    Only the last part of the path is ever a key that may not exist yet — that is
    the one being written. Everything before it has to be there already, and with
    strict a group that is not raises rather than being created: a mistyped
    "bjectives" is a typo, not a request for a new group. create overrides that for
    when a genuinely new group is wanted."""
    if not parts:
        return
    part, rest = parts[0], parts[1:]

    if part == "*":
        if isinstance(node, list):
            for item in node:
                yield from resolve(item, rest, item, create, strict)
        elif strict:
            raise KeyError(f"'*' needs a list to fan out over, found "
                           f"{type(node).__name__}")
        return

    if not isinstance(node, dict):
        if strict:
            raise KeyError(f"'{part}' needs a dict to live in, found "
                           f"{type(node).__name__}")
        return
    if not rest:
        yield node, part, scope
        return
    if part not in node:
        if create:
            node[part] = {}
        elif strict:
            raise KeyError(f"'{part}' is not there (available: "
                           f"{', '.join(node) or 'nothing'})")
        else:
            return
    yield from resolve(node[part], rest, scope, create, strict)


def coerce(value: str):
    """A CLI string as the JSON scalar it looks like. "null" included, so a _var
    companion can be cleared back to null rather than set to the text "null"."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in ("null", "none"):
        return None
    try:
        return float(value)
    except ValueError:
        return value


def evaluate(expr: str, scope: dict):
    """An expression against the observation the path landed on, plus math. Labels
    carry spaces and parentheses, so they are reached as objectives["Tool Wear (μm)"]
    rather than by attribute."""
    return eval(expr, {"math": math}, dict(scope))


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, doc: dict) -> None:
    """ensure_ascii=False and utf-8, matching what is already on disk: the files hold
    a real mu in labels like "Tool Wear (μm)", and escaping it would rewrite every
    line of every file it touches."""
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")