"""Score a campaign: how much the optimizer improved on the initial design, and how fast.

Reads every summary.json under each --study and reduces each run's metric trace to a few
numbers. Prints one row per replicate and one summary row per study, and writes the same
figures to gain.json: per study the mean/std of each gain and the per-target medians,
with the replicate rows nested under it. Undefined values are null there, not NaN.

The metric is hypervolume, or best value for a single-objective problem.

COLUMNS

  gamma       improvement over the initial design, in %
  gamma_norm  how much of the distance from the initial design to the known optimum was
              covered: 0 = no better than the initial design, 1 = optimum reached
  eta         gamma divided by the evaluations spent getting there, in % per evaluation
  it0.5       evaluations, past the initial design, needed to cover 50% of that distance;
  it0.9       likewise 90% and 99%. Set your own targets with --tau.
  it0.99
  n_c         evaluation at which the run stopped improving (--patience/--tol)

  In the per-study summary, gains are mean +- std over replicates and each it column is
  a median plus how many replicates got there at all: "15 (1/3)" means one replicate of
  three reached that target, after 15 evaluations. The other two are not averaged in;
  discarding them would make an optimizer that usually fails look fast.

READING THEM

  gamma is the headline number but it blows up when the initial design scores near zero
  (a common case: its Pareto front dominates almost nothing). Prefer gamma_norm, which
  stays in 0..1 - but it needs a known optimum, from the problem definition or --optimum.

  eta and it_tau both measure speed, against different finish lines. eta uses the run's
  own stopping point, so a run that stalls early looks fast; it_tau uses a target you
  chose, and a run that stalls simply never reaches it. Trust it_tau.

EXAMPLES

  python -m studies.analysis.gain --study data/branin_currin_study
  python -m studies.analysis.gain --study data/A --study data/B --label baseline --label more-initial
  python -m studies.analysis.gain --study data/A --tau 0.9 --tau 0.99 --optimum 59.36

DEFINITIONS

  With m(n) the metric after n evaluations, n_0 the initial design size, m* the optimum:

    gamma      = 100 (m_f - m_0) / |m_0|
    gamma_norm = (m_f - m_0) / (m* - m_0)
    eta        = gamma / (n_c - n_0)
    it_tau     = min{n : m(n) >= m_0 + tau (m* - m_0)} - n_0
"""
import json

import numpy as np
import pandas as pd

from studies.analysis.cli import build_gain_parser
from studies.analysis.utils import discover_trials, metric_frame


def jsonable(value):
    """The report, with numpy scalars unwrapped and non-finite floats as None.

    Both matter for a file another tool will read: numpy's int64/bool_ are not JSON
    types, and NaN - an undefined gamma, a target no replicate reached - has no JSON
    spelling, so it becomes null rather than a bare NaN token no strict parser accepts.
    """
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    args = build_gain_parser(description=__doc__).parse_args()
    taus = sorted(args.tau or [0.5, 0.9, 0.99])

    trials = discover_trials(args.study, args.label)
    df = metric_frame(trials)

    # Unlike a plot, these numbers are silently meaningless across different problems:
    # gamma~ and it_tau are fractions of one gap to one optimum. Studies must be
    # configurations of the same problem, not different benchmarks.
    metric = df["metric"].iloc[0]
    if df["metric"].nunique() > 1:
        raise SystemExit("These studies do not share a metric (hypervolume vs best value), "
                         "so there is no common gap to measure gains against. Compare "
                         "studies of one problem.")

    problem = trials[0].summary["problem"]
    declared = problem.get("max_hv") if metric == "hypervolume" else problem.get("best_value")
    for t in trials:
        other = t.summary["problem"]
        if (other.get("max_hv") if metric == "hypervolume" else other.get("best_value")) != declared:
            raise SystemExit(f"{t.path} declares a different optimum than {trials[0].path}. "
                             f"Compare studies of one problem, or pass --optimum.")

    # Work in maximization space, so a minimized best value improves upwards like a
    # hypervolume does and every comparison below reads the same way.
    sense = -1.0 if (metric == "best_value" and trials[0].objectives[0]["to_minimize"]) else 1.0
    df["signed"] = sense * df["value"]
    if args.optimum is not None:
        optimum, source = sense * args.optimum, "--optimum"
    elif declared is not None:
        optimum, source = sense * declared, "the problem definition"
    else:
        optimum, source = np.nanmax(df["signed"]), "the best value reached in this campaign"

    rows = []
    for (study, run), g in df.groupby(["study", "run"], sort=False):
        g = g.sort_values("evaluations")
        m, n = g["signed"].to_numpy(dtype=float), g["evaluations"].to_numpy(dtype=float)
        m0, n0, m_final = m[0], n[0], m[-1]

        # Convergence as the run itself would have called it: `patience` consecutive
        # improvements all below `tol`. Never converged means the horizon is the budget,
        # which makes eta a lower bound rather than a missing value.
        n_c, converged = n[-1], False
        for i in range(args.patience - 1, len(m)):
            window = np.diff(m[i - args.patience + 1:i + 1])
            if window.size and np.all(np.abs(window) < args.tol):
                n_c, converged = n[i], True
                break

        gap = optimum - m0
        row = {"study": study, "run": run, "n_initial": int(n0),
               "m_initial": sense * m0, "m_final": sense * m_final,
               "gamma": 100.0 * (m_final - m0) / abs(m0) if np.isfinite(m0) and m0 != 0 else np.nan,
               "gamma_norm": (m_final - m0) / gap if np.isfinite(gap) and gap > 0 else np.nan,
               "n_c": int(n_c), "converged": converged}
        row["eta"] = row["gamma"] / (n_c - n0) if n_c > n0 else np.nan
        for tau in taus:
            # Censored on purpose: a run that never clears the target has no it_tau, and
            # the aggregate below reports how many did rather than averaging the rest.
            reached = np.flatnonzero(m >= m0 + tau * gap) if np.isfinite(gap) and gap > 0 else []
            row[f"it{tau:g}"] = n[reached[0]] - n0 if len(reached) else np.nan
        rows.append(row)

    per_run = pd.DataFrame(rows)

    print(f"\nMetric: {metric} ({'maximized' if sense > 0 else 'minimized'})")
    print(f"Optimum m* = {sense * optimum:.6g}, from {source}.")
    if source.startswith("the best"):
        print("  ! No declared optimum: targets are relative to the best this campaign "
              "reached,\n    so gamma~ and it_tau are not comparable against another "
              "campaign's numbers.")
    print(f"Convergence: {args.patience} iterations improving by less than {args.tol:g}.\n")

    print(per_run.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    agg, studies = [], []
    for study, g in per_run.groupby("study", sort=False):
        row = {"study": study, "replicates": len(g),
               "gamma": f"{g['gamma'].mean():.4g} +- {g['gamma'].std(ddof=1):.3g}",
               "gamma_norm": f"{g['gamma_norm'].mean():.4g} +- {g['gamma_norm'].std(ddof=1):.3g}",
               "eta": f"{g['eta'].mean():.4g} +- {g['eta'].std(ddof=1):.3g}",
               "converged": f"{int(g['converged'].sum())}/{len(g)}"}
        entry = {"study": study, "replicates": len(g),
                 "converged": int(g["converged"].sum()),
                 "targets": {}, "runs": []}
        for name in ("gamma", "gamma_norm", "eta"):
            entry[name] = {"mean": g[name].mean(), "std": g[name].std(ddof=1)}
        for tau in taus:
            col = g[f"it{tau:g}"]
            hit = int(col.notna().sum())
            row[f"it{tau:g}"] = f"{col.median():.4g} ({hit}/{len(g)})" if hit else f"- (0/{len(g)})"
            # reached is what makes the median readable: a median over 1 of 5 replicates
            # is not the same claim as a median over 5.
            entry["targets"][f"{tau:g}"] = {"median": col.median(), "reached": hit,
                                            "total": len(g)}
        entry["runs"] = g.drop(columns="study").to_dict("records")
        agg.append(row)
        studies.append(entry)

    print("\nPer study - mean +- std, it_tau as median (reached/total):\n")
    print(pd.DataFrame(agg).to_string(index=False))

    report = {"metric": metric, "sense": "maximized" if sense > 0 else "minimized",
              "optimum": sense * optimum, "optimum_source": source,
              "convergence": {"patience": args.patience, "tol": args.tol},
              "taus": taus, "studies": studies}

    out_dir = args.output_dir or args.study[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gain.json"
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(jsonable(report), file, indent=2, allow_nan=False)
    print("\nSaved", out_path)


if __name__ == "__main__":
    main()