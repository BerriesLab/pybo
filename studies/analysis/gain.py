"""What the optimizer bought, and how fast: gain, efficiency and time-to-target.

Three scalars per replicate, reduced from the metric trace the runs already wrote. With
m(n) the metric after n evaluations (hypervolume, or best value for a single objective),
n_0 the initial design size and m* the problem's optimum:

    gamma   100 (m_f - m_0) / |m_0|            relative gain over the initial design, %
    gamma~  (m_f - m_0) / (m* - m_0)           fraction of the achievable gap closed, 0..1
    eta     gamma / (n_c - n_0)                average gain per evaluation, %/eval
    it_tau  min{n : m(n) >= m_0 + tau (m* - m_0)} - n_0     iterations to a stated target

gamma is unbounded and inflated when m_0 is small, which is why gamma~ is reported
beside it whenever the problem declares an optimum. n_c is where the run's own
convergence rule fires (--patience/--tol, mirroring BayesianOptimizer.is_converged), so
eta and it_tau answer different questions: eta averages over a horizon the run chose,
it_tau over one the reader chose. it_tau is the honest speed measure — a run that stalls
early converges "fast" by n_c but simply never reaches its target.

    python -m studies.analysis.gain --study data/branin_currin_study
    python -m studies.analysis.gain --study data/A --study data/B \
        --label baseline --label more-initial --tau 0.9

Metrics are computed per replicate and then aggregated per study: mean +- std for the
gains, median for it_tau over the replicates that reached the target, with the reached
count printed beside it. Replicates that never reach it are censored, not dropped —
averaging only the ones that made it would flatter the optimizers that mostly fail.
"""
import numpy as np
import pandas as pd

from studies.analysis.cli import build_gain_parser
from studies.analysis.utils import discover_trials, metric_frame


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

    agg = []
    for study, g in per_run.groupby("study", sort=False):
        row = {"study": study, "replicates": len(g),
               "gamma": f"{g['gamma'].mean():.4g} +- {g['gamma'].std(ddof=1):.3g}",
               "gamma_norm": f"{g['gamma_norm'].mean():.4g} +- {g['gamma_norm'].std(ddof=1):.3g}",
               "eta": f"{g['eta'].mean():.4g} +- {g['eta'].std(ddof=1):.3g}",
               "converged": f"{int(g['converged'].sum())}/{len(g)}"}
        for tau in taus:
            col = g[f"it{tau:g}"]
            hit = col.notna().sum()
            row[f"it{tau:g}"] = f"{col.median():.4g} ({hit}/{len(g)})" if hit else f"- (0/{len(g)})"
        agg.append(row)

    print("\nPer study - mean +- std, it_tau as median (reached/total):\n")
    print(pd.DataFrame(agg).to_string(index=False))

    out_dir = args.output_dir or args.study[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(out_dir / "gain.csv", index=False)
    print("\nSaved", out_dir / "gain.csv")


if __name__ == "__main__":
    main()