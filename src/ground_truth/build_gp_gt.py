import argparse
import json
import glob
import sys

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_score


def _records_to_frame(records):
    """Named DataFrame of value columns only (drops each label's '_var'
    uncertainty companion), or None if nothing was ever recorded, or if only
    some observations carry a given value (would otherwise leave NaN gaps)."""
    df = pd.DataFrame(records)
    value_cols = [c for c in df.columns if not c.endswith("_var")]
    df = df[value_cols]
    return df if not df.empty and df.notna().all(axis=None) else None


def main():
    # Labels like "Tool Wear (μm)" get printed, and stdout defaults to cp1252 on
    # Windows, which cannot encode them.
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Fit a Gaussian process ground-truth surrogate for a collected "
                    "pybo run's objectives.")
    parser.add_argument("--root-dir", default="",
                        help="Root folder of the Bayesian optimization run, searched "
                             "recursively for experiment.json files (default: %(default)s)")
    parser.add_argument("--kernel", choices=["rbf", "matern"], default="rbf",
                        help="Kernel family (default: %(default)s)")
    parser.add_argument("--nu", type=float, default=1.5,
                        help="Matern smoothness parameter, only used with --kernel matern "
                             "(default: %(default)s)")
    parser.add_argument("--n-restarts-optimizer", type=int, default=5,
                        help="Random restarts for the kernel hyperparameter optimizer "
                             "(default: %(default)s)")
    parser.add_argument("--positive", action="store_true",
                        help="Assume every observed objective value is physically "
                             "non-negative and fit log(y) instead of y, guaranteeing "
                             "strictly positive predictions. Off by default: only turn "
                             "this on when the objectives can never be zero or negative. "
                             "Applies to the objectives only, never to constraints or "
                             "trackers.")
    args = parser.parse_args()

    root_dir = args.root_dir

    paths = glob.glob(f"{root_dir}/**/*experiment.json", recursive=True)
    if not paths:
        raise SystemExit(f"No experiment.json found under {root_dir} (searched recursively)")
    parameter_records = []
    objective_records = []
    constraint_records = []
    tracker_records = []
    for path in paths:
        # utf-8 explicitly: the default on Windows is cp1252, which mangles labels
        # like "Tool Wear (μm)" on the way in.
        with open(path, encoding="utf-8") as f:
            json_file = json.load(f)
        for observation in json_file["data"]:
            parameter_records.append(observation["parameters"])
            objective_records.append(observation["objectives"])
            constraint_records.append(observation["constraints"])
            tracker_records.append(observation["trackers"])

    X = _records_to_frame(parameter_records)
    Y_obj = _records_to_frame(objective_records)
    Y_con = _records_to_frame(constraint_records)
    Y_trk = _records_to_frame(tracker_records)

    if args.kernel == "matern":
        base_kernel = Matern(length_scale=np.ones(X.shape[1]), nu=args.nu)
    else:
        base_kernel = RBF(length_scale=np.ones(X.shape[1]))
    kernel = ConstantKernel() * base_kernel + WhiteKernel()

    def _maybe_positive(regressor, positive):
        """Wrapped in log(y)/exp(...) when positive is set, guaranteeing strictly
        positive predictions; passed through unchanged otherwise. Only ever set
        for the objectives, and only when every observed value is > 0."""
        if not positive:
            return regressor
        return TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)

    for block, Y in [("objectives", Y_obj), ("constraints", Y_con), ("trackers", Y_trk)]:
        print(f"\n=== {block} ===")
        if Y is None:
            print("none recorded (or only recorded on some observations), skipped")
            continue

        # --positive is an objectives-only assumption. A constraint's sign is its
        # feasibility boundary (f(X) >= 0 is feasible), so log(y) would be
        # undefined on every infeasible observation and, on all-feasible data,
        # would build a surrogate that can never predict infeasibility at all.
        # Trackers carry no guaranteed sign either. Both are fit in raw space.
        positive = args.positive and block == "objectives"

        # MultiOutputRegressor fits one GP per column with its own kernel
        # hyperparameters (GaussianProcessRegressor alone would fit a single kernel
        # shared across all columns, which is wrong here since different
        # objectives can have very different smoothness/noise).
        pipeline = _maybe_positive(
            make_pipeline(
                StandardScaler(),
                MultiOutputRegressor(
                    GaussianProcessRegressor(
                        kernel=kernel, normalize_y=True,
                        n_restarts_optimizer=args.n_restarts_optimizer,
                    )
                ),
            ),
            positive,
        )
        pipeline.fit(X, Y)
        print(pipeline.score(X, Y))  # R^2 in original (exp-transformed if --positive) scale
        if positive:
            print((pipeline.predict(X) > 0).all())  # sanity check: every prediction positive

        # A GP has no "degree" to sweep — kernel hyperparameters (length scales,
        # noise level) are already optimized per column via marginal likelihood.
        # Print the fitted kernel per column as the equivalent diagnostic to the
        # polynomial version's coefficient table.
        regressor = pipeline.regressor_ if positive else pipeline
        gp_regressors = regressor.named_steps["multioutputregressor"].estimators_
        for label, gp in zip(Y.columns, gp_regressors):
            print(label, "->", gp.kernel_)

        # Cross-validated R^2 per column, same metric as the polynomial version's
        # degree sweep, for direct comparison against it. Single-output GPs here
        # (no MultiOutputRegressor needed) since each fold only ever scores one
        # column at a time.
        r2_per_label = {}
        for label in Y.columns:
            single_pipeline = _maybe_positive(
                make_pipeline(
                    StandardScaler(),
                    GaussianProcessRegressor(
                        kernel=kernel, normalize_y=True,
                        n_restarts_optimizer=args.n_restarts_optimizer,
                    ),
                ),
                positive,
            )
            scores = cross_val_score(single_pipeline, X, Y[label], cv=5, scoring="r2")
            r2_per_label[label] = scores.mean()
        print(pd.Series(r2_per_label, name="R2"))


if __name__ == "__main__":
    main()
