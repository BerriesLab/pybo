import argparse
import json
import glob

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
    parser = argparse.ArgumentParser(
        description="Fit a Gaussian process ground-truth surrogate for a collected "
                    "pybo run's objectives.")
    parser.add_argument("--root-dir", default="/Users/berri/Repositories/pybo/data/vformac",
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
                             "this on when the objectives can never be zero or negative.")
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
        with open(path) as f:
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

    def _maybe_positive(regressor):
        """Wrapped in log(y)/exp(...) when --positive is set, guaranteeing strictly
        positive predictions; passed through unchanged otherwise. Only turn
        --positive on when every observed objective value is > 0."""
        if not args.positive:
            return regressor
        return TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)

    # MultiOutputRegressor fits one GP per objective column with its own kernel
    # hyperparameters (GaussianProcessRegressor alone would fit a single kernel
    # shared across all objectives, which is wrong here since different
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
        )
    )
    pipeline.fit(X, Y_obj)
    print(pipeline.score(X, Y_obj))  # R^2 in original (exp-transformed if --positive) scale
    print((pipeline.predict(X) > 0).all())  # sanity check: every prediction positive

    # A GP has no "degree" to sweep — kernel hyperparameters (length scales,
    # noise level) are already optimized per objective via marginal likelihood.
    # Print the fitted kernel per objective as the equivalent diagnostic to the
    # polynomial version's coefficient table.
    regressor = pipeline.regressor_ if args.positive else pipeline
    gp_regressors = regressor.named_steps["multioutputregressor"].estimators_
    for objective, gp in zip(Y_obj.columns, gp_regressors):
        print(objective, "->", gp.kernel_)

    # Cross-validated R^2 per objective, same metric as the polynomial version's
    # degree sweep, for direct comparison against it. Single-output GPs here
    # (no MultiOutputRegressor needed) since each fold only ever scores one
    # objective column at a time.
    r2_per_objective = {}
    for objective in Y_obj.columns:
        single_pipeline = _maybe_positive(
            make_pipeline(
                StandardScaler(),
                GaussianProcessRegressor(
                    kernel=kernel, normalize_y=True,
                    n_restarts_optimizer=args.n_restarts_optimizer,
                ),
            )
        )
        scores = cross_val_score(single_pipeline, X, Y_obj[objective], cv=5, scoring="r2")
        r2_per_objective[objective] = scores.mean()
    print(pd.Series(r2_per_objective, name="R2"))


if __name__ == "__main__":
    main()
