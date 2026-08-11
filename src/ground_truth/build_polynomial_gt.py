import argparse
import json
import glob

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor


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
        description="Fit a polynomial ground-truth surrogate for a collected "
                    "pybo run's objectives.")
    parser.add_argument("--root-dir", default="",
                        help="Root folder of the Bayesian optimization run, searched "
                             "recursively for experiment.json files (default: %(default)s)")
    parser.add_argument("--degree", type=int, default=2,
                        help="Polynomial degree (default: %(default)s)")
    parser.add_argument("--positive", action="store_true",
                        help="Assume every observed objective value is physically "
                             "non-negative and fit log(y) instead of y, guaranteeing "
                             "strictly positive predictions. Off by default: only turn "
                             "this on when the objectives can never be zero or negative. "
                             "Applies to the objectives only, never to constraints or "
                             "trackers.")
    args = parser.parse_args()

    root_dir = args.root_dir
    degree = args.degree

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

        base_pipeline = make_pipeline(
            StandardScaler().set_output(transform="pandas"),
            PolynomialFeatures(degree=degree).set_output(transform="pandas"),
            LinearRegression(),
        )
        if positive:
            # An unconstrained polynomial fit has no way to know objectives can't go
            # negative. Fitting log(y) and inverse-transforming with exp() guarantees
            # strictly positive predictions everywhere, at the cost of the fit now
            # minimizing log-scale (roughly relative) error instead of absolute error.
            pipeline = TransformedTargetRegressor(
                regressor=base_pipeline, func=np.log, inverse_func=np.exp
            )
        else:
            pipeline = base_pipeline
        pipeline.fit(X, Y)
        regressor = pipeline.regressor_ if positive else pipeline
        feature_names = regressor.named_steps["polynomialfeatures"].get_feature_names_out()

        coef_table = pd.DataFrame(
            regressor.named_steps["linearregression"].coef_.T,
            index=feature_names,
            columns=Y.columns,
        )
        # PolynomialFeatures' bias column ("1") is collinear with LinearRegression's
        # own fit_intercept, so its coefficient is always 0 — replace it with the
        # actual intercept rather than showing that misleading zero. With --positive
        # these coefficients are in log-target space (multiplicative effects on y).
        coef_table = coef_table.rename(index={"1": "intercept"})
        coef_table.loc["intercept"] = regressor.named_steps["linearregression"].intercept_
        print(coef_table)
        print(f"R2 = {pipeline.score(X, Y)}")  # R^2 in original (exp-transformed) scale
        if positive:
            print(f"All positive: {(pipeline.predict(X) > 0).all()}")  # sanity check


if __name__ == "__main__":
    main()
