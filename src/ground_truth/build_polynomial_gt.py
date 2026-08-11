import argparse
import json
import glob
import sys

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
    # Labels like "Tool Wear (μm)" get printed, and stdout defaults to cp1252 on
    # Windows, which cannot encode them.
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Fit a polynomial ground-truth surrogate for a collected "
                    "pybo run's objectives.")
    parser.add_argument("--root-dir", default="",
                        help="Root folder of the Bayesian optimization run, searched "
                             "recursively for experiment.json files (default: %(default)s)")
    parser.add_argument("--degree", type=int, default=2,
                        help="Polynomial degree (default: %(default)s)")
    parser.add_argument("--out",
                        help="Write the fitted coefficients to this JSON file, for an "
                             "objective to load with _load_polynomial_gt. Nothing is "
                             "written when omitted. Suppresses the paste-ready block, "
                             "which is what you want a saved file for in the first place.")
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

    fitted_blocks = {}
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

        scaler = regressor.named_steps["standardscaler"]
        powers = regressor.named_steps["polynomialfeatures"].powers_
        linear_regression = regressor.named_steps["linearregression"]

        # Everything _load_polynomial_gt needs to rebuild the fit in torch.
        # log_space records --positive: without it a reader skips the exp() and is
        # silently wrong by orders of magnitude.
        fitted_blocks[block] = {
            "labels": list(Y.columns),
            "log_space": positive,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "powers": powers.tolist(),
            "coefficients": linear_regression.coef_.T.tolist(),
            "intercept": np.atleast_1d(linear_regression.intercept_).tolist(),
        }
        if args.out:
            # The paste-ready block below is what --out replaces: at the degrees
            # worth saving to a file it is hundreds of unusable lines.
            continue

        # The same fit as a torch expression, ready to paste into an objective's
        # evaluate_true_* — see iformac/constrained/objective.py for the shape.
        # The fit is on standardised X, so the standardisation goes with it: the
        # coefficients mean nothing applied to the raw physical values.
        for column, label in enumerate(Y.columns):
            print(f"\n# --- {block} / {label} ---")
            for k, (mean, scale) in enumerate(zip(scaler.mean_, scaler.scale_)):
                print(f"x{k} = (X[..., {k}] - {mean:.6g}) / {scale:.6g}")
            # The bias term is the all-zero row of powers, and its coefficient is 0
            # by construction (see above), so the intercept opens the sum instead.
            terms = [f"{linear_regression.intercept_[column]:.6g}"]
            for term_powers, coefficient in zip(powers, linear_regression.coef_[column]):
                if not term_powers.any():
                    continue
                factors = "".join(f" * x{k}" + (f"**{p}" if p > 1 else "")
                                  for k, p in enumerate(term_powers) if p)
                sign = "-" if coefficient < 0 else "+"
                terms.append(f"{sign} {abs(coefficient):.6g}{factors}")
            body = "\n        ".join(terms)
            # log(y) was fit, so the pasted expression has to come back out of log
            # space too, or it is silently wrong by orders of magnitude.
            print(f"return torch.exp({body})" if positive else f"return ({body})")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"model": "polynomial", "degree": degree,
                       "parameters": list(X.columns), "blocks": fitted_blocks},
                      f, indent=2, ensure_ascii=False)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
