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


def _pooled_noise_std(X, Y, decimals):
    """Pooled within-setting std per column of Y, as (std, n_groups, dof), or
    (None, 0, 0) when no setting was measured more than once.

    Rows are grouped by their parameters, so a group is one setting measured
    repeatedly and its spread is pure measurement noise — independent of whatever
    surrogate gets fitted, which is what makes it the right number to hand back to
    an objective as gt_*_noise_std. The fit's own residuals would confound that
    noise with the polynomial being the wrong shape.

    Parameters are range-normalised before rounding because they carry wildly
    different magnitudes (volts alongside nanoseconds), and one decimal count
    cannot serve both. Repeated runs normally record identical setpoints anyway,
    so the rounding is a guard against float formatting rather than real binning.

    Pooled, not averaged: group sizes are uneven, and dividing the total squared
    deviation by the total degrees of freedom weights each group by what it
    actually contributes. Groups of one carry no degrees of freedom and drop out
    rather than counting as a variance of zero, which would pull a plain average
    of per-group variances toward nothing.
    """
    span = X.max() - X.min()
    # A parameter held constant across the campaign has no scale to normalise by.
    span = span.where(span > 0, 1.0)
    key = ((X - X.min()) / span).round(decimals)
    key = pd.Series(list(key.itertuples(index=False, name=None)), index=Y.index)

    squares = pd.Series(0.0, index=Y.columns)
    n_groups = 0
    dof = 0
    for _, rows in Y.groupby(key, sort=False):
        if len(rows) < 2:
            continue
        squares += ((rows - rows.mean()) ** 2).sum()
        n_groups += 1
        dof += len(rows) - 1
    if dof == 0:
        return None, 0, 0
    return np.sqrt(squares / dof), n_groups, dof


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
                        help="Archive the fitted coefficients to this JSON file. "
                             "Nothing reads it back - objectives hold their ground "
                             "truth in their own methods - so this is a record of a "
                             "fit, not a runtime input. Nothing is written when "
                             "omitted. Suppresses the paste-ready block, which is the "
                             "output you actually paste.")
    parser.add_argument("--paste-out",
                        help="Write the pasteable body of each fitted quantity to this "
                             "text file: the standardisation lines and the named "
                             "expression, one block per quantity, with no def and no "
                             "return. That is the shape a per-quantity method in an "
                             "objective takes, so a block goes straight into one.")
    parser.add_argument("--group-decimals", type=int, default=6,
                        help="Decimals to round the range-normalised parameters to when "
                             "deciding which observations repeat the same setting, for "
                             "the pooled noise std (default: %(default)s). Lower it if "
                             "repeats were recorded with drifting setpoints and end up "
                             "in groups of one.")
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
    paste_blocks = []
    for block, Y in [("objectives", Y_obj), ("constraints", Y_con), ("trackers", Y_trk)]:
        print(f"\n=== {block} ===")
        if Y is None:
            print("none recorded (or only recorded on some observations), skipped")
            continue

        # Pure error, measured off the raw observations before anything is fitted:
        # the settings that were measured more than once are the only rows that say
        # anything about the process's own variability. It lands in the "noisy"
        # branch of the method emitted below, because the noise belongs to the
        # quantity itself rather than to any attribute of the framework.
        noise_std, n_groups, dof = _pooled_noise_std(X, Y, args.group_decimals)
        if noise_std is None:
            print(f"no setting measured more than once at --group-decimals "
                  f"{args.group_decimals}, noise std not estimated")
        else:
            print(f"\npooled noise std over {n_groups} repeated settings, {dof} dof:")
            print(noise_std)

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

        # Everything needed to rebuild the fit, kept as a record of what was
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
            # Saved next to the coefficients so the noise travels with the fit it
            # was measured alongside, instead of being retyped into an objective
            # where the two can drift apart. null when nothing was repeated.
            # noise_dof is how much the estimate is worth: at 2 dof it is barely
            # an estimate at all.
            "noise_std": None if noise_std is None else noise_std.tolist(),
            "noise_groups": n_groups,
            "noise_dof": dof,
        }
        if args.out:
            # The paste-ready block below is what --out replaces: at the degrees
            # worth saving to a file it is hundreds of unusable lines.
            continue

        # The same fit as one whole method, ready to paste into an objective and
        # replace whatever it holds today. Emitted as a
        # method rather than as one bare expression per column because the method
        # has to hand back every column of the block stacked into a single tensor,
        # which is the step a per-column expression leaves the reader to work out.
        # The fit is on standardised X, so the standardisation goes with it: the
        # coefficients mean nothing applied to the raw physical values.
        method = {"objectives": "evaluate_true_objective",
                  "constraints": "evaluate_true_constraint",
                  "trackers": "evaluate_tracker"}[block]
        print(f"\n# --- {block}: paste into {method} ---")
        print(f"# Columns are in the records' order: {', '.join(Y.columns)}.")
        print(f"# Reorder the stack to match the order the objective declares them in.")
        print(f"# Fitted from {root_dir} at degree {degree}, R2 = {pipeline.score(X, Y):.4g}.")
        print(f"    def {method}(self, X: Tensor, noisy: bool = False) -> Tensor:")
        for k, (mean, scale) in enumerate(zip(scaler.mean_, scaler.scale_)):
            print(f"        x{k} = (X[..., {k}] - {mean:.6g}) / {scale:.6g}")

        scaling = [f"        x{k} = (X[..., {k}] - {mean:.6g}) / {scale:.6g}"
                   for k, (mean, scale) in enumerate(zip(scaler.mean_, scaler.scale_))]

        names = []
        for column, label in enumerate(Y.columns):
            name = "".join(c if c.isalnum() else "_" for c in label).strip("_").lower()
            names.append(name)
            # The bias term is the all-zero row of powers, and its coefficient is 0
            # by construction (see above), so the intercept opens the sum instead.
            terms = [f"{linear_regression.intercept_[column]:.6g}"]
            for term_powers, coefficient in zip(powers, linear_regression.coef_[column]):
                if not term_powers.any():
                    continue
                factors = "".join(f" * x{k}" + (f" ** {p}" if p > 1 else "")
                                  for k, p in enumerate(term_powers) if p)
                sign = "-" if coefficient < 0 else "+"
                terms.append(f"{sign} {abs(coefficient):.6g}{factors}")
            # Continuation lines line up under the opening bracket, which sits at
            # "        <name> = (" — so the indent is the name's own length plus 12.
            body = ("\n" + " " * (len(name) + 12)).join(terms)
            # log(y) was fit, so the pasted expression has to come back out of log
            # space too, or it is silently wrong by orders of magnitude.
            opening = f"        {name} = torch.exp(" if positive else f"        {name} = ("
            expression = f"{opening}{body})"
            print(expression)

            # The standardisation is repeated for every quantity rather than shared,
            # because each one lands in a method of its own and has to stand alone.
            std_note = ("" if noise_std is None
                        else f", pooled noise std {noise_std.iloc[column]:.4g}")
            header = f"# {block} / {label} (degree {degree}{std_note})"
            paste_blocks.append("\n".join([header, *scaling, expression]))

        # The noise draws on each quantity separately, before anything derived from
        # it, so a caller that builds a hinge or a deviation out of one of these
        # inherits the noise instead of having a second draw stacked on top.
        if noise_std is not None:
            print("        if noisy:")
            for name, std in zip(names, noise_std):
                print(f"            {name} = {name} + {std:.4g} * torch.randn_like({name})")
        else:
            print("        if noisy:")
            print(f'            raise ValueError(f"{{type(self).__name__}} declares no "')
            print(f'                             f"ground-truth noise. Run with --noise false.")')

        # One column is still a column: it leaves as [..., 1] like every other block,
        # so whatever consumes it does not have to special-case the single-output case.
        if len(names) == 1:
            print(f"        return {names[0]}.unsqueeze(-1)")
        else:
            print(f"        return torch.stack([{', '.join(names)}], dim=-1)")

    if args.paste_out:
        with open(args.paste_out, "w", encoding="utf-8") as f:
            f.write("\n\n".join(paste_blocks) + "\n")
        print(f"\npasteable bodies -> {args.paste_out}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"model": "polynomial", "degree": degree,
                       "parameters": list(X.columns), "blocks": fitted_blocks},
                      f, indent=2, ensure_ascii=False)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
