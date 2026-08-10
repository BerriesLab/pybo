import glob
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import chi2

# The repo root, so the absolute import below resolves when this is launched as a script
# from a terminal. An IDE puts the content root on the path already; python does not.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tutorials.multi_objective.iformac.constrained.objective import IFormACConstrained

# --- config ---
DEVICE = torch.device("cpu")
DTYPE = torch.float64
DATA_DIR = "data/iformac"
MODEL_FILE = "gp_ground_truth.pt"
OUTPUT_FILE = "ground_truth_gp.csv"

# Steps holding results but no parameters, which the port skips for that reason. Such a
# step cannot enter the fit - there is no X to place it at - but a set of them known to be
# one repeated setting would still be worth its degrees of freedom to the noise estimate.
# Left empty: nothing in the data marks a step as a repeat of another, and guessing it from
# results that happen to sit close is circular - it assumes the spread this is measuring.
NOISE_ONLY_GROUPS = []

objective = IFormACConstrained(device=DEVICE, dtype=DTYPE)
par_labels = [c.label for c in objective.par_cfg or []]
out_labels = ([c.label for c in objective.obj_cfg or []]
              + [c.label for c in objective.ineq_Y_con_cfg or []])

# --- load the ported steps ---
# One observation per step, read by label rather than by column position, so a reordered
# problem cannot silently transpose the data.
X_rows, Y_rows = [], []
for path in sorted(glob.glob(os.path.join(DATA_DIR, "step_*", "experiment.json"))):
    record = json.loads(Path(path).read_text(encoding="utf-8"))["data"][0]
    values = dict(record["objectives"], **record["constraints"])
    X_rows.append([record["parameters"][label] for label in par_labels])
    Y_rows.append([values[label] for label in out_labels])
X_all = np.array(X_rows, dtype=float)
Y_all = np.array(Y_rows, dtype=float)
print(f"{len(X_all)} observations from {DATA_DIR}")

# --- collapse replicates ---
# Rows sharing their parameters exactly are repeat fabrications of the same cavity. The
# mean is a sufficient statistic, so the GP sees one row per unique setting; what the
# repeats buy is the noise estimate below and a sigma^2/n on the rows that have them.
keys = [tuple(np.round(row, 6)) for row in X_all]
order, groups = [], {}
for i, key in enumerate(keys):
    if key not in groups:
        groups[key] = []
        order.append(key)
    groups[key].append(i)
X = np.array([X_all[groups[key][0]] for key in order])
Y = np.array([Y_all[groups[key]].mean(axis=0) for key in order])
n = np.array([len(groups[key]) for key in order], dtype=float)
print(f"{len(X)} unique settings, {int((n > 1).sum())} replicated")

# --- pooled within-setting noise ---
# s2 = sum of squared deviations from each setting's own mean, over sum(n_i - 1), per
# output. Never pooled across outputs: minutes and microns have nothing to say to each other.
ss = np.zeros(Y.shape[1])
dof = 0
for key in order:
    rows = Y_all[groups[key]]
    if len(rows) > 1:
        ss += ((rows - rows.mean(axis=0)) ** 2).sum(axis=0)
        dof += len(rows) - 1

# The unlogged-parameter repeats, contributing to the spread but not to the fit.
for names in NOISE_ONLY_GROUPS:
    rows = []
    for name in names:
        meta = json.loads(Path(DATA_DIR, name, "metadata.json").read_text(encoding="utf-8"))
        results = meta.get("results") or {}
        # Read through the same mapping the port used, so these rows mean what the others do.
        rows.append([results["down_time_minutes"], results["wear_microns"],
                     results["orbiting_time_minutes"]])
    rows = np.array(rows, dtype=float)
    ss += ((rows - rows.mean(axis=0)) ** 2).sum(axis=0)
    dof += len(rows) - 1
    print(f"  + {names} as a noise-only repeat ({len(rows) - 1} dof)")

if dof == 0:
    raise SystemExit("No replicated settings - nothing to estimate noise from.")
sigma = np.sqrt(ss / dof)

# Chi-square interval. On this few repeats it is wide, and that width is the honest
# uncertainty of the number that goes into gt_obj_noise_std.
lo = sigma * np.sqrt(dof / chi2.ppf(0.975, dof))
hi = sigma * np.sqrt(dof / chi2.ppf(0.025, dof))
print(f"\npooled noise ({dof} dof):")
for i, label in enumerate(out_labels):
    print(f"  {label:24s} sigma = {sigma[i]:8.3f}   95% CI [{lo[i]:.3f}, {hi[i]:.3f}]")

# --- additive or relative? ---
# If a setting's spread tracks its level, one scalar sigma is the wrong model and the fit
# belongs on log Y. Too few groups to test; this is here to be looked at.
print("\nper-setting spread vs level (relative noise check):")
for i, label in enumerate(out_labels):
    print(f"  {label}")
    for key in order:
        rows = Y_all[groups[key]][:, i]
        if len(rows) > 1:
            print(f"    mean {rows.mean():9.2f}   sd {rows.std(ddof=1):8.3f}"
                  f"   sd/mean {rows.std(ddof=1) / rows.mean():6.3f}")

# --- fit, noise pinned to what the repeats measured ---
# Free noise on this few points absorbs model misfit as well as process scatter, which
# oversmooths the surface and hands the benchmark an easier problem than the real one.
train_X = torch.tensor(X, device=DEVICE, dtype=DTYPE)
train_Y = torch.tensor(Y, device=DEVICE, dtype=DTYPE)
# The mean of n repeats has variance sigma^2/n: this is the whole of how replicates weigh in.
train_Yvar = (torch.tensor(sigma ** 2, device=DEVICE, dtype=DTYPE)
              / torch.tensor(n, device=DEVICE, dtype=DTYPE).unsqueeze(-1))

models, preds = [], []
for i, label in enumerate(out_labels):
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y[..., i: i + 1],
        train_Yvar=train_Yvar[..., i: i + 1],
        input_transform=Normalize(d=objective.dim, bounds=objective.bounds),
        outcome_transform=Standardize(m=1),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    models.append(model)
    with torch.no_grad():
        preds.append(model.posterior(train_X).mean.squeeze(-1))

# --- diagnostic: what would the GP have called noise on its own? ---
# sigma_free >> sigma means the GP is filing structure under noise (wrong kernel, or a
# response that wants a log transform) - fix that rather than accepting the inflated one.
print("\nfree-noise fit vs measured noise:")
for i, label in enumerate(out_labels):
    free = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y[..., i: i + 1],
        input_transform=Normalize(d=objective.dim, bounds=objective.bounds),
        outcome_transform=Standardize(m=1),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(free.likelihood, free))
    scale = float(free.outcome_transform.stdvs.detach().squeeze())
    sigma_free = float(free.likelihood.noise.detach().sqrt().squeeze()) * scale
    print(f"  {label:24s} sigma_free = {sigma_free:8.3f}   measured = {sigma[i]:8.3f}"
          f"   ratio {sigma_free / sigma[i]:5.2f}")

# --- how much of the box did the campaign actually teach? ---
# Away from the data a GP posterior mean falls back to the prior, i.e. to the data mean -
# a flat plateau, not an extrapolation. This says how much of the design space that is.
probe = objective.bounds[0] + (objective.bounds[1] - objective.bounds[0]) * torch.rand(
    4096, objective.dim, device=DEVICE, dtype=DTYPE)
print("\nposterior std over the box, as a fraction of the data spread:")
for i, label in enumerate(out_labels):
    with torch.no_grad():
        rel = models[i].posterior(probe).variance.sqrt().squeeze(-1) / train_Y[..., i].std()
    print(f"  {label:24s} median {float(rel.median()):.2f}   "
          f"share of box above 0.9 (unlearned): {float((rel > 0.9).float().mean()):.0%}")

# --- validation: the replicated settings are the only places truth is known twice ---
print("\nfit vs observed mean at the replicated settings (expect within sigma/sqrt(n)):")
for i, label in enumerate(out_labels):
    for row in np.nonzero(n > 1)[0]:
        err = float(preds[i][row]) - Y[row, i]
        se = sigma[i] / n[row] ** 0.5
        print(f"  {label:24s} err {err:8.3f}   sigma/sqrt(n) {se:7.3f}   "
              f"{'ok' if abs(err) < 2 * se else 'CHECK'}")

# --- plot: fit against the averaged data, error bars = sigma/sqrt(n) ---
fig, axes = plt.subplots(1, len(out_labels), figsize=(4 * len(out_labels), 4))
for ax, label, i in zip(np.atleast_1d(axes), out_labels, range(len(out_labels))):
    ax.errorbar(Y[:, i], preds[i].numpy(), xerr=sigma[i] / np.sqrt(n), fmt="o", alpha=0.6, ms=4)
    edge = [Y[:, i].min(), Y[:, i].max()]
    ax.plot(edge, edge, "k--")
    ax.set_xlabel(f"{label} observed")
    ax.set_ylabel(f"{label} GP mean")
    ax.set_title(f"sigma = {sigma[i]:.2f}")
plt.tight_layout()
plt.show()

# --- save ---
# The raw training data alongside the weights: a SingleTaskGP is its data, and the state
# dict alone cannot rebuild it.
torch.save({"train_X": train_X,
            "train_Y": train_Y,
            "train_Yvar": train_Yvar,
            "outputs": out_labels,
            "sigma": torch.tensor(sigma),
            "state_dicts": [m.state_dict() for m in models]}, MODEL_FILE)
print("\nsaved →", MODEL_FILE)

header = par_labels + [f"{label} mean" for label in out_labels] + \
         [f"{label} fit" for label in out_labels] + ["n"]
table = np.column_stack([X, Y, np.column_stack([p.numpy() for p in preds]), n])
# Opened here rather than by name: savetxt would use the locale encoding, which on Windows
# is cp1252 and cannot write the mu in "Tool Wear (um)".
with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as file:
    np.savetxt(file, table, delimiter=",", header=",".join(header), comments="")
print("saved →", OUTPUT_FILE)

# --- paste into the objective ---
print("\ngt_obj_noise_std=[" + ", ".join(f"{sigma[i]:.3f}" for i in range(len(objective.obj_cfg)))
      + "],  # " + ", ".join(c.label for c in objective.obj_cfg))
print("gt_con_noise_std=[" + ", ".join(f"{sigma[i]:.3f}" for i in
                                       range(len(objective.obj_cfg), len(out_labels)))
      + "],  # " + ", ".join(c.label for c in objective.ineq_Y_con_cfg))
