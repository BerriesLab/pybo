import json
import sys
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

root = Path(sys.argv[1]).resolve()
degrees = [int(d) for d in sys.argv[2:]] or [1, 2, 3]

par_rows, obj_rows, con_rows, trk_rows = [], [], [], []
for path in sorted(root.glob("*/experiment.json")):
    for experiment in json.loads(path.read_text(encoding="utf-8"))["data"]:
        for rows, group in ((par_rows, "parameters"), (obj_rows, "objectives"),
                            (con_rows, "constraints"), (trk_rows, "trackers")):
            rows.append({k: v for k, v in experiment[group].items()
                         if not k.endswith("_var")})

par_labels = list(par_rows[0])
obj_labels = list(obj_rows[0])
con_labels = list(con_rows[0])
trk_labels = list(trk_rows[0])

X = np.array([[row[label] for label in par_labels] for row in par_rows], dtype=float)
Y_obj = np.array([[row[label] for label in obj_labels] for row in obj_rows], dtype=float)
Y_con = np.array([[row[label] for label in con_labels] for row in con_rows], dtype=float)
Y_trk = np.array([[row[label] for label in trk_labels] for row in trk_rows], dtype=float)

print(f"{len(X)} observations from {root}")
print(f"X     {X.shape}  {par_labels}")
print(f"Y_obj {Y_obj.shape}  {obj_labels}")
print(f"Y_con {Y_con.shape}  {con_labels}")
print(f"Y_trk {Y_trk.shape}  {trk_labels}")

# --- fit ---
# X is scaled to the unit box first: td1 and td2 run in the tens of thousands, and their
# squares and cross terms would otherwise sit orders of magnitude above the constant term,
# which is what leaves the fit ill-conditioned.
lo, hi = X.min(axis=0), X.max(axis=0)
Xs = (X - lo) / (hi - lo)

# The ridge penalty rides along as extra residual rows - minimizing |Ac - y|^2 + alpha|c|^2
# is the same as least-squares on [Ac - y; sqrt(alpha) c]. The constant term is left
# unpenalized, or the fit would be pulled towards zero rather than towards the data mean.
# It is needed because every degree past 2 spends more terms than 94 points can afford:
# unregularized, the fit interpolates and the held-out predictions go to pieces.
residual = lambda c, A_, y, alpha: np.concatenate(
    [A_ @ c - y, np.sqrt(alpha) * np.r_[0.0, c[1:]]])

ALPHAS = np.logspace(-6, 2, 9)

# Shuffled once with a fixed seed, then cut into 5: the steps sit in campaign order, and
# contiguous folds would split the run by phase rather than at random.
folds = np.array_split(np.random.default_rng(0).permutation(len(X)), 5)
rows = np.arange(len(X))

# In-sample R2 only ever rises with degree, so both the degree and alpha are chosen on the
# CV column: each point predicted by a fit that never saw it.
print(f"\n{'degree':>6}  {'terms':>5}   " +
      "   ".join(f"{label:>26s}" for label in obj_labels))
print(f"{'':>6}  {'':>5}   " + "   ".join(f"{'R2 (cv R2) @alpha':>26s}" for _ in obj_labels))
for degree in degrees:
    terms = [t for d in range(degree + 1)
             for t in combinations_with_replacement(range(X.shape[1]), d)]
    A = np.column_stack([np.prod(Xs[:, list(t)], axis=1) for t in terms])

    cells = []
    for j, label in enumerate(obj_labels):
        y = Y_obj[:, j]
        ss_tot = ((y - y.mean()) ** 2).sum()

        # One objective at a time: least_squares wants a 1-D residual.
        best = None
        for alpha in ALPHAS:
            y_cv = np.empty(len(X))
            for fold in folds:
                train = np.setdiff1d(rows, fold)
                fit = least_squares(residual, np.zeros(len(terms)),
                                    args=(A[train], y[train], alpha))
                y_cv[fold] = A[fold] @ fit.x
            r2_cv = 1 - ((y - y_cv) ** 2).sum() / ss_tot
            if best is None or r2_cv > best[0]:
                best = (r2_cv, alpha)

        r2_cv, alpha = best
        fit = least_squares(residual, np.zeros(len(terms)), args=(A, y, alpha))
        r2 = 1 - ((y - A @ fit.x) ** 2).sum() / ss_tot
        cells.append(f"{r2:6.3f} ({r2_cv:6.3f}) @{alpha:<7.1e}")
    print(f"{degree:>6}  {len(terms):>5}   " + "   ".join(f"{c:>26s}" for c in cells))