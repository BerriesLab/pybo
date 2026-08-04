"""How a group's repeats and their recorded variance become one error bar.

A group is one setting measured n times (build_group_map), each measurement its own row.
The <label>_var a row carries is the variance of *that observation* - every source it is
subject to, process scatter and instrument together, not the instrument's spec alone:

    y_i = mu + e_i        Var(e_i) = V = tau^2 + sigma_meas^2

That is also what BoTorch's train_Yvar means, and the two must agree. The distinction
bites when a run averages its repeats before recording them: the average of n draws has
variance V/n, so a row holding a mean must record V/n, not V. pybo's tutorials record one
row per measurement, so a row records V.

Given that, there are two estimates of the same V: the recorded one, and the sample
variance s^2 of the group's n values. The trap is treating them as separate sources and
adding them - s^2 already estimates the whole of V, so adding would count it twice. They
are reconciled instead:

    V_hat = max(s^2, V_recorded)

The recorded value acts as a floor. That is what it is worth: at n = 3 the sample variance
has 100% relative standard deviation, so repeats that happen to agree do not buy precision
the process does not have - and a point measured once gets a real bar instead of a
zero-width one.

Both estimators below are offered because they answer different questions.
"""
import numpy as np


def total_sd(values, variances) -> float:
    """The spread of a single measurement at this setting: sqrt(max(s^2, V)).

    Answers "measure this setting again and the reading lands about this far away".
    Repeating a setting does not shrink it; it only measures it better.

    `variances` are the recorded observation variances of the same rows, with None where
    a row recorded none. With none recorded this is the plain sample std, which is what
    the plots drew before any run recorded a variance.
    """
    values = [v for v in values if v is not None]
    s2 = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
    known = [v for v in variances if v is not None]
    # Constant by assumption, so the mean is just a robust way to read the one value.
    sigma2 = float(np.mean(known)) if known else 0.0
    return float(np.sqrt(max(s2, sigma2)))


def mean_sd(values, variances) -> float:
    """The uncertainty of the group's mean: total_sd / sqrt(n).

    Answers "how well do I know where this point sits", which is what a bar on a Pareto
    point is read as. Unlike total_sd it rewards repeating a setting - n measurements
    shrink it by sqrt(n).

    This is also the standard deviation the group's mean would carry as an observation in
    its own right, so a run that fed the GP one averaged row per setting would want
    exactly its square as that row's train_Yvar.
    """
    n = len([v for v in values if v is not None])
    if n == 0:
        return 0.0
    return total_sd(values, variances) / (n ** 0.5)