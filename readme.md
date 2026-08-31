# pyBO — A Python Library for Bayesian Optimization

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17610683.svg)](https://doi.org/10.5281/zenodo.17610683)

`pyBO` is a Python library for Bayesian Optimization of single and multi
objective problems. Built on top of BoTorch and using Gaussian Processes as
surrogate models, it provides a user-friendly framework for optimizing single
and multiple competing objectives, finding optimal or pareto-optimal solutions.

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Experimental workflow](#experimental-workflow)
- [pyBO](#pybo)
    - [Data Format](#data-format)
- [Tutorials](#tutorials)
    - [Running a tutorial](#running-a-tutorial)
    - [Single Objective Problems](#single-objective-problems)
    - [Multi Objective Problems](#multi-objective-problems)
- [Notes](#notes)

## Key Features

Version 0.3.0 currently supports:

- Single-objective optimization problems for vector-valued
  functions $\mathbf{f}_0: \mathbb{R}^n \to \mathbb{R}^1$.
- Multi-objective optimization problems for vector-valued
  functions $\mathbf{f}_0: \mathbb{R}^n \to \mathbb{R}^m$.
- Noisy observations.
- Single and batch mode (q-batch) parallel evaluations.
- Linear equality input constraints.
- Linear inequality input constraints.
- Nonlinear inequality input constraints.
- Linear and non-linear inequality output constraints.
- GPyTorch kernels (e.g. scale, RBF, Matern, Periodic...).
- Any BoTorch acquisition function, single or multi objective, analytic
  (e.g. EI, LogEI) or Monte Carlo based (e.g. qEHVI, qLogEHVI, qNEHVI,
  qLogNEHVI, qNParEGO): the optimizer takes the acquisition function class and
  instantiates it against the current model.
- Two custom acquisition functions on top of those: qDWNEHVI and qEWNEHVI.
- The following samplers: UniformGrid, Sobol, LatinHypercube, Random.
- Two non-Bayesian baseline arms, Sobol and Random, sharing the Bayesian
  optimizer's interface and output format, so a run can be compared against
  pure sampling on the same budget.
- Resuming an interrupted run from its own step records, and replaying a
  previous run's initial design instead of drawing a fresh one.
- Easy-to-write custom objectives, including trackers and penalization.
- A plotter library for objectives, optimization results and metrics.
- CUDA, Apple Metal Framework, and CPU.

## Installation

Currently, the package is only available for local installation.

Clone the repository first:

```
git clone https://github.com/BerriesLab/pybo
cd pybo
```

**Editable install (recommended).** This is what the tutorials and the studies
need, since both are run from the repository root:

```
pip install -e .
```

Two optional extras are available:

```
pip install -e ".[dev]"   # adds pytest
pip install -e ".[gui]"   # adds PySide6, required only by the GUI
```

**Wheel install.** Note that a wheel contains the `pybo` and `pybo_gui`
packages only — not the `tutorials/` or `studies/` trees — so the tutorials
linked below still have to be run from a clone:

1. Install the build tools (if not already installed): <br>
   `python -m pip install --upgrade build setuptools wheel`
2. Build the package: <br>`python -m build`
3. Install the package locally: <br>
   `pip install dist/pyBO-0.3.0-py3-none-any.whl` <br> (Replace
   pyBO-0.3.0-py3-none-any.whl with the actual filename in your dist/ folder)

## Experimental workflow

`pyBO` is designed to integrate seamlessly into any experimental optimization
loop. A typical experimental optimization problem starts with an initial dataset
of (X, Y) pairs, which serve as the prior knowledge for the Bayesian
optimization. Based on this data, the Bayesian optimization suggests a new set
of parameters, `new_X`. The user can then run a new experiment using `new_X` to
obtain new observables, including new objectives `new_Y_obj`, new constraint
values `new_Y_con`, and new tracker values `new_Y_trk`. The new objective and
constraint values, together with the `new X`, are subsequently used to update
the prior belief, by initializing and fitting a new Gaussian Process model. This
model is then optimized, and the process is repeated iteratively until a
convergence criterion is satisfied.

## pyBO

The library lives under [`src/pybo`](src/pybo) and consists of the following
packages (in alphabetical order):

- [acqf](src/pybo/acqf): Includes custom or user-defined acquisition functions.
- [constraints](src/pybo/constraints): Includes constraint definitions for
  objectives.
- [ground_truth](src/pybo/ground_truth): Fits a polynomial surrogate to the
  parameters and observables recorded in a run, to stand in for an expensive
  real experiment. See [its README](src/pybo/ground_truth/README.md).
- [metadata_fixes](src/pybo/metadata_fixes): Command-line tools to add, rename,
  delete or join entries in a recorded run's `experiment.json` metadata.
- [objectives](src/pybo/objectives): Includes single and multi objective classes.
- [optimizer](src/pybo/optimizer): Includes a stateful class that manages the
  Bayesian optimization loop, plus the Sobol and Random baseline arms.
- [plotters](src/pybo/plotters): Includes classes for visualizing optimization
  results, trackers, metrics, and parameters evolution.
- [samplers](src/pybo/samplers): Includes Uniform Grid, Sobol, Latin Hypercube
  and Random Samplers. Provides functionality for constrained sampling.
- [utils](src/pybo/utils): Includes utility functions used across the package,
  among them the shared tutorial CLI, run resuming and initial-design replay.

Two companion trees sit beside the library and are documented separately: the
optional PySide6 GUI and the campaign-analysis command-line tools under
`src/pybo_gui` (installed with the `gui` extra, started with
`python -m pybo_gui.main`, and documented in
[its README](src/pybo_gui/modules/bayesian_campaign_analysis/README.md)), and
the sweep harness that repeatedly launches a tutorial CLI over seeds and
settings in [`studies/`](studies/README.md).

### Data Format

The optimizer accepts the following data:

- $\mathbf{X} \in \mathbb{R}^{n \times d}$: the input data matrix.
- $\mathbf{Y}_{\mathrm{obj}} \in \mathbb{R}^{n \times m}$: the objective value
  matrix.
- $\mathbf{Y}_{\mathrm{obj, \sigma}} \in \mathbb{R}^{n \times m}$: the objective
  variance value matrix (optional).
- $\mathbf{Y}_{\mathrm{con}} \in \mathbb{R}^{n \times c}$: the constraint value
  matrix (optional).
- $\mathbf{Y}_{\mathrm{con, \sigma}} \in \mathbb{R}^{n \times c}$: the
  constraint variance value matrix (optional).

where

- $n$ is the number of observations.
- $d$ is the number of parameters, or input space dimension.
- $m$ is the number of objectives.
- $c$ is the number of output constraints.

## Tutorials

Explore the following examples to learn how to use `pyBO`, and make sure to
review the corresponding objective for learning how to create a custom
objective. Every tutorial is a directory holding exactly two files: a `main.py`
that runs one optimization trial, and an `objective.py` that defines the
problem.

### Running a tutorial

Each tutorial is a runnable module. Run it **from the repository root** — the
`tutorials/` tree carries no `__init__.py`, so it resolves as a namespace
package relative to the current directory:

```
python -m tutorials.single_objective.quadratic.main
python -m tutorials.multi_objective.branin_currin.main --n-evals 64 --q-batch 2
```

All tutorials share the same flags, defined once in
[`pybo.utils.cli`](src/pybo/utils/cli.py) rather than per tutorial:

```
--n-evals --q-batch --noise --repeats --n-initial --init-data --shuffle-init
--seed --output-dir --resume --device --dtype --strategy --verbose
```

Pass `--help` to any tutorial for the full description of each one.
`--strategy` picks the arm to run: `bo` (default), or `sobol` / `random` for
the two sampling baselines.

A run writes to `--output-dir`, defaulting to `<tutorial_dir>/data/<timestamp>/`.
A previous run is never overwritten: an occupied directory is redirected to the
first free `_001`, `_002`, ... sibling. Inside it a run writes `run_config.json`
(the settings the run was started with, which `--resume` checks against),
`summary.bin` at the root, rewritten after every step, and one
`step_NNN_repMM/experiment.json` per step and repeated measurement. The initial
design is measured inside the loop and recorded as steps too, so a run holds
`n_initial + n_evals` step directories, not `n_evals`. Sweeping these same CLIs
over seeds and settings is the job of [`studies/`](studies/README.md).

### Single Objective Problems

- [Quadratic](tutorials/single_objective/quadratic/main.py): An
  unconstrained problem for a quadratic objective function of single
  variable, solved using a scaled RBF kernel.
- [Polynomial](tutorials/single_objective/polynomial/main.py): An
  unconstrained problem for a polynomial objective function of single
  variable, solved using a scaled RBF kernel.
- [Constrained Polynomial](tutorials/single_objective/polynomial_constrained/main.py):
  A constrained problem for a polynomial objective function of single
  variable, solved using a scaled RBF kernel.
- [Harmonic](tutorials/single_objective/harmonic/main.py): An unconstrained
  problem for a harmonic objective function of single variable, solved using
  a scaled cosine kernel.
- [Periodic](tutorials/single_objective/periodic/main.py): An unconstrained
  problem for a periodic objective function of single variable, solved using
  a scaled periodic kernel.
- [Wave Packet](tutorials/single_objective/wave_packet/main.py): An
  unconstrained problem for a wave packet-like objective function of single
  variable, solved using a scaled periodic kernel.
- [Ackley](tutorials/single_objective/ackley/main.py): An unconstrained
  problem for the Ackley test function of two variables, solved using a
  scaled RBF kernel.
- [Rosenbrock](tutorials/single_objective/rosenbrock/main.py): An
  unconstrained problem for the Rosenbrock test function of two variables,
  solved using a scaled RBF kernel.
- [Constrained Rosenbrock](tutorials/single_objective/rosenbrock_constrained/main.py):
  A constrained problem for the Rosenbrock test function of two variables,
  solved using a scaled RBF kernel.

### Multi Objective Problems

- [Branin-Currin](tutorials/multi_objective/branin_currin/main.py): The
  unconstrained two-objective Branin-Currin optimization problem, solved
  using a scaled RBF kernel. This is the reference implementation of the
  trial CLI contract: start here when writing a new one.
- [Linear Equality Test](tutorials/multi_objective/linear_equality/main.py):
  A two-objective optimization test problem with linear equality input
  constraints, solved using a scaled RBF kernel.
- [Linear Inequality Test](tutorials/multi_objective/linear_inequality/main.py):
  A two-objective optimization test problem with linear inequality input
  constraints, solved using a scaled RBF kernel.
- [Binh and Korn](tutorials/multi_objective/binh_and_korn/main.py): The Binh
  and Korn two-objective optimization problem featuring nonlinear inequality
  input constraints, solved using a scaled RBF kernel.
- [Osyczka-Kundu](tutorials/multi_objective/osyczka_kundu/main.py): The
  Osyczka and Kundu two-objective optimization problem, featuring linear
  and nonlinear inequality input constraints, solved using a scaled RBF
  kernel.
- [C2DTLZ2](tutorials/multi_objective/c2dtlz2/main.py): The two-objective
  C2-DTLZ2 optimization problem, featuring nonlinear output constraints,
  solved using a scaled RBF kernel.
- [Tanaka](tutorials/multi_objective/tanaka/main.py): The two-objective
  Tanaka optimization problem with two output constraints, solved using a
  scaled RBF kernel.
- [VFormAC](tutorials/multi_objective/vformac/main.py): An unconstrained
  two-objective problem against a polynomial surrogate of a real V-form
  electrical discharge machining process, fitted from recorded experiments.
  See [its README](tutorials/multi_objective/vformac/README.md).
- [IFormAC](tutorials/multi_objective/iformac/main.py): The constrained
  sibling of VFormAC — the same script shape against an I-form electrical
  discharge machining process, plus one measured output constraint. See
  [its README](tutorials/multi_objective/iformac/README.md).

## Notes

- Sporadically, the following error has been observed: `RuntimeError: main thread is not in main loop Tcl_AsyncDelete:
  async handler deleted by the wrong thread`. This traceback appears to
  originate from the TkAgg backend, which depends on Tkinter. To suppress this
  error, `matplotlib` is imported with the appropriate backend configuration.
  Unfortunately, this approach prevents figures from being displayed on screen,
  meaning that `show()` will no longer function.
