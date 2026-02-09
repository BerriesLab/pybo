# pyBO — A Python Library for Bayesian Optimization

`pyBO` is a Python library for Bayesian Optimization of single and multi
objective problems. Built on top of BoTorch and using Gaussian Processes as
surrogate models, it provides a user-friendly framework for optimizing single
and multiple competing objectives, finding optimal or pareto-optimal solutions.

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Experimental workflow](#experimental-workflow)
- [pyBO](#pybo-internal-workflow)
    - [Data Format](#data-format)
    - [Visualization](#visualization)
    - [Tutorials](#tutorials)
    - [Notes](#notes)

## Key Features

Version 0.3 currently supports:

- Single-objective optimization problems for vector-valued
  functions $\mathbf{f}_0: \mathbb{R}^n \to \mathbb{R}^1$.
- Multi-objective optimization problems for vector-valued
  functions $\mathbf{f}_0: \mathbb{R}^n \to \mathbb{R}^m$.
- Noisy observations.
- Single and batch mode (q-batch) parallel evaluations.
- Linear equality input constraints.
- Linear inequality input constraints.
- Nonlinear inequality inout constraints.
- Linear and non-linear inequality outptut constraints.
- GPyTorch kernels (e.g. scale, RBF, Matern, Periodic...).
- The following analytical acquisition functions: EI, LogEI,
- The following Monte Carlo based acquisition functions: qEHVI, qLogEHVI,
  qNEHVI, qLogNEHVI, qDWNEHVI, qEWNEHVI, qNParEGO.
- The following samplers: UniformGrid, Sobol, LatinHypercube.
- Easy-to-write custom objectives, including trackers and penalization.
- A plotter library for objectives, optimization results and metrics.
- CUDA, Apple Metal Framework, and CPU.

## Installation

Currently, the package is only available for local installation.

1. Clone the repository: <br>`git clone https://github.com/BerriesLab/pybo`
2. Install required build tools (if not already installed): <br>
   `python -m pip install --upgrade build setuptools wheel`
3. Build the package: <br>`python -m build`
4. Install the package locally: <br>
   `pip install dist/pyBO-0.1.0-py3-none-any.whl` <br> (Replace
   pyBO-0.1.0-py3-none-any.whl with the actual filename in your dist/ folder)

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

pyBO consists of the following packages (in alphabetical order):

- [acqf](acqf): Includes custom or user-defined acquisition functions.
- [constraints](constraints): Includes constraint definitions for objectives.
- [objectives](objectives): Includes single and multi objective classes.
- [optimizer](optimizer): Includes a stateful class that manages the Bayesian
  optimization loop.
- [plotters](plotters): Includes classes for visualizing optimization results,
  trackers, metrics, and parameters evolution.
- [samplers](samplers): Includes Uniform Grid, Sobol and Latin Hypercube
  Samplers. Provides functionality for constrained sampling.
- [utils](utils): Includes utility functions used across the package.

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
objective.

- **Single Objective Problems**
    - [Quadratic](tutorials/single_objective/quadratic/main.py): An
      unconstrained problem for a quadratic objective function of single
      variable, solved using a scaled RBF kernel.
    - [Polynomial](tutorials/single_objective/polynomial/main.py): An
      unconstrained problem for a polynomial objective function of single
      variable, solved using a scaled RBF kernel.
    - [Constrained Polynomial](tutorials/single_objective/polynomial_constrained):
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


- **Multi Objective Problems**
    - [Branin-Currin](tutorials/multi_objective/branin_currin/main.py): The    
      unconstrained two-objective Branin-Currin optimization problem, solved
      using a scaled RBF kernel.
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

## Notes

- Sporadically, the following error has been observed: `RuntimeError: main thread is not in main loop Tcl_AsyncDelete:
  async handler deleted by the wrong thread`. This traceback appears to
  originate from the TkAgg backend, which depends on Tkinter. To suppress this
  error, `matplotlib` is imported with the appropriate backend configuration.
  Unfortunately, this approach prevents figures from being displayed on screen,
  meaning that `show()` will no longer function.
