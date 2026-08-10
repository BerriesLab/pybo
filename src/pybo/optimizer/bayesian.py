import copy
import inspect
import warnings
import torch
import botorch
import gpytorch
from typing import Type
from gpytorch.kernels import Kernel
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.optim.optimize import optimize_acqf_list
from botorch.sampling import MCSampler, SobolQMCNormalSampler
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition import AcquisitionFunction, MCAcquisitionFunction
from botorch.acquisition.multi_objective.parego import qLogNParEGO

from pybo.objectives.base_class import MCSingleObjectiveBase, MCMultiObjectiveBase
from pybo.optimizer.base_class import OptimizerBase
from pybo.samplers.sobol import SobolSampler


class BayesianOptimizer(OptimizerBase):
    """
    A wrapper around BoTorch for Bayesian Optimization supporting both
    single-objective and multi-objective optimization.
    """

    experiment_type = "bayesian"

    # The MC sampler and the acquisition function instances do not survive pickling.
    _unpicklable_attrs = ("_acquisition_function", "_sampler", "_acq_func_list")

    def __init__(
            self,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
            objective: MCSingleObjectiveBase | MCMultiObjectiveBase | None = None,
            X: torch.Tensor | None = None,
            Y_obj: torch.Tensor | None = None,
            Y_obj_var: torch.Tensor | None = None,
            Y_con: torch.Tensor | None = None,
            Y_con_var: torch.Tensor | None = None,
            Y_trk: torch.Tensor | None = None,
            Y_trk_var: torch.Tensor | None = None,
            acqf: Type[AcquisitionFunction] | None = None,
            kernel: Kernel | None = None,
            batch_size: int = 1,
            mc_samples: int = 256,
            raw_samples: int = 1024,
            n_acqf_opt_max_iter: int = 250,
            n_acqf_opt_restarts: int = 20,
            n_model_fit_restarts: int = 50,
    ):
        super().__init__(
            device=device,
            dtype=dtype,
            objective=objective,
            X=X,
            Y_obj=Y_obj,
            Y_obj_var=Y_obj_var,
            Y_con=Y_con,
            Y_con_var=Y_con_var,
            Y_trk=Y_trk,
            Y_trk_var=Y_trk_var,
            batch_size=batch_size,
        )

        # ===== Model Attributes =====
        self._model: ModelListGP | None = None
        self._mll: SumMarginalLogLikelihood | None = None
        self._acquisition_function_list: list[AcquisitionFunction] | None = None
        self._acqf_instance: AcquisitionFunction | None = None
        self._partitioning: NondominatedPartitioning | None = None

        # ===== Optimization attributes =====
        self._acqf = acqf  # The acquisition function instance
        self._sampler = None  # A Sobol QMC sampler used to numerically compute the acquisition function
        self._kernel = kernel  # The kernel instance
        self._n_acqf_opt_max_iter = n_acqf_opt_max_iter  # Number of iterations for acquisition function optimization
        self._n_acqf_opt_restarts = n_acqf_opt_restarts  # The number of initial guesses used to optimize the acquisition function
        self._n_model_fit_restarts = n_model_fit_restarts  # Fit attempts per output model (BoTorch resamples hyperparameters after the first)
        self._num_mc_samples = mc_samples  # Number of samples drawn from the predictive posterior distribution to estimate the acquisition function
        self._num_raw_samples = raw_samples  # Number of random points sampled in the search space to initialize the optimizer that maximizes the acquisition function

    """ =================================== """
    """ ===== EXPERIMENTAL PROPERTIES ===== """
    """ =================================== """

    @property
    def X_baseline(self):
        return self.X  # This is required for dynamic instantiation of acqf.

    @property
    def acqf(self) -> Type[AcquisitionFunction] | None:
        return self._acqf

    @property
    def sampler(self) -> MCSampler | None:
        """The MC sampler handed to the acquisition function, or None until one is built.

        This is BoTorch's posterior sampler, not a pybo SamplerBase: it draws from the
        model's posterior to estimate an MC acquisition value, and never sees X. The
        candidate-point sampler with the same name lives on SobolOptimizer."""
        return self._sampler

    @property
    def n_acqf_opt_iter(self) -> int:
        return self._n_acqf_opt_max_iter

    @property
    def n_acqf_opt_restarts(self) -> int:
        return self._n_acqf_opt_restarts

    @property
    def n_model_fit_restarts(self) -> int:
        return self._n_model_fit_restarts

    @property
    def num_mc_samples(self):
        return self._num_mc_samples

    @property
    def num_raw_samples(self):
        return self._num_raw_samples

    @acqf.setter
    def acqf(self, af_type):
        if not (inspect.isclass(af_type) and issubclass(af_type, AcquisitionFunction)):
            raise ValueError(
                f"Acquisition function must be a class inheriting from AcquisitionFunction. "
                f"Got {type(af_type).__name__}: {af_type}"
            )
        self._acqf = af_type

    @n_acqf_opt_iter.setter
    def n_acqf_opt_iter(self, n_acqf_opt_iter):
        if not isinstance(n_acqf_opt_iter, int):
            raise ValueError("n_acqf_opt_max_iter must be of type int")
        self._n_acqf_opt_max_iter = n_acqf_opt_iter

    @n_acqf_opt_restarts.setter
    def n_acqf_opt_restarts(self, value: int):
        if not isinstance(value, int):
            raise ValueError("n_acqf_opt_restarts must be of type int")
        self._n_acqf_opt_restarts = value

    @n_model_fit_restarts.setter
    def n_model_fit_restarts(self, value: int):
        if not isinstance(value, int):
            raise ValueError("n_model_fit_restarts must be of type int")
        # This is BoTorch's max_attempts, whose loop is range(1, 1 + max_attempts):
        # anything below 1 means the model is never fitted at all, and surfaces as
        # a bare ModelFittingError rather than as the misconfiguration it is.
        if value < 1:
            raise ValueError("n_model_fit_restarts must be at least 1 (it is the number of fit attempts per output)")
        self._n_model_fit_restarts = value

    @num_mc_samples.setter
    def num_mc_samples(self, mc_samples: int):
        if not isinstance(mc_samples, int):
            raise ValueError("num_mc_samples must be of type int")
        self._num_mc_samples = mc_samples

    @num_raw_samples.setter
    def num_raw_samples(self, raw_samples: int):
        if not isinstance(raw_samples, int):
            raise ValueError("num_raw_samples must be of type int")
        self._num_raw_samples = raw_samples

    """ ============================ """
    """ ===== STATE PROPERTIES ===== """
    """ ============================ """

    @property
    def model(self) -> SingleTaskGP | ModelListGP | None:
        if self._model is None:
            print("A model has not been generated yet.")
        return self._model

    @property
    def mll(self) -> SumMarginalLogLikelihood | None:
        if self._mll is None:
            print("A model has not been generated yet.")
        return self._mll

    @property
    def acquisition_function_list(self) -> list[AcquisitionFunction] | None:
        if self._acquisition_function_list is None:
            print("The acquisition function has not been initialized yet.")
        return self._acquisition_function_list

    @property
    def partitioning(self) -> NondominatedPartitioning:
        if self._partitioning is None:
            raise AttributeError("A partitioning has not been computed yet.")
        return self._partitioning

    @property
    def acqf_instance(self) -> AcquisitionFunction | None:
        if self._acqf_instance is None:
            print("An acquisition function has not been instantiated yet.")
        return self._acqf_instance

    @property
    def kernel(self) -> Kernel | None:
        return self._kernel

    @kernel.setter
    def kernel(self, kernel: Kernel) -> None:
        if not isinstance(kernel, Kernel):
            raise ValueError("kernel_type must be of type Kernel.")
        self._kernel = kernel

    """ ===================== """
    """ ===== Optimizer ===== """
    """ ===================== """

    def _propose(self, verbose=True):
        """Fit the surrogate to the scored data, then maximize the acquisition function
        over it."""

        # === Surrogate Modeling ===
        # Fit the GPs to the data we just analyzed
        self._initialize_model(verbose=verbose)
        self._fit_model(verbose=verbose)

        # === Acquisition Strategy ===
        # Initialize and optimize the acquisition function
        self._initialize_acquisition_function(verbose=verbose)
        self._optimize_acquisition_function(verbose=verbose)

    def _initialize_model(self, verbose=True):
        """ Initialize Gaussian Process model(s) for the objectives and constraints.

        This method prepares the training dataset by combining the objective and constraint
        observations (and optionally their variances). Then, it creates an independent
        SingleTaskGP model for each output dimension (including objectives and constraints).
        The SingleTaskGP are finally combined into a ModelListGP to jointly represent the full
        multi-output model.

        Important: The GP model in BoTorch is a pure regression model: it simply fits the data
        it receives. It does not know or care about whether the model is for objectives to
        minimize or maximize.

        Note: by setting an input transform and an outcome transform, input (X) and output data (Y) are
        transformed and untransformed accordingly across the whole optimization pipeline, including
        the optimization of the acquisition function. For example, by setting the outcome transform to
        standardization, the Ys are standardized before the optimization and unstandardized right after.
        However, if a penalty is added in the forward method, this is not standardized properly and a
        pre-factor (or scaling) results in different penalty weights.

        Note: the kernel instance must be deep-copied, otherwise the same kernel is shared across
        all the models. This is critical in multi objective problems. """

        if verbose:
            print("Initializing model... ", end="")

        # Prepare dataset by concatenating the objectives and initialize models - one model
        # for each objective (or observable). Note that a base noise of 1e-4 is the default value
        train_x, train_y, train_y_var = self._prepare_training_dataset()
        models = []
        for i in range(0, train_y.shape[-1]):
            models.append(
                SingleTaskGP(
                    train_X=train_x,
                    train_Y=train_y[..., i: i + 1],
                    train_Yvar=(train_y_var[..., i: i + 1] if train_y_var is not None else None),
                    input_transform=Normalize(d=self.objective.dim, bounds=self.objective.bounds),
                    outcome_transform=Standardize(m=1),
                    covar_module=copy.deepcopy(self._kernel),
                    # Passing an explicit likelihood makes BoTorch ignore train_Yvar entirely, so
                    # the fixed-noise path has to leave it None and let BoTorch build the
                    # FixedNoiseGaussianLikelihood itself. The noise floor only bites when the
                    # noise is learned, so nothing is lost by dropping it there.
                    likelihood=(None if train_y_var is not None else
                                gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-4))),
                )
            )

        self._model = ModelListGP(*models)
        self._mll = SumMarginalLogLikelihood(self._model.likelihood, self._model)

        if verbose:
            self._print_success()

    def _prepare_training_dataset(self):
        """ Prepare the training dataset for fitting the surrogate model. """
        train_x = self.X.clone()
        train_y = self._Y_obj.clone()

        # Concatenate constraints if available (not None)
        if self._Y_con is not None:
            train_y_con = self._Y_con.clone()
            train_y = torch.cat((train_y, train_y_con), dim=-1)

        # Define train_y_var
        if self._Y_obj_var is None and self._Y_con_var is None:
            train_var = None
        else:
            # Create tensor filled with a tiny base noise instead of uninitialised memory
            train_var = torch.full(train_y.shape, 1e-6, dtype=train_y.dtype, device=train_y.device)
            # Map objective variances
            if self._Y_obj_var is not None:
                train_var[:, :self._Y_obj.shape[-1]] = self._Y_obj_var.clone()
            # Map constraint variances
            if self._Y_con_var is not None:
                train_var[:, self._Y_obj.shape[-1]:] = self._Y_con_var.clone()

        return train_x, train_y, train_var

    def _fit_model(self, verbose=True):
        """Fit the surrogate models, letting BoTorch own the retry budget.

        BoTorch retries internally: attempt 1 starts from the models' initial
        hyperparameters and each retry resamples them via sample_all_priors. The
        budget is per output model, so a constraint GP that struggles retries on
        its own instead of discarding the objective GPs that already converged -
        which is what an outer loop around fit_gpytorch_mll cannot avoid doing.
        """
        if not isinstance(self._model, ModelListGP):
            raise ValueError("Model must be initialised before fitting.")

        if verbose:
            print(f"Fitting model (up to {self._n_model_fit_restarts} attempts per output)... ", end="")

        try:
            botorch.fit_gpytorch_mll(self._mll, max_attempts=self._n_model_fit_restarts)
        except Exception as e:
            raise RuntimeError(
                f"Fitting failed after {self._n_model_fit_restarts} attempts per output model. Last error: {e}"
            ) from e

        if verbose:
            self._print_success()

    def _initialize_acquisition_function(self, verbose=True):
        """ Initialize an acquisition function instance using the acqf. """
        with warnings.catch_warnings(record=True) as caught:

            sig = inspect.signature(self.acqf.__init__)
            kwargs = {}

            # Initialize sampler if required by the acqf
            if issubclass(self._acqf, MCAcquisitionFunction):
                self._initialize_sampler(verbose=verbose)

            # Initialize partitioning if required by the acqf
            if "partitioning" in sig.parameters and getattr(self, "_partitioning", None) is None:
                self._initialize_partitioning(verbose=verbose)

            if verbose:
                name = self.acqf.__name__
                print(f"Initializing acquisition function of type {name}... ", end="", flush=True)

            for name, param in sig.parameters.items():
                if name in ('self', 'args', 'kwargs'):
                    continue

                if hasattr(self, name):
                    kwargs[name] = getattr(self, name)
                elif hasattr(self.objective, name):
                    kwargs[name] = getattr(self.objective, name)
                elif param.default is not inspect.Parameter.empty:
                    kwargs[name] = param.default

            self._acqf_instance = self.acqf(**kwargs)

        self._warnings.extend(caught)

        if verbose:
            self._print_success()
            self._print_caught_warnings()

        self._warnings = []

    def _initialize_partitioning(self, verbose=True):
        if verbose:
            print("Initializing partitioning... ", end="")

        if self._ref_point is None:
            raise AttributeError("No reference point is defined.")

        # Make sure feasibility mask exists
        if getattr(self, "_feasible_mask", None) is None:
            self._feasible_mask = self._compute_feasible_output_mask()

        # Use observed objective values ONLY
        Y_obj = self._Y_obj[self._feasible_mask]  # (n_feas, M)
        if Y_obj.numel() == 0:
            self._partitioning = None
            return

        # Convert to maximization space
        Y = Y_obj.clone()
        Y[..., self.objective.to_minimize] *= -1

        self._partitioning = NondominatedPartitioning(
            ref_point=self._ref_point,
            Y=Y
        )

        if verbose:
            self._print_success()

    def _initialize_sampler(self, verbose=True):
        """Initialize sampler for Monte Carlo acquisition functions."""
        if verbose:
            print("Initializing sampler... ", end="")

        self._sampler = SobolQMCNormalSampler(torch.Size([self._num_mc_samples]))

        if verbose:
            self._print_success()

    def _optimize_acquisition_function(self, verbose=True):

        if verbose:
            print(f"Optimizing acquisition function... ", end="")

        with warnings.catch_warnings(record=True) as caught:

            if isinstance(self._acqf_instance, qLogNParEGO):
                self._new_X, _ = optimize_acqf_list(
                    acq_function_list=self._acquisition_function_list,
                    bounds=self._objective.bounds,
                    num_restarts=self._n_acqf_opt_restarts,
                    raw_samples=self._num_raw_samples,
                    options={"batch_limit": 5, "maxiter": self._n_acqf_opt_max_iter},
                )
            else:
                # If nonlinear inequality **input** constraints are provided, use a custom initial condition
                # generator that selects "num_restarts" points. These points are distributed according to
                # "fraction_of_previous_X" between the optimal points and randomly generated points.
                self._new_X, _ = optimize_acqf(
                    acq_function=self._acqf_instance,
                    bounds=self._objective.bounds,
                    q=self._batch_size,
                    num_restarts=self._n_acqf_opt_restarts,
                    raw_samples=self._num_raw_samples,
                    options={"maxiter": self._n_acqf_opt_max_iter, "disp": False},
                    inequality_constraints=self.objective.lin_ineq_X_con,
                    equality_constraints=self.objective.lin_eq_X_con,
                    nonlinear_inequality_constraints=self.objective.nonlin_ineq_X_con,
                    sequential=True,
                    ic_generator=self._ic_generator
                    if self.objective.nonlin_ineq_X_con is not None
                    else None,
                    **{
                        "fraction_of_previous_X": 0.8,
                        "noise_scale": 0,
                    } if self.objective.nonlin_ineq_X_con is not None
                    else {}
                )

        self._warnings.extend(caught)

        if verbose:
            self._print_success(msg=f"New X: {self._new_X.detach().cpu().numpy()}")
            self._print_caught_warnings()

        self._warnings = []

    def _ic_generator(
            self,
            acq_function,  # noqa
            q: int,  # noqa
            num_restarts: int,
            raw_samples: int,
            **kwargs: dict
    ) -> torch.Tensor:
        """
        Generates initial conditions for constrained acquisition optimization.
        Mixes previous points and new samples from a constraint-aware sampler.
        """

        frac_prev = kwargs.get("fraction_of_previous_X", 0.5)
        noise_scale = kwargs.get("noise_scale", 0.0)
        sampler = SobolSampler(device=self._device, dtype=self._dtype, objective=self.objective)

        # 1. Collect previous observations
        X_feas, Y_feas = self._compute_feasible_XY()

        if X_feas is not None:
            X = X_feas.clone()
            Y = Y_feas.clone()
            # Adjust for minimization (BoTorch assumes maximization)
            Y[..., self.objective.to_minimize] *= -1
            n_requested = int(frac_prev * num_restarts)

            # LOGIC SPLIT: Multi-Objective vs Single-Objective
            if self.objective.num_obj > 1:
                # Multi-objective: use Pareto front
                mask = is_non_dominated(Y)
                X_candidates = X[mask]
                n_to_take = min(int(frac_prev * num_restarts), X_candidates.shape[0])
                idx = torch.randperm(X_candidates.shape[0])[:n_to_take]
                prev_points = X_candidates[idx]
            else:
                # Single-objective: use feasible top performing points
                _, indices = torch.sort(Y.squeeze(-1), descending=True)
                n_to_take = min(n_requested, indices.shape[0])
                prev_points = X[indices[:n_to_take]]
                shuffle_idx = torch.randperm(prev_points.shape[0])
                prev_points = prev_points[shuffle_idx]
        else:
            # No feasible points found in history yet
            prev_points = torch.empty(0, self.objective.dim, device=self._device, dtype=self._dtype)

        # 2. Generate new points (Exploration)
        n_new = max(0, num_restarts - prev_points.shape[0])
        raw_X = sampler.draw_samples(n=raw_samples)

        if n_new > raw_X.shape[0]:
            raise RuntimeError(
                f"Requested {n_new} new points, but only {raw_X.shape[0]} feasible raw samples available.")
        idx = torch.randperm(raw_X.shape[0])[:n_new]
        new_points = raw_X[idx]

        # Add small noise for exploration
        if noise_scale > 0:
            new_points += noise_scale * torch.randn_like(new_points)

        # 3. Combine previous points and new points
        combined = torch.cat([prev_points, new_points], dim=0)

        # Ensure proper shape for BoTorch: (num_restarts, q, d)
        return combined.unsqueeze(1)  # q=1 for sequential optimization

    """ ================== """
    """ ===== PROBES ===== """
    """ ================== """

    def compute_acquisition_function_value_at_X(self, X: torch.Tensor, verbose=True):
        if not isinstance(X, torch.Tensor):
            raise ValueError("X must be of type torch.Tensor.")
        acq_val = self._acqf_instance(X)
        if verbose:
            print(f"Acquisition function value at {X.detach().cpu().numpy()}: {acq_val.detach().cpu().numpy()}")

    def compute_posterior_mean_at_X(self, X: torch.Tensor, verbose=True):
        if not isinstance(X, torch.Tensor):
            raise ValueError("X must be of type torch.Tensor.")
        posterior = self._model.posterior(X)
        if verbose:
            print(f"Posterior mean at {X.detach().cpu().numpy()}: {posterior.mean.detach().cpu().numpy()}")
        return posterior

    def compute_posterior_variance_at_X(self, X: torch.Tensor, verbose=True):
        if not isinstance(X, torch.Tensor):
            raise ValueError("X must be of type torch.Tensor.")
        posterior_var = self._model.posterior(X).variance
        if verbose:
            print(f"Posterior variance at {X.detach().cpu().numpy()}: {posterior_var.detach().cpu().numpy()}")
        return posterior_var
