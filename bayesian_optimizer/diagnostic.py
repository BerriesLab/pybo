from typing import Dict, List, Tuple
import torch
import matplotlib.pyplot as plt
import numpy as np
from botorch.models.model import Model


class ModelDiagnostics:
    """
    A class for evaluating and diagnosing Gaussian Process model fit quality.

    Key features:
    - Hyperparameter sanity checks
    - Visual diagnostics (residuals, calibration, predictions)
    - Quantitative metrics (MSE, NLPD, R²)
    - Overall fit assessment
    """

    def __init__(
            self,
            model: Model,
            X_train: torch.Tensor,
            Y_train: torch.Tensor,
            bounds: torch.Tensor,
    ):
        """
        :param model: Fitted GP model (e.g., SingleTaskGP)
        :param X_train: Training inputs (n x d)
        :param Y_train: Training outputs (n x 1)
        :param bounds: Input bounds (2 x d)
        """
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.bounds = bounds

        # Compute predictions once
        with torch.no_grad():
            posterior = self.model.posterior(X_train)
            self.pred_mean = posterior.mean.squeeze()
            self.pred_var = posterior.variance.squeeze()
            self.pred_std = self.pred_var.sqrt()

    @property
    def lengthscale(self) -> torch.Tensor:
        return self.model.covar_module.base_kernel.lengthscale.detach()

    @property
    def outputscale(self) -> float:
        return self.model.covar_module.outputscale.item()

    @property
    def noise(self) -> float:
        return self.model.likelihood.noise.item()

    @property
    def domain_size(self) -> torch.Tensor:
        return self.bounds[1] - self.bounds[0]

    def get_hyperparameters(self) -> Dict[str, float]:
        """Return dictionary of current hyperparameters."""
        return {
            "lengthscale": self.lengthscale.cpu().numpy(),
            "outputscale": self.outputscale,
            "noise": self.noise,
        }

    def print_hyperparameters(self) -> None:
        """Print current hyperparameters."""
        print("=" * 40)
        print("GP Hyperparameters")
        print("=" * 40)
        print(f"Lengthscale: {self.lengthscale.cpu().numpy()}")
        print(f"Outputscale: {self.outputscale:.6f}")
        print(f"Noise:       {self.noise:.6f}")
        print(f"Domain size: {self.domain_size.cpu().numpy()}")
        print("=" * 40)

    def check_hyperparameters(self, verbose: bool = True) -> List[str]:
        """
        Check hyperparameters for common issues.

        :param verbose: If True, print warnings
        :return: List of warning messages (empty if all checks pass)
        """
        warnings = []
        domain_max = self.domain_size.max().item()

        # Lengthscale checks (per dimension if ARD)
        lengthscale = self.lengthscale.squeeze()
        if lengthscale.ndim == 0:
            lengthscale = lengthscale.unsqueeze(0)

        for i, ls in enumerate(lengthscale):
            ls_val = ls.item()
            dim_size = self.domain_size[i].item() if i < len(self.domain_size) else domain_max

            if ls_val > 2 * dim_size:
                warnings.append(
                    f"⚠️ Lengthscale[{i}] = {ls_val:.4f} too large "
                    f"(> 2× domain size {dim_size:.2f}) → flat mean, no local structure"
                )
            elif ls_val < 0.01 * dim_size:
                warnings.append(
                    f"⚠️ Lengthscale[{i}] = {ls_val:.4f} too small "
                    f"(< 0.01× domain size {dim_size:.2f}) → overfitting, wiggly mean"
                )

        # Outputscale checks
        if self.outputscale < 1e-4:
            warnings.append(
                f"⚠️ Outputscale = {self.outputscale:.6f} too small → collapsed variance"
            )

        y_var = self.Y_train.var().item()
        if self.outputscale > 100 * y_var:
            warnings.append(
                f"⚠️ Outputscale = {self.outputscale:.4f} much larger than "
                f"Y variance = {y_var:.4f} → potential issue"
            )

        # Noise checks
        if self.noise > self.outputscale:
            warnings.append(
                f"⚠️ Noise = {self.noise:.6f} > Outputscale = {self.outputscale:.6f} "
                f"→ model thinks it's all noise"
            )

        if self.noise < 1e-6:
            warnings.append(
                f"⚠️ Noise = {self.noise:.8f} too small → potential numerical issues"
            )

        if verbose:
            if warnings:
                print("\n".join(warnings))
            else:
                print("✓ All hyperparameter checks passed")

        return warnings

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute quantitative fit metrics.

        :return: Dictionary with MSE, NLPD, R², and standardized residual stats
        """
        Y = self.Y_train.squeeze()

        # Residuals
        residuals = Y - self.pred_mean

        # Mean Squared Error
        mse = (residuals ** 2).mean().item()

        # Root Mean Squared Error
        rmse = np.sqrt(mse)

        # Negative Log Predictive Density (lower is better)
        nlpd = 0.5 * (
                torch.log(2 * np.pi * self.pred_var) + residuals ** 2 / self.pred_var
        ).mean().item()

        # R² score
        ss_res = (residuals ** 2).sum()
        ss_tot = ((Y - Y.mean()) ** 2).sum()
        r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0

        # Standardized residuals
        std_residuals = (residuals / self.pred_std).cpu().numpy()

        return {
            "mse": mse,
            "rmse": rmse,
            "nlpd": nlpd,
            "r2": r2,
            "std_residuals_mean": np.mean(std_residuals),
            "std_residuals_std": np.std(std_residuals),
        }

    def print_metrics(self) -> Dict[str, float]:
        """Compute and print fit metrics."""
        metrics = self.compute_metrics()

        print("=" * 40)
        print("Fit Metrics")
        print("=" * 40)
        print(f"MSE:  {metrics['mse']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"NLPD: {metrics['nlpd']:.4f}")
        print(f"R²:   {metrics['r2']:.4f}")
        print(f"Standardized residuals: mean={metrics['std_residuals_mean']:.3f}, "
              f"std={metrics['std_residuals_std']:.3f} (should be ~0, ~1)")
        print("=" * 40)

        return metrics

    def plot_diagnostics(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Create diagnostic plots.

        :param figsize: Figure size
        :return: matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        Y_np = self.Y_train.squeeze().cpu().numpy()
        mean_np = self.pred_mean.cpu().numpy()
        std_np = self.pred_std.cpu().numpy()
        residuals = Y_np - mean_np
        std_residuals = residuals / std_np

        # 1. Predicted vs Observed
        ax = axes[0, 0]
        ax.scatter(Y_np, mean_np, alpha=0.7, edgecolors='black', linewidths=0.5)
        lims = [min(Y_np.min(), mean_np.min()), max(Y_np.max(), mean_np.max())]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect fit')
        ax.set_xlabel('Observed')
        ax.set_ylabel('Predicted')
        ax.set_title('Predicted vs Observed')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Residuals vs Predicted
        ax = axes[0, 1]
        ax.scatter(mean_np, residuals, alpha=0.7, edgecolors='black', linewidths=0.5)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals (should be random around 0)')
        ax.grid(True, alpha=0.3)

        # 3. Standardized residuals histogram
        ax = axes[1, 0]
        ax.hist(std_residuals, bins=20, density=True, alpha=0.7, edgecolor='black')

        # Overlay N(0,1)
        x = np.linspace(-4, 4, 100)
        ax.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), 'r-', linewidth=2, label='N(0,1)')
        ax.set_xlabel('Standardized Residuals')
        ax.set_ylabel('Density')
        ax.set_title('Standardized Residuals (should be ~N(0,1))')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Uncertainty across domain (1D only, otherwise vs index)
        ax = axes[1, 1]
        if self.X_train.shape[1] == 1:
            X_np = self.X_train.squeeze().cpu().numpy()
            ax.scatter(X_np, std_np, alpha=0.7, edgecolors='black', linewidths=0.5)
            ax.set_xlabel('X')
        else:
            ax.scatter(range(len(std_np)), std_np, alpha=0.7, edgecolors='black', linewidths=0.5)
            ax.set_xlabel('Observation Index')
        ax.set_ylabel('Posterior Std')
        ax.set_title('Uncertainty at Training Points')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def is_fit_acceptable(
            self,
            min_r2: float = 0.5,
            max_lengthscale_factor: float = 2.0,
            min_lengthscale_factor: float = 0.01,
            verbose: bool = True,
    ) -> bool:
        """
        Check if the overall fit is acceptable.

        :param min_r2: Minimum acceptable R² score
        :param max_lengthscale_factor: Max lengthscale as factor of domain size
        :param min_lengthscale_factor: Min lengthscale as factor of domain size
        :param verbose: If True, print results
        :return: True if fit is acceptable
        """
        checks = {}
        domain_max = self.domain_size.max().item()

        # Lengthscale check
        lengthscale = self.lengthscale.squeeze()
        if lengthscale.ndim == 0:
            lengthscale = lengthscale.unsqueeze(0)

        ls_ok = all(
            min_lengthscale_factor * domain_max < ls.item() < max_lengthscale_factor * domain_max
            for ls in lengthscale
        )
        checks["lengthscale_in_range"] = ls_ok

        # Outputscale check
        checks["outputscale_positive"] = self.outputscale > 1e-4

        # Noise vs signal check
        checks["noise_less_than_signal"] = self.noise < self.outputscale

        # R² check
        metrics = self.compute_metrics()
        checks["r2_acceptable"] = metrics["r2"] > min_r2

        # Standardized residuals check
        checks["residuals_calibrated"] = (
                abs(metrics["std_residuals_mean"]) < 0.5 and
                0.5 < metrics["std_residuals_std"] < 2.0
        )

        all_passed = all(checks.values())

        if verbose:
            print("=" * 40)
            print("Fit Assessment")
            print("=" * 40)
            for check, passed in checks.items():
                status = "✓" if passed else "✗"
                print(f"{status} {check}")
            print("=" * 40)
            print(f"Overall: {'ACCEPTABLE' if all_passed else 'NEEDS ATTENTION'}")
            print("=" * 40)

        return all_passed

    def full_report(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Generate a full diagnostic report.

        :param figsize: Figure size for plots
        :return: Tuple of (is_acceptable, metrics, fig)
        """
        print("\n" + "=" * 50)
        print("MODEL DIAGNOSTICS REPORT")
        print("=" * 50 + "\n")

        self.print_hyperparameters()
        print()

        self.check_hyperparameters()
        print()

        metrics = self.print_metrics()
        print()

        is_ok = self.is_fit_acceptable()
        print()

        fig = self.plot_diagnostics(figsize=figsize)

        return is_ok, metrics, fig
