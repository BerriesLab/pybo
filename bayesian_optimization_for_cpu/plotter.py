import matplotlib.pyplot as plt
import numpy as np


def _plot(self):
    # Handle single-objective cases
    if self._n_objectives == 1:

        if len(self._bounds) == 1:
            self._plot_from_R1_to_R1()
        elif len(self._bounds) == 2:
            self._plot_from_R2_to_R1()
        else:
            raise ValueError("Only 1D and 2D plots are supported.")

    # Handle bi-objective cases
    elif self._n_objectives == 2:
        self.plot_multi_objective_from_RN_to_R2()

    # Handle tri-objective cases
    elif self._n_objectives == 3:
        self.plot_multi_objective_from_RN_to_R3()

    else:
        raise ValueError("Cannot display pareto front for more than 3-objectives without PCA.")


def _plot_from_R1_to_R1(self):
    # Initialize figure
    self._fig, self._ax = plt.subplots()
    self._ax.set_title(r'Bayesian Optimization for $f_0:\mathbb{R} \rightarrow \mathbb{R}$')
    self._ax.set_xlabel(r"$\mathcal{X}$")
    self._ax.set_ylabel(r"$\mathcal{Y}$")
    self._ax.set_xlim(self._bounds[0][0] * 1.1, self._bounds[0][1] * 1.1)

    # Build domain
    self._build_domain_from_bounds()

    # Plot objective function if known
    if self._f0:
        self._ax.plot(self._domain, self._f0[0](self._domain), color="black", linestyle="--", label=r"True $f_0$")

    # Plot posterior
    self._predict_gaussian_process_on_domain()
    self._ax.plot(self._domain, self._mu[0], label="Mean", zorder=2)
    for i in range(1, 4):
        self._ax.fill_between(x=self._domain.flatten(),
                              y1=self._mu[0] - i * self._sigma[0],
                              y2=self._mu[0] + i * self._sigma[0],
                              alpha=0.2 / i,
                              color="blue",
                              label=rf"{i}$\sigma$")

    # Plot new location
    y_min, y_max = self._ax.get_ylim()
    self._ax.vlines(self._new_X, ymin=y_min, ymax=y_max, color='red', alpha=0.3, linestyle="--", label="New X")

    # Plot all observations except minimum
    min_y_idx = np.argmin(self._Yobj)
    mask = np.ones(len(self._Yobj), dtype=bool)
    mask[min_y_idx] = False
    self._ax.scatter(self._X[mask], self._Yobj[mask], marker="o", s=50, color='red', label='Observations')

    # Plot minimum Y value
    self._ax.scatter(self._X[min_y_idx], self._Yobj[min_y_idx], marker='*', s=200, color='green', label='Min Y')
    self._ax.legend()


def _plot_from_R2_to_R1(self):
    # Initialize figure
    self._fig = plt.figure()
    self._ax = self._fig.add_subplot(111, projection='3d')
    self._ax.set_title(r'Bayesian Optimization for $f_0:\mathbb{R}^2 \rightarrow \mathbb{R}$')
    self._ax.set_xlabel('$x_1$')
    self._ax.set_ylabel('$x_2$')
    self._ax.set_zlabel('$f(\mathcal{X})$')
    self._ax.set_xlim(self._bounds[0][0], self._bounds[0][1])
    self._ax.set_ylim(self._bounds[1][0], self._bounds[1][1])

    # Build domain
    self._build_domain_from_bounds()

    # Plot objective function if known
    if self._f0 and all([isinstance(f, Callable) for f in self._f0]):
        x_grid = np.linspace(self._bounds[0][0], self._bounds[0][1], 100)
        y_grid = np.linspace(self._bounds[1][0], self._bounds[1][1], 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        points = np.c_[X.ravel(), Y.ravel()]
        Z = self._f0[0](points).reshape(X.shape)
        self._ax.plot_wireframe(X, Y, Z, lw=0.5, alpha=0.4, color='black', label=r"True $f_0$")

    # Plot new X location with vertical line
    z_line = np.linspace(0, self._ax.get_zlim()[1], 100)
    x_line = np.full_like(z_line, self._new_X[0, 0])
    y_line = np.full_like(z_line, self._new_X[0, 1])
    self._ax.plot(x_line, y_line, z_line, 'r--', alpha=0.5, label='New Location')

    # Plot all observations except minimum
    min_y_idx = np.argmin(self._Yobj)
    mask = np.ones(len(self._Yobj), dtype=bool)
    mask[min_y_idx] = False
    self._ax.scatter(self._X[mask, 0], self._X[mask, 1], self._Yobj[mask, 0], c='red', marker='o', s=50,
                     label='Observations')

    # Plot minimum Y value
    mask = np.invert(mask)
    self._ax.scatter(self._X[mask, 0], self._X[mask, 1], self._Yobj[mask, 0], c='green', marker='*', s=200, label='Min Y')

    # Plot posterior
    self._predict_gaussian_process_on_domain()
    x_grid = self._domain[:, 0]
    y_grid = self._domain[:, 1]
    X, Y = np.meshgrid(x_grid, y_grid)
    for i in range(1, 4):
        self._ax.plot_surface(X, Y, (self._mu[0] - i * self._sigma[0]).reshape(X.shape),
                              lw=0.5, rstride=8, cstride=8,
                              alpha=0.2, cmap='coolwarm',
                              zorder=2)
        self._ax.plot_surface(X, Y, (self._mu[0] + i * self._sigma[0]).reshape(X.shape),
                              lw=0.5, rstride=8, cstride=8,
                              alpha=0.2, cmap='coolwarm',
                              zorder=3)


def _plot_from_RN_to_R2(self):
    # Initialize figure
    self._fig = plt.figure()
    self._ax = self._fig.add_subplot()
    self._ax.set_title(r'Multi objective Bayesian Optimization for $\mathbf{f_0}:\mathbb{R}^N \rightarrow '
                       r'\mathbb{R}^2$')
    self._ax.set_xlabel('$f_{01}$')
    self._ax.set_ylabel('$f_{02}$')
    # Plot observations
    self._ax.scatter(self._Yobj[:, 0], self._Yobj[:, 1], marker="o", s=50, color='red', label='Observations')
    # Plot Pareto Front
    self._ax.scatter(self._pareto_front[:, 0], self._pareto_front[:, 1], marker="x", s=50, color='olive',
                     label='Pareto Front')
    plt.legend()


def _plot_from_RN_to_R3(self):
    raise ValueError("Multi-objective plot for 3 objectives is not supported yet")