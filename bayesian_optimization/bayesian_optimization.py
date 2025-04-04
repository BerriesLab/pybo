import matplotlib.pyplot as plt
import numpy as np
from skopt import gp_minimize
from skopt.plots import plot_convergence


# Define the function to optimize
def objective_function(x):
    return (x[0] - 2) ** 2 + np.sin(5 * x[0]) + (x[1] + 1) ** 2 - np.cos(3 * x[1])


evaluated_points = []


def live_plot(optim_result):
    global evaluated_points
    evaluated_points.append(optim_result.x_iters[-1])

    plt.clf()
    points = np.array(evaluated_points)

    # Create potential map in the z-axis
    x = np.linspace(bounds[0][0], bounds[0][1], 100)
    y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array(
        [[objective_function([x_val, y_val]) for x_val, y_val in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Objective Function Value')

    if len(points) > 1:
        plt.scatter(points[:-1, 0], points[:-1, 1], c='blue', label='Evaluated Points')
        plt.scatter(points[-1, 0], points[-1, 1], c='green', marker='o', s=100, label='Current Point')
        plt.scatter(optim_result.x[0], optim_result.x[1], c='red', marker='*', s=200, label='Optimal Point')

        for i in range(len(points) - 1):
            dx = points[i + 1, 0] - points[i, 0]
            dy = points[i + 1, 1] - points[i, 1]
            plt.arrow(points[i, 0], points[i, 1], dx, dy, head_width=0.1, head_length=0.1, fc='gray', ec='gray',
                      alpha=0.6, length_includes_head=True)

    plt.xlabel('Wear')
    plt.ylabel('Machining Time')
    plt.legend()
    plt.title('Live Optimization Progress')
    plt.grid(True)
    plt.xlim(bounds[0])
    plt.ylim(bounds[1])
    plt.pause(0.2)


# Define the bounds of the search space
bounds = [(-2.0, 4.0), (-3.0, 3.0)]  # Search space for x and y

# Perform Bayesian Optimization
result = gp_minimize(objective_function, bounds, n_calls=50, random_state=42, callback=live_plot)

# Plot the convergence
plt.figure()
plot_convergence(result)
plt.show()

# Display the best found values
print(f"Best x: {result.x[0]:.4f}, Best y: {result.x[1]:.4f}")
print(f"Best function value: {result.fun:.4f}")