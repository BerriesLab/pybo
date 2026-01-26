import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


""" Functions to create a 2nd order polynomial ground truth by fitting the prior belief. """


def build_csv_from_json(filepath: str or Path):

    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    # Load JSON file
    with open(filepath, "r") as f:
        data = json.load(f)

    # Extract parameters and results into a flat list of dictionaries
    records = []
    for exp in data:
        if "parameters" in exp and "results" in exp:
            record = {
                "experiment_id": exp.get("experiment_id"),
                "experiment_type": exp.get("experiment_type"),
                "I_MAX": exp["parameters"].get("I_MAX"),
                "I_P": exp["parameters"].get("I_P"),
                "tau_R_max": exp["parameters"].get("tau_R_max"),
                "wear_microns": exp["results"].get("wear_microns"),
                "down_time_minutes": exp["results"].get("down_time_minutes"),
                "orbiting_time_minutes": exp["results"].get("orbiting_time_minutes")
            }
            records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    # Save to CSV
    df.to_csv(filepath.parent / "avagama_experiments.csv", index=False)

    print("CSV file saved as 'experiments_summary.csv'")

def fit_polynomial_models(csv_path):
    """
    Fit second-order polynomial models for wear, down_time, and orbiting_time
    as a function of I_MAX, I_P, and tau_R_max.

    Parameters:
        csv_path (str): Path to the CSV file containing the experiment data.

    Returns:
        dict: A dictionary of fitted LinearRegression models.
    """
    # Load data
    df = pd.read_csv(csv_path)
    df = df.dropna()

    # Input features and targets
    X = df[["I_MAX", "I_P", "tau_R_max"]].values
    y_targets = {
        "wear_microns": df["wear_microns"].values,
        "down_time_minutes": df["down_time_minutes"].values,
        "orbiting_time_minutes": df["orbiting_time_minutes"].values,
    }

    # Generate second-order polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    models = {}
    for target_name, y in y_targets.items():
        model = LinearRegression()
        model.fit(X_poly, y)
        models[target_name] = model

    print("Polynomial models fitted successfully.")
    return models, poly

def plot_polynomial_surface(models, poly, resolution=30):
    """
    Plot the polynomial fit using:
    - x-axis: down_time_minutes
    - y-axis: wear_microns
    - colormap: orbiting_time_minutes

    Parameters:
        models (dict): Dictionary containing fitted models for each target.
        poly (PolynomialFeatures): The fitted polynomial transformer.
        resolution (int): Grid resolution for each input variable.
    """
    # Define a mesh grid of I_MAX, I_P, tau_R_max values
    I_MAX_range = np.linspace(7.5, 15, resolution)
    I_P_range = np.linspace(3, 7.5, resolution)
    tau_R_max_range = np.linspace(0.1, 78, resolution)

    # Create a mesh of all input combinations
    grid = np.array(np.meshgrid(I_MAX_range, I_P_range, tau_R_max_range)).T.reshape(-1, 3)
    X_poly = poly.transform(grid)

    # Predict with all three models
    wear_pred = models["wear_microns"].predict(X_poly)
    down_time_pred = models["down_time_minutes"].predict(X_poly)
    orbiting_time_pred = models["orbiting_time_minutes"].predict(X_poly)

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(down_time_pred, wear_pred, c=orbiting_time_pred, cmap="viridis", alpha=0.3)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Orbiting Time (minutes)", fontsize=12)

    plt.xlabel("Down Time (minutes)", fontsize=12)
    plt.ylabel("Wear (microns)", fontsize=12)
    plt.title("Wear vs Down Time (Color: Orbiting Time)", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_3d_model_surface(models, poly, resolution=30):
    """
    Plot the fitted second-order polynomial model in 3D:
    - X-axis: down_time_minutes
    - Y-axis: wear_microns
    - Z-axis: orbiting_time_minutes

    Parameters:
        models (dict): Dictionary of fitted models from fit_polynomial_models().
        poly (PolynomialFeatures): Polynomial transformer used to fit the models.
        resolution (int): Number of points per axis in the 3D grid.
    """
    # Define a grid of input parameters (I_MAX, I_P, tau_R_max)
    I_MAX_range = np.linspace(6, 15, resolution)
    I_P_range = np.linspace(3, 8, resolution)
    tau_R_max_range = np.linspace(10, 80, resolution)

    # Create meshgrid
    I_MAX_grid, I_P_grid, tau_R_max_grid = np.meshgrid(I_MAX_range, I_P_range, tau_R_max_range)
    grid_points = np.vstack([
        I_MAX_grid.ravel(),
        I_P_grid.ravel(),
        tau_R_max_grid.ravel()
    ]).T

    # Transform features
    X_poly = poly.transform(grid_points)

    # Predict using models
    down_time = models["down_time_minutes"].predict(X_poly)
    wear = models["wear_microns"].predict(X_poly)
    orbiting_time = models["orbiting_time_minutes"].predict(X_poly)

    # Plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(down_time, wear, orbiting_time, c=orbiting_time, cmap='viridis', s=10)

    ax.set_xlabel("Down Time (min)", fontsize=12)
    ax.set_ylabel("Wear (µm)", fontsize=12)
    ax.set_zlabel("Orbiting Time (min)", fontsize=12)
    ax.set_title("Model Surface: Wear vs Down Time vs Orbiting Time", fontsize=14)

    plt.tight_layout()
    plt.show()



build_csv_from_json("avagama_experiments.json")
models, poly = fit_polynomial_models("avagama_experiments.csv")
plot_polynomial_surface(models, poly)
# plot_3d_model_surface(models, poly)