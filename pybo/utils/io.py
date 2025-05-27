import datetime
import numpy as np
from pathlib import Path
import torch


def create_experiment_directory(main_directory: Path or str, experiment_name: str):
    if isinstance(main_directory, str):
        main_directory = Path(main_directory)
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory = main_directory / Path(f'{date_time} - {experiment_name}')
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_dataset_to_csv(
        X: torch.Tensor,
        Yobj: torch.Tensor,
        Yobj_var: torch.Tensor or None,
        Ycon: torch.Tensor or None,
        Ycon_var: torch.Tensor or None,
        header: str,
):
    """
    Save input and target tensors to a CSV file.

    This function combines the provided tensors into a single tensor,
    transfers it to CPU, converts it to a NumPy array, and saves it
    to a CSV file. The file name is generated using the
    compose_dataset_filename function.
    """

    # Handle missing objective values cases
    if Yobj is None:
        Yobj = torch.zeros([X.shape[0], 1]).fill_(torch.nan)
    if Yobj_var is None:
        Yobj_var = torch.zeros_like(Yobj).fill_(torch.nan)

    if Ycon is not None:
        # Handle constrained problems
        if Ycon_var is None:
            Ycon_var = torch.zeros_like(Ycon).fill_(torch.nan)
        Y = torch.cat([Yobj, Yobj_var, Ycon, Ycon_var], dim=-1)
    else:
        # Handle unconstrained problems
        Y = torch.cat([Yobj, Yobj_var], dim=-1)

    XY = torch.cat([X, Y], dim=-1)
    XY = XY.detach().cpu().numpy()
    filepath = compose_dataset_filename()
    np.savetxt(filepath, XY, delimiter=",", comments="", )


def load_dataset_from_csv(
        input_space_dimension: int,  # input_space_dimension: int,
        objective_space_dimension: int,  # objective_space_dimension: int,
        constraint_space_dimension: int,  # constraint_space_dimension: int = 0,
        objective_variance: bool = False,
        constraint_variance: bool = False,
        filepath: str or None = None,
        skiprows: int = 0,
        skipcols: int = 0,
):
    """ Assumes that the dataset is saved in the CSV format and columns are ordered as follows:
        X ¦ Yobj ¦ Yobj_var ¦ Ycon ¦ Ycon_var."""

    if filepath is None:
        csv_files = list(Path('').glob('*.csv'))
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the current directory")
        filepath = max(csv_files, key=lambda x: x.stat().st_mtime)

    xy = np.loadtxt(filepath, delimiter=",", skiprows=skiprows)

    i = skipcols + 0
    j = skipcols + input_space_dimension
    X = torch.tensor(xy[..., i:j])

    if objective_space_dimension > 0:
        i = j
        j += objective_space_dimension
        Yobj = torch.tensor(xy[..., i:j])

        if objective_variance:
            i = j
            j += objective_space_dimension
            Yobj_var = torch.tensor(xy[..., i:j])
        else:
            Yobj_var = None
    else:
        Yobj = None
        Yobj_var = None

    if constraint_space_dimension > 0:
        i = j
        j += constraint_space_dimension
        Ycon = torch.tensor(xy[..., i:j])

        if constraint_variance:
            i = j
            j += constraint_space_dimension
            Ycon_var = torch.tensor(xy[..., i:j])
        else:
            Ycon_var = None
    else:
        Ycon = None
        Ycon_var = None

    return X, Yobj, Yobj_var, Ycon, Ycon_var


def compose_filename(iteration_number: int or None = None):
    if iteration_number is None:
        return ''
    else:
        return f'{iteration_number:04d}'


def compose_model_filename(iteration_number: int or None = None):
    return compose_filename(iteration_number) + ".dat"


def compose_figure_filename(iteration_number: int or None = None, postfix: str = ""):
    return compose_filename(iteration_number) + postfix + ".png"


def compose_dataset_filename(iteration_number: int or None = None):
    return compose_filename(iteration_number) + ".csv"
