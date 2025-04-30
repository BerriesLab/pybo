import datetime
import os
import glob

import numpy as np
from pathlib import Path


def create_experiment_directory(main_directory: Path or str, experiment_name: str):
    if isinstance(main_directory, str):
        main_directory = Path(main_directory)
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory = main_directory / Path(f'{date_time} - {experiment_name}')
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_dataset_to_csv(data):
    filepath = compose_dataset_filename()
    np.savetxt(filepath, data, delimiter=",", comments="")


def load_dataset_from_csv(filepath: str or None = None,):
    if filepath is None:
        csv_files = list(Path('.').glob('*.csv'))
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the current directory")
        filepath = max(csv_files, key=lambda x: x.stat().st_mtime)

    return np.loadtxt(filepath, delimiter=",")

def compose_filename():
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f'{date_time}'


def compose_model_filename():
    return compose_filename() + ".dat"


def compose_figure_filename():
    return compose_filename() + ".png"

def compose_dataset_filename():
    return compose_filename() + ".csv"