import datetime


def compose_filepath(directory, experiment_name, datetime, n_samples, previous=False):
    filename = compose_filename(experiment_name, datetime, n_samples, previous)
    filepath = f"{directory}/{filename}"
    return filepath


def compose_filename(experiment_name: str, datetime: datetime.datetime, n_samples: int, previous: bool =False):
    if previous:
        filename = f'{experiment_name} - {datetime.strftime("%Y-%m-%d_%H-%M-%S")} - {n_samples - 1} samples'
    else:
        filename = f'{experiment_name} - {datetime.strftime("%Y-%m-%d_%H-%M-%S")} - {n_samples} samples'
    return filename