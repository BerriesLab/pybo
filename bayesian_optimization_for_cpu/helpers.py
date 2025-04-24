def _compose_filepath(self, directory, previous=False):
    filename = self._compose_filename(previous)
    filepath = f"{directory}/{filename}"
    return filepath


def _compose_filename(self, previous=False):
    if previous:
        filename = f'{self._experiment_name} - {self._datetime.strftime("%Y-%m-%d_%H-%M-%S")} - {self._X.shape[0] - 1} samples'
    else:
        filename = f'{self._experiment_name} - {self._datetime.strftime("%Y-%m-%d_%H-%M-%S")} - {self._X.shape[0]} samples'
    return filename


def _build_domain_from_bounds(self, n=None):
    if n is None:
        n = [100 for _ in range(len(self._bounds))]
    elif isinstance(n, int):
        n = [n for _ in range(len(self._bounds))]
    elif isinstance(n, list) and not len(n) == len(self._bounds):
        raise ValueError("The length of n must match the number of dimensions in bounds.")
    else:
        raise ValueError("n must be None, an integer, or a list of integers.")

    domain = []
    for i in range(len(self._bounds)):
        domain.append(np.linspace(self._bounds[i][0], self._bounds[i][1], n[i]).reshape(-1, 1))
    self._domain = np.concatenate(domain, axis=1)