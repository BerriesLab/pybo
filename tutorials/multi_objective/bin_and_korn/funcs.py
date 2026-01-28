import torch


def f1(X: torch.Tensor) -> torch.Tensor:
    x1, x2 = X[..., 0], X[..., 1]
    return 4.0 * x1 ** 2 + 4.0 * x2 ** 2


def f2(X: torch.Tensor) -> torch.Tensor:
    x1, x2 = X[..., 0], X[..., 1]
    return (x1 - 5.0) ** 2 + (x2 - 5.0) ** 2


def c1(X: torch.Tensor) -> torch.Tensor:
    """ A constraint on the input: (x0 - 5)^2 + x1^2 <= 25 """
    x1, x2 = X[..., 0], X[..., 1]
    return 25.0 - ((x1 - 5.0) ** 2 + x2 ** 2)  # >=0 feasible


def c2(X: torch.Tensor) -> torch.Tensor:
    """ A constraint on the input: (x0 - 8)^2 + (x1 + 3)^2 >= 7.7 """
    x1, x2 = X[..., 0], X[..., 1]
    return (x1 - 8.0) ** 2 + (x2 + 3.0) ** 2 - 7.7
