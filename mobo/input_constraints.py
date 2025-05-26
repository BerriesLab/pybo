import torch


def c1(x: torch.Tensor):
    """ c1 is a constraint on the input variables """
    return torch.square((x[..., -2] - 5)) + torch.square(x[..., -1]) - 25


def c2(x: torch.Tensor):
    """ c2 is a constraint on the input variables """
    return 7.7 - torch.square((x[..., -2] - 8)) - torch.square((x[..., -1] + 3))
