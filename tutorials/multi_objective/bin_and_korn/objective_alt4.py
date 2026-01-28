import torch
from objectives.base_class import MCMultiObjectiveBase
from objectives.variable_registry import *


class BinhAndKorn(MCMultiObjectiveBase):
    """ Two objective problem composed of the Binh and Korn functions. """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(device=device, dtype=dtype)
