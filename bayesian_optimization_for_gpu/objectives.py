from abc import ABC
from botorch.acquisition.multi_objective import IdentityMCMultiOutputObjective, MCMultiOutputObjective


class IdentityMCMultiOutputObjectiveWrapper:
    def __init__(self, outcomes_indexes=(0, 1)):
        self.outcomes_indexes = outcomes_indexes

    def __call__(self):
        IdentityMCMultiOutputObjective(outcomes=self.outcomes_indexes),


class Avagama(MCMultiOutputObjective, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, X):
        return -X[:, 0] * X[:, 1]