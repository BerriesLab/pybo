from botorch.acquisition.multi_objective import IdentityMCMultiOutputObjective


class IdentityMCMultiOutputObjectiveWrapper:
    def __init__(self, outcomes=(0, 1)):
        self.outcomes=outcomes

    def __call__(self):
        IdentityMCMultiOutputObjective(outcomes=self.outcomes),
