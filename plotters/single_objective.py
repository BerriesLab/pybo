from plotters.base_class import PlotterBase


class SingleObjectivePlotter(PlotterBase):
    def __init__(self, title, labels):
        super().__init__(
            title=title,
            labels=labels,
        )

        raise NotImplementedError("SingleObjectivePlotter is not yet implemented.")
