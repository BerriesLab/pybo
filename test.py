import datetime

import numpy as np


class Experiment:
    """ A class designed to collect experimental data and instrumentation
    settings in a single object. """

    def __init__(self,
                 main_path: str,
                 experiment_id: str,
                 datetime: datetime.datetime,
                 ):
        self.main_path = main_path
        self.experiment_id = experiment_id
        self.datetime = datetime
        self.data = None


class TimeSeries:
    """ A class for time series data """

    def __init__(self,
                 t: np.array,
                 y: np.array
                 ):
        self.t = t
        self.y = y

    # TODO: a method to linearly interpolate null values between the closest know values

# class Settings:
#     def __init__(self):
