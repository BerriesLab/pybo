import numpy as np


def test_f0_1d(x):
    return 0.1 * (x[:, 0] - 2) ** 3 + 2 * np.sin(5 * x[:, 0])  # * np.cos(3 * x[:, 0])


def test_f0_2d(x):
    return 1 / 2 * (x[:, 0] - 2) ** 2 + 2 * np.sin(2 * x[:, 0]) - x[:, 1] ** 2


def test_f0_3d(x):
    return (x[:, 0] - 2) ** 2 + np.sin(5 * x[:, 0]) + (x[:, 1] + 1) ** 3 - np.cos(3 * x[:, 2])


def test_f0_wear(x):
    i_0 = x[:, 0]
    i_max = x[:, 1]
    t_on = x[:, 2]
    return (i_max - i_0) * t_on / 2


def test_f0_machining_time(x):
    i_0 = x[:, 0]
    i_max = x[:, 1]
    t_on = x[:, 2]
    return 2 / (i_max - i_0) * t_on