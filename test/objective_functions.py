import numpy as np


def test_f0_1d(x):
    return 0.1 * (x[:, 0] - 2) ** 3 + 2 * np.sin(5 * x[:, 0])  # * np.cos(3 * x[:, 0])


def test_f0_2d(x):
    return 1 / 2 * (x[:, 0] - 2) ** 2 + 2 * np.sin(2 * x[:, 0]) - x[:, 1] ** 2


def test_f0_3d(x):
    return (x[:, 0] - 2) ** 2 + np.sin(5 * x[:, 0]) + (x[:, 1] + 1) ** 3 - np.cos(3 * x[:, 2])


def test_f0_wear(x):
    # x[:, 0] =  i_0
    # x[:, 1] =  i_max
    # x[:, 2] =  t_on
    A = 1
    B = 2
    C = 3
    return A * x[:, 0] + B * x[:, 1] + C * x[:, 2]


def test_f0_speed(x):
    # x[:, 0] =  i_0
    # x[:, 1] =  i_max
    # x[:, 2] =  t_on
    A = 4
    B = 2
    C = 10
    return A * x[:, 0] + B * x[:, 1] + C * x[:, 2]
