from autograd.tensor import Tensor, Dependency
from typing import *
import numpy as np


def normal(x: Union[Tensor, float], mean: Union[Tensor, float], var: Union[Tensor, float], inv: np.matrix = None,
           det: Union[np.matrix, float] = None) -> float:
    # TODO: 格式检查
    x = np.mat(x)
    n = x.shape[-1]
    mean = np.mat(mean)
    var = np.mat(var)
    # print((x.T-mean).shape)
    # assert 0
    if inv is None:
        inv = np.linalg.inv(var)
    if det is None:
        det = np.linalg.det(var)
    return 1 / (np.float_power(2 * np.pi, n / 2) * (det ** 0.5)) * (
        np.exp(-0.5 * (x.T - mean).T @ inv @ (x.T - mean)))[0, 0]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = [normal(xx, 0.0, 1.0) for xx in np.arange(-10, 10, 0.01)]
    print(x)
    # print(np.arange(-3,3,0.01))
    plt.plot(np.arange(-10, 10, 0.01), x)
    # print(normal(0.0, 0.0, 1.0))
    plt.show()
