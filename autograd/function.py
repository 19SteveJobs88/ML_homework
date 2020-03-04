import numpy as np

from autograd.tensor import Tensor, Dependency


def RMSE(tensor: Tensor, other: Tensor) -> Tensor:
    x = tensor - other
    x = x * x
    x = sqrt(x.sum() * (1. / x.shape[0]))
    return x


def MSE(tensor: Tensor, other: Tensor) -> Tensor:
    x = tensor - other
    x = x * x
    x = x.sum() * (1. / x.shape[0])
    return x


def sqrt(tensor: Tensor) -> Tensor:
    data = np.sqrt(tensor.data)
    requires_grad = tensor.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (0.5 * (1 / (data + 1e-18)))

        last_op = [Dependency(tensor, grad_fn)]
    else:
        last_op = []
    return Tensor(data, requires_grad, last_op)


def tanh(tensor: Tensor) -> Tensor:
    data = np.tanh(tensor.data)
    requires_grad = tensor.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data * data)

        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)
