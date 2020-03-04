import numpy as np
from typing import *


class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]


Arrayable = Union[float, list, np.ndarray]
Tensorable = Union['Tensor', float, np.ndarray]


def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


class Tensor:
    def __init__(self, data: Arrayable, requires_grad: bool = False, last_op: List[Dependency] = None) -> None:
        self._data = ensure_array(data)
        self.requires_grad = requires_grad
        self.last_op = last_op or []
        # self.gradW = np.zeros(shape=shape)
        self.shape = self.data.shape
        self.grad: Optional['Tensor'] = None
        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = new_data
        # Setting the data manually means we invalidate the gradient.
        self.grad = None

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

    def __getitem__(self, idxs) -> 'Tensor':
        return _slice(self, idxs)

    def __repr__(self) -> str:
        return "Tensor({}, requires_grad:{})".format(self.data, self.requires_grad)

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad
        if grad is None:
            if len(self.shape) == 0:
                grad = Tensor(1.0)
            else:
                raise RuntimeError('no grad for none-0-tensor')
        self.grad.data = self.grad.data + grad.data
        for op in self.last_op:
            backward_grad = op.grad_fn(grad.data)
            op.tensor.backward(Tensor(backward_grad))

    def __add__(self, other):
        return _add(self, ensure_tensor(other))

    def __radd__(self, other):
        return _add(ensure_tensor(other), self)

    def __mul__(self, other):
        return _mul(self, ensure_tensor(other))

    def __rmul__(self, other) -> 'Tensor':
        return _mul(ensure_tensor(other), self)

    def __rsub__(self, other):
        return _sub(ensure_tensor(other), self)

    def __sub__(self, other):
        return _sub(self, ensure_tensor(other))

    def __neg__(self):
        return _neg(self)

    def sum(self) -> 'Tensor':
        return _tensor_sum(self)

    def __matmul__(self, other):
        return _matmul(self, ensure_tensor(other))

    def __rmatmul__(self, other):
        return _matmul(ensure_tensor(other), self)

    def __iadd__(self, other) -> 'Tensor':
        self.data = self.data + ensure_tensor(other).data
        return self

    def __isub__(self, other) -> 'Tensor':
        self.data = self.data - ensure_tensor(other).data
        return self

    def __imul__(self, other) -> 'Tensor':
        """when we do t *= other"""
        self.data = self.data * ensure_tensor(other).data
        return self


def _tensor_sum(t: Tensor) -> Tensor:
    data = t.data.sum()
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.ones_like(t.data)

        last_op = [Dependency(t, grad_fn)]
    else:
        last_op = []
    return Tensor(data=data, requires_grad=requires_grad, last_op=last_op)


def _add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    last_op: List[Dependency] = []
    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            diff_dim = grad.ndim - t1.data.ndim
            for _ in range(diff_dim):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        last_op.append(Dependency(t1, grad_fn1))
    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            diff_dim = grad.ndim - t2.data.ndim
            for _ in range(diff_dim):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        last_op.append(Dependency(t2, grad_fn2))
    return Tensor(data=data, requires_grad=t1.requires_grad or t2.requires_grad, last_op=last_op)


def _mul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data * t2.data
    last_op: List[Dependency] = []
    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad * t2.data
            diff_dim = grad.ndim - t1.data.ndim
            for _ in range(diff_dim):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        last_op.append(Dependency(t1, grad_fn1))
    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = grad * t1.data
            diff_dim = grad.ndim - t2.data.ndim
            for _ in range(diff_dim):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        last_op.append(Dependency(t2, grad_fn2))
    return Tensor(data=data, requires_grad=t1.requires_grad or t2.requires_grad, last_op=last_op)


def _neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        depends_on = [Dependency(t, lambda x: -x)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 + -t2


def _matmul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data @ t2.data
    last_op: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return grad @ t2.data.T

        last_op.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return t1.data.T @ grad

        last_op.append(Dependency(t2, grad_fn2))

    return Tensor(data, t1.requires_grad or t2.requires_grad, last_op)


def _slice(t: Tensor, idxs) -> Tensor:
    data = t.data[idxs]
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(data)
            bigger_grad[idxs] = grad
            return bigger_grad

        depends_on = Dependency(t, grad_fn)
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)
