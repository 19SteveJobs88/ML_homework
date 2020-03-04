from autograd.tensor import Tensor, Tensorable, ensure_tensor
import numpy as np
from typing import Iterator
import inspect


class Parameter(Tensor):
    def __init__(self, *shape) -> None:
        data = np.random.randn(*shape)
        super().__init__(data, requires_grad=True)


class Module:
    def parameters(self) -> Iterator[Parameter]:
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()


class Linear(Module):
    def __init__(self, n_in, n_out, bias=True):
        self.W = Parameter(n_in, n_out)
        self.bias = bias
        if bias:
            self.b = Parameter(n_out)

    def forward(self, x: Tensorable):
        assert self.W.shape[0] == x.shape[-1]
        x = ensure_tensor(x)
        if self.bias:
            ans = x @ self.W + self.b
        else:
            ans = x @ self.W
        return ans
