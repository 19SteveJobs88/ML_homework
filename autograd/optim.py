from autograd.layer import Module
from autograd.tensor import Tensor


class SGD:
    def __init__(self, lr: float = 0.01, lambda_1: float = 0, lambda_2: float = 0) -> None:
        self.lr = lr
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def step(self, model: Module) -> None:
        for parameter in model.parameters():
            loss = (parameter * parameter).sum()
            loss = self.lambda_2 * loss
            loss.backward()
        # loss = self.lambda_2 * loss
        # loss.backward()
        for parameter in model.parameters():
            parameter -= parameter.grad * self.lr
