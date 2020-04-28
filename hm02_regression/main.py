import pandas as pd
import numpy as np
from typing import Iterator, Tuple, List, NewType
from autograd.tensor import Tensor
from autograd.layer import Linear, Module, Parameter
from autograd.optim import SGD
from autograd.function import tanh, RMSE, MSE
import matplotlib.pyplot as plt


class DataLoader:
    train: np.ndarray
    label: np.ndarray

    def __init__(self, path):
        assert 'csv' in path
        _file = pd.read_csv(path)
        _x = _file.iloc[:, :-1]
        _y = _file.iloc[:, -1]
        _x = np.array(_x, dtype=np.float)
        _y = np.array(_y, dtype=np.float)
        min_data = _x.min(0)
        max_data = _x.max(0)
        self.train = (_x - min_data) / (max_data - min_data)
        self.label = _y

    def split(self, alpha: float = 0.90) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :param alpha:
        :return: x_train, x_test, y_train，y_test
        """
        assert 0 <= alpha <= 1
        state = np.random.get_state()
        np.random.shuffle(self.train)
        np.random.set_state(state)
        np.random.shuffle(self.label)
        _len = int(alpha * len(self.train))
        return self.train[:_len], self.train[_len:], self.label[:_len], self.label[_len:]


class FizzBuzzModel(Module):
    def __init__(self, num_hidden: int = 50) -> None:
        self.w1 = Parameter(11, num_hidden)
        self.b1 = Parameter(num_hidden)

        self.w2 = Parameter(num_hidden, 1)
        self.b2 = Parameter(1)

    def forward(self, x: Tensor) -> Tensor:
        # inputs will be (batch_size, 10)
        x1 = x @ self.w1
        x1 = x1 + self.b1  # (batch_size, num_hidden)
        x2 = tanh(x1)  # (batch_size, num_hidden)
        x3 = x2 @ self.w2 + self.b2  # (batch_size, 4)

        return x3


if __name__ == '__main__':
    file = DataLoader("winequality-white.csv")
    x_train, x_test, y_train, y_test = file.split()
    print(len(x_train), len(x_test))

    y_train = Tensor(y_train)
    batch_size = len(x_train)
    starts = np.arange(0, x_train.shape[0], batch_size)
    x_test = Tensor(x_test)
    y_test = Tensor(y_test)

    model = Linear(x_train.shape[1], 1)
    # model = FizzBuzzModel()
    optimizer = SGD(lr=1e-2)
    losses = []
    epoch_rmse = []
    for epoch in range(20):
        # epoch_loss = []
        # ac = model.forward(x_test)
        # b = RMSE(ac, y_test)
        # epoch_rmse.append(b.data)
        # print(RMSE(ac, y_test))
        np.random.shuffle(starts)
        for start in starts:
            end = start + batch_size

            model.zero_grad()

            inputs = Tensor(x_train[start:end])

            predicted = model.forward(inputs)
            # print(predicted)
            actual = y_train[start:end]
            # print(actual, predicted)
            loss = RMSE(predicted, actual)
            loss.backward()
            # print(loss.data)
            # epoch_loss.append(loss.data)
            optimizer.step(model)
            model.zero_grad()
        # losses.extend(epoch_loss)

        model.zero_grad()

    for p in model.parameters():
        print(p)
    model = Linear(x_train.shape[1], 1)
    # model = FizzBuzzModel()
    optimizer = SGD(lr=1e-2,lambda_2=1)
    losses = []
    epoch_rmse = []
    for epoch in range(20):
        epoch_loss = []
        np.random.shuffle(starts)
        for start in starts:
            end = start + batch_size
            model.zero_grad()
            inputs = Tensor(x_train[start:end])
            predicted = model.forward(inputs)
            # print(predicted)
            actual = y_train[start:end]
            loss = RMSE(predicted, actual)
            loss.backward()
            optimizer.step(model)
            model.zero_grad()
        model.zero_grad()
    for p in model.parameters():
        print(p)
