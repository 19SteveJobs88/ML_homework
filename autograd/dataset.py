import numpy as np
from typing import *
import pandas as pd


class DataLoader:
    train: np.ndarray
    label: np.ndarray

    def __init__(self, train: Union[np.ndarray, list, None], label: Union[np.ndarray, list, None], path: str = None,
                 mod: str = 'set'):
        if train is None and label is None and path is not None:
            # 使用文件的形式读取
            if mod in ['csv', 'excel', 'csv_first', 'excel_first']:
                # 暂只支持csv和excel
                if 'csv' in mod and 'csv' in path:
                    _file = pd.read_csv(path)
                elif 'excel' in mod:
                    _file = pd.read_excel(path)
                else:
                    assert 0
                # 标签位于数据的首部或尾部
                if 'first' in mod:
                    _x = _file.iloc[:, 1:]
                    _y = _file.iloc[:, 0]
                else:
                    _x = _file.iloc[:, :-1]
                    _y = _file.iloc[:, -1]
                train = np.array(_x, dtype=np.float)
                label = np.array(_y, dtype=np.float)
        train = np.array(train)
        label = np.array(label)
        min_data = train.min(0)
        max_data = train.max(0)
        self.train = (train - min_data) / (max_data - min_data)  # 数据归一化
        self.label = label

    def split(self, alpha: float = 0.90) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        按照测试集占总体alpha的比例划分数据集
        :param alpha:
        :return: x_train, x_test, y_train，y_test
        """
        assert 0 <= alpha <= 1
        # 数据集打乱
        state = np.random.get_state()
        np.random.shuffle(self.train)
        np.random.set_state(state)
        np.random.shuffle(self.label)
        _len = int(alpha * len(self.train))
        return self.train[:_len], self.train[_len:], self.label[:_len], self.label[_len:]

    def hierarchy(self, nums: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        按照每nums次抽一条的方式分层抽样
        :param nums:
        :return: x_train, x_test, y_train，y_test
        '''
        # 数据集打乱
        state = np.random.get_state()
        np.random.shuffle(self.train)
        np.random.set_state(state)
        np.random.shuffle(self.label)
        train_data = [x for i, x in enumerate(self.train) if i % nums != 0]
        test_data = [x for i, x in enumerate(self.train) if i % nums == 0]
        train_label = [x for i, x in enumerate(self.label) if i % nums != 0]
        test_label = [x for i, x in enumerate(self.label) if i % nums == 0]
        return np.array(train_data), np.array(test_data), np.array(train_label), np.array(test_label)
