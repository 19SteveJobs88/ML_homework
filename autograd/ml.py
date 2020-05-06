import numpy as np
import abc
from typing import *


class BaseML(abc.ABC):
    @abc.abstractmethod
    def fit(self, *args):
        pass

    @abc.abstractmethod
    def predict(self, *args) -> np.ndarray:
        pass

    def cal_acc(self, label: np.ndarray, Y: Optional[np.ndarray] = None, X: Optional[np.ndarray] = None) -> float:
        '''
        计算准确率
        :param label: 标签
        :param Y: 预测值
        :param X: 数据
        :return: 准确率
        '''
        if Y is None:
            if X is None:
                assert 0
            Y, _ = self.predict(X)
        acc = [i == j for i, j in zip(Y, label)]  # 判断是否和标签一致
        return sum(acc) / len(acc)

    def cal_mix(self, label: np.ndarray, is_pos, Y: Optional[np.ndarray] = None, X: Optional[np.ndarray] = None):
        '''
        :param label: 标签
        :param is_pos: 判断一个标签是否是正例的函数（用来将多分类变成二分类）
        :param Y: 预测的标签
        :param X: 需要预测的数据
        :return: tp,fp,fn,tn 对于每一个数据的混淆矩阵
        '''
        if Y is None:
            if X is None:
                assert 0
            Y, _ = self.predict(X)
        tp = []
        fp = []
        fn = []
        tn = []
        for y, r in zip(Y, label):
            if is_pos(r):
                if y == r:
                    tp.append(1)
                    fp.append(0)
                    fn.append(0)
                    tn.append(0)
                else:
                    tp.append(0)
                    fp.append(0)
                    fn.append(1)
                    tn.append(0)
            else:
                if is_pos(y):
                    tp.append(0)
                    fp.append(1)
                    fn.append(0)
                    tn.append(0)
                else:
                    tp.append(0)
                    fp.append(0)
                    fn.append(0)
                    tn.append(1)

        return tp, fp, fn, tn
