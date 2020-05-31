# -*- coding: UTF-8 -*-
import numpy as np
from autograd.dataset import DataLoader
from autograd.utils import normal
from typing import *
from autograd.model.ml import BaseML
import matplotlib.pyplot as plt


def read_data(path):
    with open(path, 'r') as f:
        labels = []
        features = []
        for line in f.readlines():
            tmp = line.split(',')
            labels.append(int(tmp[0]))
            features.append(list(map(float, tmp[1:])))
    return np.array(features), np.array(labels)


class Normal:
    def __init__(self, data=None, mean=None, var=None):
        if mean is None and var is None:
            n = len(data[0])
            mean = np.zeros([n, 1])
            var = np.zeros([n, n])
            for _i in range(n):
                # 假设变量都是独立的，所以协方差矩阵为对角阵
                mean[_i, 0] = np.mean(data[:, _i])
                var[_i, _i] = np.var(data[:, _i])
        self.mean = mean
        self.var = var
        self.mean_mat = np.mat(mean)
        self.var_mat = np.mat(var)
        # 预处理计算协方差矩阵的特征值与逆
        self.inv_mat = np.linalg.inv(self.var_mat)
        self.var_det = np.linalg.det(self.var_mat)

    def prob(self, x):
        # 调用计算高斯分布的函数获得概率值
        return normal(x, self.mean, self.var, self.inv_mat, self.var_det)


class NaiveBayes(BaseML):
    items: list
    prior: list

    def __init__(self):
        self.items = []
        self.prior = []

    def fit(self, train_data: np.ndarray, train_label: np.ndarray, classes: int = 3,
            disstribution: str = 'normal'):
        self.items = []
        if disstribution == 'normal':
            Dis = Normal
        else:
            Dis = Normal
        train_data_label = [np.array([train_data[j] for j, x in enumerate(train_label) if x == i]) for i in
                            range(1, classes + 1)]  # 将数据根据标签划分为三类
        self.items = [Dis(one) for one in train_data_label]  # 对这三类分别构建高斯分布的模型
        self.prior = [1 / len(one) for one in train_data_label]  # 分别计算三类的先验概率

    def predict(self, X: np.ndarray) -> Tuple[Union[np.ndarray, list], Union[np.ndarray, list]]:
        # 对于每一个数据计算三类的概率，乘以先验后取给出最大值的类作为标签
        labels = [np.argmax([p.prob(x) * self.prior[i] for i, p in enumerate(self.items)]) + 1 for x in X]
        # 计算它属于每一个类的概率，为后续画图做准备
        prob = [[p.prob(x) * self.prior[i] for x in X] for i, p in enumerate(self.items)]
        return labels, prob


def get_pos(i):
    def is_pos(j):
        if i == j:
            return True
        return False

    return is_pos


def draw(prob: np.ndarray, test_label):
    prob = np.array(prob)
    test_label = np.array(test_label)
    fpr, tpr = [[0], [0], [0]], [[0], [0], [0]]
    auc = [0] * 3
    plt.figure()
    for i in range(3):
        # 计算fpr和tpr，阈值从大到小
        for j in np.argsort(prob[i])[-2:0:-1]:
            threshold = prob[i][j]
            # 根据阈值分类
            tp = [1 if l == i + 1 and p >= threshold else 0 for l, p in zip(test_label, prob[i])]
            fp = [1 if l != i + 1 and p >= threshold else 0 for l, p in zip(test_label, prob[i])]
            fn = [1 if l == i + 1 and p < threshold else 0 for l, p in zip(test_label, prob[i])]
            tn = [1 if l != i + 1 and p < threshold else 0 for l, p in zip(test_label, prob[i])]
            fpr[i].append(sum(fp) / (sum(fp) + sum(tn)))
            tpr[i].append(sum(tp) / (sum(tp) + sum(fn)))
        # AUC的值为曲线与X轴围成的面积
        fpr[i].append(1)
        tpr[i].append(1)
        auc[i] = np.trapz(tpr[i], fpr[i])
        # 绘图
        plt.title('ROC')
        plt.plot(fpr[i], tpr[i], label=str(i + 1) + "  ROC")
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        print(str(i + 1) + '  AUC:', auc[i])
    plt.legend()
    # plt.savefig('ROC.png')
    plt.show()


if __name__ == '__main__':
    np.random.seed(7)
    train_data, label = read_data('wine.data')
    data_loader = DataLoader(train_data, label)
    train_data, test_data, train_label, test_label = data_loader.hierarchy()
    NB = NaiveBayes()
    NB.fit(train_data=train_data, train_label=train_label)
    ans, _ = NB.predict(test_data)
    acc = NB.cal_acc(test_label, ans)
    print(acc)
    for i in range(1, 4):
        tp, fp, fn, tn = NB.cal_mix(test_label, get_pos(i), ans)
        tp_num = sum(tp)
        fp_num = sum(fp)
        fn_num = sum(fn)
        tn_num = sum(tn)
        print([[tp_num, fp_num], [fn_num, tn_num]])
        accuracy = (tp_num + tn_num) / (tp_num + tn_num + fp_num + fn_num)
        precision = tp_num / (tp_num + fp_num)
        recall = tp_num / (tp_num + fn_num)
        F1 = 2 * precision * recall / (precision + recall)
        print(i, accuracy, precision, recall, F1)
    ans, prob = NB.predict(test_data)
    draw(prob, test_label)
