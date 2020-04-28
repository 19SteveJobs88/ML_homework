from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors.kde import KernelDensity as kde
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


def loadData(path: str):
    '''
    读数据
    '''
    data = pd.read_csv(path)
    data = np.array(data)
    m, n = data.shape
    X = data[:, 0:n - 1]
    Y = data[:, -1]
    X = X[:,0].reshape(-1,1)
    # print(X)
    std = StandardScaler()
    X = std.fit_transform(X)
    return X, Y


class Gaussian:
    def __init__(self, x=None):
        self.mu = None
        self.cov = None
        if x is not None:
            self.fit(x)

    def fit(self, x):
        self.mu = np.average(x, axis=0)  # 计算均值
        self.cov = np.cov(x.T)  # 计算协方差

    def predict(self, x):
        shape = x.shape[0]
        sub = (x - self.mu).reshape(shape, 1)
        p = 1 / (math.pow(2 * math.pi, shape / 2) * math.sqrt(np.linalg.det(self.cov)))
        p = p * math.exp(-1 / 2 * np.matmul(np.matmul(sub.T, np.linalg.inv(self.cov)), sub))  # 计算公式
        return p


def params_estimate(x, y):
    kfold = RepeatedKFold(n_splits=10, n_repeats=10)  # 10次十折交叉验证
    corr = []
    for train_index, test_index in kfold.split(x):
        x_train, x_test = x[train_index], x[test_index]  # 特征矩阵的训练集与测试集
        y_train, y_test = y[train_index], y[test_index]  # 标签的数据集
        Gaussians = [Gaussian(x_train[np.where(y_train == i)]) for i in (1, 2, 3)]  # 生成三类高斯分布的模型
        count = [yy - 1 == np.argmax([i.predict(xx) for i in Gaussians]) for xx, yy in zip(x_test, y_test)]  # 计算最有可能的标签
        corr.append(np.sum(count) / y_test.shape[0])
    return np.average(corr)


def unparams_estimate(x, y):
    band_width = np.arange(0.5, 0.75, 0.01)
    acc = []
    for width in band_width:
        kfold = RepeatedKFold(n_splits=10, n_repeats=10)  # 10次十折交叉验证
        corr = []
        for train_index, test_index in kfold.split(x):
            x_train, x_test = x[train_index], x[test_index]  # 特征矩阵的训练集与测试集
            y_train, y_test = y[train_index], y[test_index]  # 标签的数据集
            Models = [kde(kernel="gaussian", bandwidth=width).fit(x_train[np.where(y_train == i)]) for i in
                      (1, 2, 3)]  # 生成三类的模型
            count = [yy - 1 == np.argmax([model.score_samples(xx.reshape(1, -1)) for model in Models]) for xx, yy in
                     zip(x_test, y_test)]  # 计算最有可能的标签
            corr.append(np.sum(count) / y_test.shape[0])
        acc.append(np.average(corr))
    print(band_width)
    plt.plot(band_width, acc)
    plt.show()


def unparams_estimate_knn(x, y):
    band_width = range(3, 20)
    acc = []
    for k in band_width:
        kfold = RepeatedKFold(n_splits=10, n_repeats=10)  # 10次十折交叉验证
        corr = []
        for train_index, test_index in kfold.split(x):
            x_train, x_test = x[train_index], x[test_index]  # 特征矩阵的训练集与测试集
            y_train, y_test = y[train_index], y[test_index]  # 标签的数据集
            Model = KNeighborsClassifier(n_neighbors=k)
            Model.fit(x_train, y_train)
            corr.append(Model.score(x_test, y_test))
        acc.append(np.average(corr))
    print(band_width)
    plt.plot(band_width, acc)
    plt.show()


def pltknn(x, k):
    # 对于特征x1进行画图
    N = len(x)
    num = 1000
    # mm = x[:,0].min()
    # mx = x[:,0].max()
    # x[:,0] = (x[:,0]- mm)/(mx-mm)
    xx = np.linspace(-2, 2, num)  # 得到间隔相等的一组数
    p = []  # 概率
    for i in range(num):
        dist = []
        for j in x:
            dist.append(np.abs(xx[i] - j[0])+1e-6)  # 计算距离
        dist.sort()  # 由小到大排序
        h = dist[k - 1]  # 取第k近的
        p.append(k / (N * h * 2))  # 计算概率密度
        # 2 是一维单位超球的体积
    # 画图
    plt.figure()
    plt.plot(xx, p)
    plt.title("k=" + str(k))
    # plt.savefig("k为%d.png" % k, dpi=1200)
    plt.show()


if __name__ == '__main__':
    x, y = loadData(r'HWData3.csv')
    # a = params_estimate(x, y)
    # unparams_estimate_knn(x, y)
    # pass
    # print(a)
    # print(x)
    for k in (1, 3, 5):
        pltknn(x[np.where(y == 1)], k)
