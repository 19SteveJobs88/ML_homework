import numpy as np
from autograd.ml import BaseML
from autograd.utils import PersistentUnionSet
from typing import *


class SubCluster:
    simple: np.ndarray
    double: float
    count: int
    num: int

    def __init__(self, data: np.ndarray = None):
        if data is not None:
            self.simple = data
            self.double = (data * data).sum()
            self.count = 1
        else:
            self.simple = np.zeros(1)
            self.double = 0.0
            self.count = 0
        self.num = 0

    def __add__(self, other):
        new_one = SubCluster()
        new_one.simple = self.simple + other.simple
        new_one.double = self.double + other.double
        new_one.count = self.count + other.count
        new_one.num = self.num
        return new_one

    @staticmethod
    def dis(a: "SubCluster", b: "SubCluster"):
        return (b.count * a.double + a.count * b.double - 2 * (a.simple * b.simple).sum()) / (a.count * b.count)


class AgglomerativeCluster(BaseML):
    Set: Optional[PersistentUnionSet]
    allDist: Optional[np.ndarray]
    dataCnt: int
    all_data: Optional[np.ndarray]

    def __init__(self):
        super(AgglomerativeCluster, self).__init__()
        self.Set = None
        self.dataCnt = 0

    def fit(self, datas, method='min'):
        self.dataCnt = datas.shape[0]
        self.all_data = datas
        self.Set = PersistentUnionSet(self.dataCnt)
        method = self.__getattribute__(method + '_method')
        method = method()
        k = 0
        print(self.dataCnt)
        for _ in range(self.dataCnt - 1):
            a, b = next(method)
            self.Set.union(a, b)
            k += 1
            if k % 100 == 0:
                print(k / self.dataCnt)
        print("cluster finish !")
        pass

    def min_method(self):
        allDist = np.zeros((self.dataCnt, self.dataCnt))
        for i in range(self.dataCnt):
            for j in range(i):
                allDist[i][j] = allDist[j][i] = np.sum((self.all_data[i] - self.all_data[j]) ** 2)
        self.allDist = allDist
        L = []
        for i in range(self.dataCnt):
            for j in range(i):
                L.append((self.allDist[i][j], i + 1, j + 1))
        L.sort(reverse=True)
        fa = [i for i in range(self.dataCnt + 1)]

        def find(x):
            if fa[x] == x:
                return x
            fa[x] = find(fa[x])
            return fa[x]

        def fast_uion(a, b):
            aa = find(a)
            bb = find(b)
            cc = min(aa, bb)
            fa[aa] = fa[bb] = cc
            # print(L)

        while len(L) > 0:
            _, a, b = L[-1]
            L.pop()
            if find(a) != find(b):
                fast_uion(a, b)
                yield a, b
        print('error')
        assert 0

    def max_method(self):
        allDist = np.zeros((self.dataCnt, self.dataCnt))
        for i in range(self.dataCnt):
            for j in range(i):
                allDist[i][j] = allDist[j][i] = np.sum((self.all_data[i] - self.all_data[j]) ** 2)
        self.allDist = allDist
        clusterDist = np.zeros((self.dataCnt, self.dataCnt)) + 999999999.0
        for i in range(self.dataCnt):
            for j in range(i + 1, self.dataCnt):
                clusterDist[i][j] = clusterDist[j][i] = self.allDist[i][j]
        setList, clusterCount = [[i] for i in range(self.dataCnt)], self.dataCnt
        for _ in range(self.dataCnt - 1):
            res = np.argmin(clusterDist)
            dest, src = int(res / clusterCount), int(res % clusterCount)
            yield setList[dest][0] + 1, setList[src][0] + 1
            modify = np.max(clusterDist[[dest, src]], axis=0)
            clusterDist[dest] = modify
            clusterDist[:, dest] = modify
            clusterDist = np.delete(clusterDist, src, axis=0)
            clusterDist = np.delete(clusterDist, src, axis=1)
            clusterDist[dest][dest] = 999999999.0
            setList[dest] = setList[dest] + setList[src]
            del setList[src]
            clusterCount -= 1

    def ave_method(self):
        C = []
        for i, s in enumerate(self.all_data):
            C.append(SubCluster(s))
            C[-1].num = i + 1
        clusterDist = np.zeros((self.dataCnt, self.dataCnt)) + 999999999.0
        for i in range(self.dataCnt):
            for j in range(i + 1, self.dataCnt):
                clusterDist[i][j] = clusterDist[j][i] = SubCluster.dis(C[i], C[j])
        clusterCount = self.dataCnt
        for _ in range(self.dataCnt - 1):
            res = np.argmin(clusterDist)
            dest, src = int(res / clusterCount), int(res % clusterCount)
            yield C[dest].num, C[src].num
            p = np.zeros_like(clusterDist[dest])
            newC = C[dest] + C[src]
            for j in range(clusterCount):
                if j == dest or j == src:
                    p[j] = 999999999.0
                else:
                    p[j] = SubCluster.dis(newC, C[j])
            # print(p)
            clusterDist[dest] = p
            clusterDist[:, dest] = p
            clusterDist = np.delete(clusterDist, src, axis=0)
            clusterDist = np.delete(clusterDist, src, axis=1)
            clusterDist[dest][dest] = 999999999.0
            C[dest] = newC
            del C[src]
            clusterCount -= 1
        pass

    def label(self, k: int):
        l = [self.Set.find(i, self.dataCnt - k) for i in range(1, self.dataCnt + 1)]
        s = set(l)
        d = {}
        for i, j in enumerate(s):
            d[j] = i
        ll = [d[k] for k in l]
        return ll

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


if __name__ == '__main__':
    data = np.random.normal(0, 1, [2000, 2])
    # data = np.ones([3,2])
    # a = SubCluster(data[0])
    # a = a+a
    # print(a.simple, a.double, a.count)
    # assert 0
    # data =[
    #     [0,0.1],
    #     [0,0.2],
    #     [1,0.1],
    #     [1,0.2]
    # ]
    # print(data)
    import matplotlib.pyplot as plt

    data = np.array(data)
    P = AgglomerativeCluster()
    P.fit(data, 'ave')
    K = 4
    s = P.label(K)
    color = 'rbg'
    colors = 'rgbyckm'  # 每个簇的样本标记不同的颜色
    markers = 'o^sP*DX'
    for i in range(len(s)):
        predict = s[i]
        plt.scatter(data[i, 0], data[i, 1],
                    color=colors[predict % len(colors)], alpha=0.5)

    plt.show()
