from autograd.tensor import Tensor, Dependency
from typing import *
import numpy as np
import copy


class PersistentArray:
    _cnt: int
    nodes: List
    _root: List

    class _Node:
        l: int
        r: int
        val: Any

        def __init__(self, val, l: int = -1, r: int = -1):
            self.l = l
            self.r = r
            self.val = val

    def _clone(self, idx: int):
        self.nodes.append(copy.deepcopy(self.nodes[idx]))
        return len(self.nodes) - 1

    def __init__(self, nsize: int, values: List):
        assert nsize == len(values)
        self.nodes = []
        self._cnt = 0
        self.values = values
        self.nsize = nsize
        self.last_version = 0
        self._root = []
        self._root.append(self._build(1, nsize))
        self.cur_version = 0

    def _build(self, l: int, r: int) -> int:
        if l == r:
            self.nodes.append(self._Node(self.values[l - 1]))
            return len(self.nodes) - 1
        mid = (l + r) >> 1
        tmp = self._Node(-1)
        idx = len(self.nodes)
        self.nodes.append(tmp)
        tmp.l = self._build(l, mid)
        tmp.r = self._build(mid + 1, r)
        return idx

    def _query(self, k: int, l: int, r: int, pos: int) -> Any:
        if l == r:
            return self.nodes[k].val
        mid = (l + r) >> 1
        if pos <= mid:
            return self._query(self.nodes[k].l, l, mid, pos)
        return self._query(self.nodes[k].r, mid + 1, r, pos)

    def _insert(self, x: int, l: int, r: int, pos: int, val: int) -> int:
        node = self._clone(x)
        if l == r:
            self.nodes[node].val = val
        else:
            mid = (l + r) >> 1
            if pos <= mid:
                self.nodes[node].l = self._insert(self.nodes[node].l, l, mid, pos, val)
            else:
                self.nodes[node].r = self._insert(self.nodes[node].r, mid + 1, r, pos, val)
        return node

    @staticmethod
    def create(nsize: int, values: List) -> "PersistentArray":
        '''
        :param nsize: the length of list
        :param values: the first version of the array
        :return: corresponding persistent array
        '''
        return PersistentArray(nsize, values)

    def lookup(self, idx: int, version: int) -> Any:
        '''
        :param idx:
        :param version:
        :return: the value of array[idx] in that version
        '''
        return self._query(self._root[version], 1, self.nsize, idx)

    def append(self, idx):
        self._root.append(self._root[idx])

    def update(self, v, i, x):
        '''
        make the value of array[i] to x which version is v, and the new version is (last verstion +1)
        :param v: version
        :param i: index
        :param x: new value
        :return: None
        '''
        self._root.append(self._insert(self._root[v], 1, self.nsize, i, x))

    def __len__(self):
        return len(self._root)


class PersistentUnionSet:
    def __init__(self, n: int):
        self.fa = PersistentArray(n, [i + 1 for i in range(n)])

    def find(self, x: int, version: int = -1) -> int:
        '''
        输入版本和对象，返回它的类别
        :param x: object
        :param version: version
        :return: which cluster
        '''
        if self.fa.lookup(x, version) != x:
            return self.find(self.fa.lookup(x, version), version)
        return x

    def union(self, x, y):
        '''
        合并某两个对象，默认最新版本
        :param x: first object
        :param y: second object
        :return: 1 if success else 0
        '''
        fax = self.find(x, len(self.fa) - 1)
        fay = self.find(y, len(self.fa) - 1)
        if fax != fay:
            fam = min(fax, fay)
            if fam == fax:
                self.fa.update(len(self.fa) - 1, fay, fam)
            else:
                self.fa.update(len(self.fa) - 1, fax, fam)
            return 1
        return 0



def normal(x: Union[Tensor, float], mean: Union[Tensor, float], var: Union[Tensor, float], inv: np.matrix = None,
           det: Union[np.matrix, float] = None) -> float:
    # TODO: 格式检查
    x = np.mat(x)
    n = x.shape[-1]
    mean = np.mat(mean)
    var = np.mat(var)
    # print((x.T-mean).shape)
    # assert 0
    if inv is None:
        inv = np.linalg.inv(var)
    if det is None:
        det = np.linalg.det(var)
    return 1 / (np.float_power(2 * np.pi, n / 2) * (det ** 0.5)) * (
        np.exp(-0.5 * (x.T - mean).T @ inv @ (x.T - mean)))[0, 0]


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    #
    # x = [normal(xx, 0.0, 1.0) for xx in np.arange(-10, 10, 0.01)]
    # print(x)
    # # print(np.arange(-3,3,0.01))
    # plt.plot(np.arange(-10, 10, 0.01), x)
    # # print(normal(0.0, 0.0, 1.0))
    # plt.show()
    # n, m = 5, 6
    # a = [59, 46, 14, 87, 41]
    n, m = input().split()
    n = int(n)
    m = int(m)
    # a = input().split()
    # a = [int(x) for x in a]
    # S = PersistentArray(n, a)
    S = PersistentUnionSet(n)
    # print(S.fa.cur_version)
    # print([S.find(j+1, 0) for j in range(n)])
    opss = [
        '1 1 2',
        '3 1 2',
        '2 0',
        '3 1 2',
        '2 1',
        '3 1 2',
    ]
    # pass 3919
    for i in range(1, m + 1):
        # ops = opss[i - 1].split()
        # print('now is ',i)
        # print([S.find(j+1,i-1) for j in range(n)])
        ops = input().split()
        if ops[0] == '1':
            a = S.union(int(ops[1]), int(ops[2]))
            if a == 0:
                S.fa.append(i - 1)
            S.set_version(i)
        elif ops[0] == '2':
            S.set_version(int(ops[1]))
            S.fa.append(int(ops[1]))
        else:
            aa = S.find(int(ops[1]))
            bb = S.find(int(ops[2]))
            # print(aa,bb)
            if S.find(int(ops[1])) == S.find(int(ops[2])):
                print(1)
            else:
                print(0)
            S.fa.append(S.fa.cur_version)
            S.set_version(i)
