import numpy as np
from typing import *
from autograd.model.ml import BaseML
import copy


class _TreeNode:
    def __init__(self, attr: int, label: float, v: int, deep: int = 0):
        self.is_continue = False
        self.split_attr = attr
        self.label = label
        self.attr_value = v
        self.deep = deep
        self.is_leaf = False
        self.children = []


def _cal_ent_d(Data):
    s = 0
    for k in set(Data):
        p_k = np.sum(np.where(Data == k, 1, 0)) / np.shape(Data)[0]
        if p_k == 0:
            continue
        s += p_k * np.log2(p_k)
    return -s


class ID3(BaseML):
    def __init__(self, max_deep: int = 10, min_simple: int = 1):
        super(ID3, self).__init__()
        self.root = None
        self.cal = self.cal_advantage
        self.max_deep = max_deep
        self.min_simple = min_simple

    def cal_advantage(self, X, Y, attr):
        x_attr_col = X[:, attr]
        ent_Dv = []
        weight_Dv = []
        for x_v in set(x_attr_col):
            index_x_equal_v = np.where(x_attr_col == x_v)
            y_x_equal_v = Y[index_x_equal_v]
            ent_Dv.append(_cal_ent_d(y_x_equal_v))  # 计算熵
            weight_Dv.append(np.shape(y_x_equal_v)[0] / np.shape(Y)[0])  # 计算条件概率
        return _cal_ent_d(Y) - np.sum(np.array(ent_Dv) * np.array(weight_Dv))  # 计算收益

    def is_same(self, X: np.ndarray, attrs: np.ndarray):
        for x in X[:, attrs]:
            if np.array(x != X[0, attrs]).any():
                return False
        return True

    def fit(self, data, label):
        self.root = _TreeNode(-1, -1, -1, 0)
        attrs = [i for i in range(len(data[0]))]
        self._fit(self.root, data, label, attrs)

    def fit_prun(self, data, label, test_data, test_label):
        self.root = _TreeNode(-1, -1, -1, 0)
        attrs = [i for i in range(len(data[0]))]
        self._fit_prun(self.root, data, label, test_data, test_label, attrs)

    def _fit_prun(self, node: _TreeNode, data, label, test_data, test_label, attrs):
        if len(set(label)) == 1:
            node.label = label[0]
            node.is_leaf = True
            return None
        if len(attrs) == 0 or self.is_same(data, attrs) or node.deep > self.max_deep or len(data) < self.min_simple:
            node.label = np.argmax(np.bincount(label))
            node.is_leaf = True
            return None
        advantage = np.array([self.cal(data, label, i) for i in attrs])
        idx = np.argmax(advantage)
        attr = attrs[idx]
        node.split_attr = attr
        del attrs[idx]

        T_node = (np.argmax(np.bincount(label)) == test_label).sum()
        T_a = 0
        tmp = data[:, attr].tolist()
        tmp.extend(test_data[:, attr].tolist())
        for ax in set(tmp):
            D_v = np.where(data[:, attr] == ax)
            T_v = np.where(test_data[:, attr] == ax)
            if len(D_v[0]) == 0:
                C_v = np.argmax(np.bincount(label))
            else:
                C_v = np.argmax(np.bincount(label[D_v]))
            T_a += (C_v == T_v).sum()
        if T_node > T_a:
            node.is_leaf = True
            node.label = np.argmax(np.bincount(label))
        else:
            for x_v in set(data[:, attr]):
                n = _TreeNode(-1, -1, x_v, node.deep + 1)
                node.children.append(n)
                self._fit(n, data[np.where(x_v == data[:, attr])], label[np.where(x_v == data[:, attr])],
                          copy.deepcopy(attrs))
            return None

        pass

    def _fit(self, node: _TreeNode, data, label, attrs):
        if len(set(label)) == 1:
            # 如果所有样本都是相同的标签
            node.label = label[0]
            node.is_leaf = True
            return None
        if len(attrs) == 0 or self.is_same(data, attrs) or node.deep > self.max_deep or len(data) < self.min_simple:
            # 如果所有的标签相同，或者深度达到最大深度或者样本数小于最少样本
            # 标签为样本中最多的标签
            node.label = np.argmax(np.bincount(label))
            node.is_leaf = True
            return None
        advantage = np.array([self.cal(data, label, i) for i in attrs])  # 计算信息熵
        idx = np.argmax(advantage)  # 选择最大的
        attr = attrs[idx]  # 决定属性
        node.split_attr = attr
        del attrs[idx]
        # 离散数据处理
        for x_v in set(data[:, attr]):
            n = _TreeNode(-1, -1, x_v, node.deep + 1)
            node.children.append(n)
            self._fit(n, data[np.where(x_v == data[:, attr])], label[np.where(x_v == data[:, attr])],
                      copy.deepcopy(attrs))
        return None

    def _predict(self, node, x):
        if node.is_leaf:
            # 如果是叶子节点，返回标签
            return node.label
        else:
            for child in node.children:
                # 根据所选择的特征，递归处理
                if child.attr_value == x[node.split_attr]:
                    return self._predict(child, x)
        return None

    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        label = np.array([self._predict(self.root, i) for i in data])
        return label, None

    def cal_acc(self, label: np.ndarray, Y: Optional[np.ndarray] = None, X: Optional[np.ndarray] = None) -> float:
        return super(ID3, self).cal_acc(label, Y, X)


class C45(BaseML):
    class _Node:
        def __init__(self, attr: int, label: float, v: int, deep: int = 0):
            self.is_continue = False
            self.split_attr = attr
            self.label = label
            self.attr_value = v
            self.deep = deep
            self.is_leaf = False
            self.children = []

    def __init__(self, max_deep: int = 10, min_simple: int = 1):
        super(C45, self).__init__()
        self.root = None
        self.cal = self.cal_advantage
        self.max_deep = max_deep
        self.min_simple = min_simple

    def cal_advantage(self, X, Y, attr_d, attr_c):
        ad_d = None
        ad_c = None
        ts = None
        if attr_d is not None:
            # 如果是计算离散的标签
            info_D = _cal_ent_d(Y)
            x_attr_d = X[:, attr_d]
            ent_Dv = []
            weight_Dv = []
            splite_info = []
            for x_v in set(x_attr_d):
                index_x_equal_v = np.where(x_attr_d == x_v)
                y_x_equal_v = Y[index_x_equal_v]
                ent_Dv.append(_cal_ent_d(y_x_equal_v))
                v = np.shape(y_x_equal_v)[0] / len(Y)
                weight_Dv.append(v)
                splite_info.append(-v * np.log2(v))  # 固有信息
            ad_d = (info_D - np.sum(np.array(ent_Dv) * np.array(weight_Dv))) / (np.sum(np.array(splite_info))+1e-6)  # 信息增益率
        elif attr_c is not None:
            # 如果是计算连续的标签
            x_attr_c = X[:, attr_c]
            splite_info = []
            p = np.sort(x_attr_c)  # 对连续的取值进行排序
            ent_cv = []
            ts = []
            if len(p) == 1:
                ad_c = 0
                ts = p[0]
            else:
                for i in range(1, len(p)):
                    t = (p[i] + p[i - 1]) / 2
                    ts.append(t)
                    idx_x_leq_v = np.where(x_attr_c <= t)
                    idx_x_geq_v = np.where(x_attr_c > t)
                    ent_cv.append((i / len(p)) *
                                  _cal_ent_d(Y[idx_x_leq_v]) + (1 - (i / len(p))) *
                                  _cal_ent_d(Y[idx_x_geq_v]))  # 每一次划分的信息熵
                    splite_info.append(
                        -(i / len(p)) * np.log2(i / len(p)) - (1 - (i / len(p))) * np.log2((1 - i / len(p))))  # 固有信息
                idx = np.argmin(np.array(ent_cv))  # 选择最小的信息熵
                ad_c = (_cal_ent_d(Y) - ent_cv[idx]) / splite_info[idx]  # 计算信息增益率
                ts = (p[idx + 1] + p[idx]) / 2  # 确定所选划分的值
        else:
            assert 0

        return ad_d, ad_c, ts

    def is_same(self, X: np.ndarray, attrs: List[int]):
        if len(attrs) == 0:
            return True
        for x in X[:, attrs]:
            if np.array(x != X[0, attrs]).any():
                return False
        return True

    def fit(self, data, label, attrs_d, attrs_c):
        self.root = _TreeNode(-1, -1, -1, 0)
        self._fit(self.root, data, label, attrs_d, attrs_c)

    def _fit(self, node: _TreeNode, data, label, attrs_d, attrs_c):
        if len(set(label)) == 1:
            # 如果只有一类标签
            node.label = label[0]
            node.is_leaf = True
            return None
        if len(attrs_d) + len(attrs_c) == 0 or node.deep > self.max_deep or len(data) < self.min_simple:
            # 如果没有可选择的特征获得超过最大深度或者小于最少样本
            node.label = np.argmax(np.bincount(label))
            node.is_leaf = True
            return None
        if self.is_same(data, attrs_c) and self.is_same(data, attrs_d):
            # 如果所对应的特征都是相同的
            node.label = np.argmax(np.bincount(label))
            node.is_leaf = True
            return None

        max_d = -9999
        max_c = -9999
        max_idx_d = 0
        max_idx_c = 0
        max_idx_t = 0
        if len(attrs_d) > 0:
            ad = []
            for i in attrs_d:
                v, _, _ = self.cal_advantage(data, label, i, None)
                ad.append(v)
            ad = np.array(ad)
            # 选择离散变量最好的划分
            max_d = np.max(ad)
            max_idx_d = np.argmax(ad)
        if len(attrs_c) > 0:
            ad = []
            ts = []
            for i in attrs_c:
                _, v, t = self.cal_advantage(data, label, None, i)
                ad.append(v)
                ts.append(t)
            # 选择连续变量最好的划分
            max_c = np.max(np.array(ad))
            max_idx_c = np.argmax(np.array(ad))
            max_idx_t = ts[max_idx_c]
        if max_d > max_c:
            # 离散
            attr = attrs_d[max_idx_d]
            node.split_attr = attr
            del attrs_d[max_idx_d]

            for x_v in set(data[:, attr]):
                n = _TreeNode(-1, -1, x_v, node.deep + 1)
                node.children.append(n)
                self._fit(n, data[np.where(x_v == data[:, attr])], label[np.where(x_v == data[:, attr])],
                          copy.deepcopy(attrs_d), copy.deepcopy(attrs_c))
        else:
            # 连续
            attr = attrs_c[max_idx_c]
            node.split_attr = attr
            # 这里可以选择连续的状态使用一次或者使用多次
            # del attrs_c[max_idx_c]
            nl = _TreeNode(-1, -1, max_idx_t, node.deep + 1)
            # 设置为连续的决策变量
            nl.is_continue = True
            node.children.append(nl)
            # 小于的递归左子树
            self._fit(nl, data[np.where(data[:, attr] <= max_idx_t)], label[np.where(data[:, attr] <= max_idx_t)],
                      copy.deepcopy(attrs_d), copy.deepcopy(attrs_c))
            nr = _TreeNode(-1, -1, max_idx_t, node.deep + 1)
            nr.is_continue = True
            node.children.append(nr)
            # 大于的递归右子树
            self._fit(nr, data[np.where(data[:, attr] > max_idx_t)], label[np.where(data[:, attr] > max_idx_t)],
                      copy.deepcopy(attrs_d), copy.deepcopy(attrs_c))
        return None

    def _predict(self, node, x):
        if node.is_leaf:
            return node.label
        else:
            if node.children[0].is_continue:
                # 对于连续变量
                if x[node.split_attr] <= node.children[0].attr_value:
                    return self._predict(node.children[0], x)  # 左子树
                else:
                    return self._predict(node.children[1], x)  # 右子树
            else:
                for child in node.children:
                    if child.attr_value == x[node.split_attr]:
                        return self._predict(child, x)
        return None

    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        label = np.array([self._predict(self.root, i) for i in data])
        return label, None

    def cal_acc(self, label: np.ndarray, Y: Optional[np.ndarray] = None, X: Optional[np.ndarray] = None) -> float:
        return super(C45, self).cal_acc(label, Y, X)
