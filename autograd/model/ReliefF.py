import numpy as np
from autograd.model.ml import BasePreprocess
from typing import *
from sklearn.neighbors import KDTree
from scipy.spatial.distance import euclidean


class ReliefF(BasePreprocess):

    def __init__(self):
        self.feature_scores = None
        self.top_features = None

    def _find_nm(self, sample, X):
        s = [euclidean(sample, s) for s in X]
        return np.argmin(np.array(s))

    def fit(self, X, y):
        self.feature_scores = np.zeros(X.shape[1], dtype=np.float64)

        # 计算先验
        labels, counts = np.unique(y, return_counts=True)
        Prob = counts / float(len(y))
        for label in labels:
            select = (y == label)
            tree = KDTree(X[select, :])
            nh = tree.query(X[select, :], k=2, return_distance=False)[:, 1:]
            nh = (nh.T[0]).tolist()

            # 计算 -diff(x, x_nh)
            nh_mat = np.square(np.subtract(X[select, :], X[select, :][nh, :])) * -1

            # 找到other里面最邻近的
            nm_mat = np.zeros_like(X[select, :])
            for prob, other_label in zip(Prob[labels != label], labels[labels != label]):
                other_select = (y == other_label)
                nm = [np.argmin(np.array([euclidean(sample, s) for s in X[other_select, :]])) for sample in
                      X[select, :]]
                # 计算 -diff(x, x_nm)
                nm_tmp = np.square(np.subtract(X[select, :], X[other_select, :][nm, :])) * prob
                nm_mat = np.add(nm_mat, nm_tmp)

            mat = np.add(nh_mat, nm_mat)
            self.feature_scores += np.sum(mat, axis=0)

        # 计算顺序
        self.top_features = np.argsort(self.feature_scores)[::-1]
        self.feature_scores = self.feature_scores[self.top_features]

        return self.top_features, self.feature_scores

    def transform(self, X: np.ndarray, k: int):
        return X[:, self.top_features[:k]]

    def fit_transform(self, X: np.ndarray, y: np.ndarray, k: int):
        self.fit(X, y)
        return self.transform(X, k)
