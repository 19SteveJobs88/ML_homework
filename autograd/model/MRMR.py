import numpy as np
from autograd.model.ml  import BasePreprocess

class MRMR(BasePreprocess):
    def __init__(self):
        self._selected_features = []

    def fit(self, X, y):
        """
        fit an array data
        :param X: a numpy array
        :param y: the label, a list or one dimension array
        :return:
        """

        MIs = self.feature_label_MIs(X, y)
        max_MI_arg = np.argmax(MIs)
        feature_num = X.shape[1]
        selected_features = []
        MIs = list(zip(range(len(MIs)), MIs))
        selected_features.append(MIs.pop(int(max_MI_arg)))
        ffMIs = [[None for _ in range(X.shape[1])] for _ in range(X.shape[1])]

        # 预处理各种特征之间的交叉熵

        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                ffMIs[i][j] = ffMIs[j][i] = self.feature_feature_MIs(X[:, i], X[:, j])

        while True:
            max_theta = float("-inf")
            max_theta_index = None

            for mi_outset in MIs:
                ff_mis = [ffMIs[mi_outset[0]][mi_inset[0]] for mi_inset in selected_features]
                theta = mi_outset[1] - 1 / len(selected_features) * sum(ff_mis)
                if theta >= max_theta:
                    max_theta = theta
                    max_theta_index = mi_outset
            selected_features.append(max_theta_index)
            MIs.remove(max_theta_index)

            if len(selected_features) >= feature_num - 1:
                break

        self._selected_features = [ind for ind, mi in selected_features]

    def transform(self, X, k: int):
        return X[:, self._selected_features[:k]]

    def fit_transform(self, X, y, k):
        self.fit(X, y)
        return self.transform(X, k)

    def entropy(self, c):
        """
        entropy calculation
        :param c:
        :return:
        """
        c_normalized = c / float(np.sum(c))
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        H = -sum(c_normalized * np.log2(c_normalized))
        return H

    def feature_label_MIs(self, arr, y):
        """
        calculate feature-label mutual information
        :param arr:
        :param y:
        :return:
        """
        m, n = arr.shape
        MIs = []
        p_y = np.histogram(y)[0]
        h_y = self.entropy(p_y)

        for i in range(n):
            p_i = np.histogram(arr[:, i])[0]
            p_iy = np.histogram2d(arr[:, 0], y)[0]

            h_i = self.entropy(p_i)
            h_iy = self.entropy(p_iy)

            MI = h_i + h_y - h_iy
            MIs.append(MI)
        return MIs

    def feature_feature_MIs(self, x, y):
        """
        calculate feature-feature mutual information
        :param x:
        :param y:
        :return:
        """
        p_x = np.histogram(x)[0]
        p_y = np.histogram(y)[0]
        p_xy = np.histogram2d(x, y)[0]

        h_x = self.entropy(p_x)
        h_y = self.entropy(p_y)
        h_xy = self.entropy(p_xy)

        return h_x + h_y - h_xy

    @property
    def important_features(self):
        return self._selected_features
