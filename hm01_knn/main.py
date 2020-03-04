# -*- coding:utf-8 -*-
import os
import numpy as np
from typing import Iterator, Tuple, List, NewType
from sklearn.neighbors import KNeighborsClassifier

Array = NewType('Array', np.ndarray)


# 获取类标签
def getLabel(hot: List, size: int) -> int:
    for i in range(size):
        if hot[i] == '1':
            return i
    return -1


def getDistMatrix(train_data: Array, test_data: Array, dis_function):
    mat = []
    for i in train_data:
        mat.append(dis_function(i, test_data))
    return mat


def mydis_gen(order: int = 2):
    return lambda x, y: np.linalg.norm(x - y, order)


def myknn(train_data: Array, test_data: Array, train_label: Array, test_label: Array, n_neighbors: int,
          dis_algorithm) -> Tuple[int, List]:
    # 通过K近邻分类，获取测试集的预测标签
    correct_cnt = 0
    pre_label = []
    it = iter(test_label)
    for test in test_data:
        dist = getDistMatrix(train_data, test, dis_algorithm)
        distSorted = np.argsort(dist)
        classCount = [0 for _ in range(10)]
        for num in range(n_neighbors):
            classCount[int(train_label[distSorted[num]])] += n_neighbors - num + 1
        pre_label.append(int(np.argmax(classCount)))
        if pre_label[-1] == next(it):
            correct_cnt += 1
    return correct_cnt, pre_label


class DataLoader:
    train: Array
    label: np.ndarray

    def __init__(self, path: str, *args):
        assert 'csv' in path
        train = []
        label = []
        with open(path, 'r') as train_file:
            for row in train_file:
                line = row.strip().split(' ')
                train.append(line[:-10])
                label.append(getLabel(line[-10:], 10))
        for other_path in args:
            with open(other_path, 'r') as train_file:
                for row in train_file:
                    line = row.strip().split(' ')
                    train.append(line[:-10])
                    label.append(getLabel(line[-10:], 10))
        # 归一化处理
        train = np.array(train, dtype=float)
        label = np.array(label, dtype=int)
        min_data = train.min(0)
        max_data = train.max(0)
        train = (train - min_data) / (max_data - min_data)
        self.train = train
        self.label = label

    def KFold(self, n_splits: int = 10) -> Iterator[Tuple[Array, Array, Array, Array]]:
        state = np.random.get_state()
        np.random.shuffle(self.train)
        np.random.set_state(state)
        np.random.shuffle(self.label)
        total_nums = self.train.shape[0]
        little_nums = total_nums // n_splits
        pieces = [self.train[i * little_nums:(i + 1) * little_nums] for i in range(n_splits)]
        labels = [self.label[i * little_nums:(i + 1) * little_nums] for i in range(n_splits)]
        for i in range(n_splits):
            yield np.array([pieces[j] for j in range(n_splits) if j != i]).reshape((-1, 256)), pieces[i], np.array(
                [labels[j] for j in range(n_splits) if j != i]).reshape((-1, 1)), np.array(labels[i])

    def simple(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.train, self.label


def main():
    data_loader = DataLoader(os.path.dirname(__file__) + '/semeion_train.csv')
    test_loader = DataLoader(os.path.dirname(__file__) + '/semeion_test.csv')
    total_loader = DataLoader(os.path.dirname(__file__) + '/semeion_train.csv',
                              os.path.dirname(__file__) + '/semeion_test.csv')
    # ans = []
    # for k in range(1, 10):
    #     ave_acc = 0
    #     for train_data, test_data, train_label, test_label in data_loader.KFold(n_splits=4):
    #         correct_cnt, pre_label = myknn(train_data, test_data, train_label, test_label, k, mydis(2))
    #         acc = correct_cnt / np.shape(test_data)[0]
    #         ave_acc += acc
    #     ans.append(ave_acc / 4)
    #     print('k为', k, '时,训练集分类精度为%.2f' % (ave_acc / 4 * 100), '%')
    # import matplotlib.pyplot as plt
    # plt.plot(list(range(1, 10)), ans)
    # plt.show()
    train_data, train_label = data_loader.simple()
    test_data, test_label = test_loader.simple()
    for k in (5, 6, 7):
        correct_cnt, pre_label = myknn(train_data, test_data, train_label, test_label, k, mydis_gen(2))
        acc = correct_cnt / np.shape(test_data)[0]
        print('测试分类精度为%.02f' % (acc * 100), '%')

    for k in range(1, 8):
        KNN = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='kd_tree')
        KNN.fit(train_data, train_label)
        print(KNN.score(test_data, test_label))


if __name__ == '__main__':
    main()
