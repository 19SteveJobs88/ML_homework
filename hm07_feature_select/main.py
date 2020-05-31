#coding:utf-8
import scipy.io as scio
import numpy as np
from autograd.dataset import DataLoader
from autograd.model.ReliefF import ReliefF
from autograd.model.MRMR import MRMR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import label_binarize


def core_test(model, train_data, train_label, test_data, test_label):
    y_one_hot = label_binarize(test_label, np.arange(9))
    model.fit(train_data, train_label)
    y_score = label_binarize(model.predict(test_data), np.arange(9))
    fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(), y_score.ravel())
    auc_v = metrics.auc(fpr, tpr)
    acc_v = model.score(test_data, test_label)
    return acc_v, auc_v


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import copy
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

    path = r'urban.mat'
    data = scio.loadmat(path)

    test_model = {
        'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='kd_tree'),
        'NB': MultinomialNB(),
        'SVM': OneVsRestClassifier(svm.SVC(kernel='linear')),
        'RF': RandomForestClassifier(random_state=0),
    }
    namess = ['KNN', 'NB', 'SVM', 'RF']

    for more_size in (0, 50, 100, 150, 200):
        p = [[] for _ in range(4)]
        pp = [[] for _ in range(4)]
        R = ReliefF()
        if more_size != 0:
            dataX = copy.deepcopy(np.concatenate((data['X'], np.random.normal(0, 100, size=[data['X'].shape[0], more_size])), axis=1))
        else:
            dataX = copy.deepcopy(data['X'])
        # print(dataX.shape[0])
        # print(dataX.shape)
        # assert 0
        R.fit(dataX, data['Y'].ravel())
        # M = MRMR()
        # M.fit(data['X'], data['Y'].ravel())

        for i in range(1, 6):
            print(i)
            x = R.transform(dataX, i * dataX.shape[0] // 6)
            ACC = []
            AUC = []
            data_loader = DataLoader(train=x, label=data['Y'])
            train_data, test_data, train_label, test_label = data_loader.split(no_shuffle=True)
            train_label = train_label.ravel()
            test_label = test_label.ravel()
            for name, value in test_model.items():
                acc, auc = core_test(value, train_data, train_label, test_data, test_label)
                ACC.append(acc)
                AUC.append(auc)
                # print(name, f'acc is {acc},auc is {auc}')
            for ii in range(4):
                p[ii].append(ACC[ii])
                pp[ii].append(AUC[ii])
            # print()
            # print()
        plt.figure()
        plt.xlabel('使用特征数量')
        plt.ylabel('准确率')
        plt.title(f'使用relieff算法新增{more_size}随机特征')
        for ii in range(4):
            plt.plot([i * dataX.shape[1] // 6 for i in range(1, 6)], p[ii], label=namess[ii])
        plt.legend()
        plt.savefig(f'Racc_{more_size}.png')
        plt.figure()
        plt.xlabel('使用特征数量')
        plt.ylabel('AUC值')
        plt.title(f'使用relieff算法新增{more_size}随机特征')
        for ii in range(4):
            plt.plot([i * dataX.shape[1] // 6 for i in range(1, 6)], pp[ii], label=namess[ii])
        plt.legend()
        plt.savefig(f'Rauc_{more_size}.png')
    # assert 0
