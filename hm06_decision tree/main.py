import numpy as np
from autograd.model.DecisionTree import C45
from sklearn import tree

feature_dict = {"色泽": ["青绿", "乌黑", "浅白"],
                "根蒂": ["蜷缩", "稍蜷", "硬挺"],
                "敲声": ["浊响", "沉闷", "清脆"],
                "纹理": ["清晰", "稍糊", "模糊"]
                }
lable_list = ["否", "是"]
feature_list = ["色泽", "根蒂", "敲声", "纹理"]


def load_data(path):
    datas = []
    labels = []
    with open(path, "r", encoding="GBK") as f:
        line = f.readline()
        line = f.readline()
        while line:
            d = line.rstrip("\r\n").split(',')
            re = []
            re.append(feature_dict.get("色泽").index(d[1]))
            re.append(feature_dict.get("根蒂").index(d[2]))
            re.append(feature_dict.get("敲声").index(d[3]))
            re.append(feature_dict.get("纹理").index(d[4]))
            if '2' in path:
                re.append(float(d[5]))
            # re.append(lable_list.index(d[-1]))
            datas.append(np.array(re))
            labels.append(np.array(lable_list.index(d[-1])))
            line = f.readline()
    return np.array(datas), np.array(labels)


if __name__ == '__main__':
    from sklearn.metrics import accuracy_score

    X_train_t, Y_train_t = load_data('Watermelon-train2.csv')
    X_test, Y_test = load_data('Watermelon-test2.csv')
    Tree = C45()
    A=[]
    for i in range(1,len(X_train_t)):
        a = X_train_t[:i, :]
        b = X_train_t[i:, :]
        c = Y_train_t[:i]
        d = Y_train_t[i:]
        X_train = a
        X_valid = b
        Y_train = c
        Y_valid = d
        Tree = tree.DecisionTreeClassifier()
        Tree.fit(X_train_t,Y_train_t)
        print(accuracy_score(Tree.predict(X_test),Y_test))
        assert 0
        Tree = ID3()
        # print(Tree.predict(X_test),Y_test)
        Tree.fit_prun(X_train, Y_train, X_valid, Y_valid)
        A.append(Tree.cal_acc(Y_test, None, X_test))
        print(Tree.cal_acc(Y_test, None, X_test))
    import matplotlib.pyplot as plt
    XX  = [i/len(X_train_t) for i in range(1,len(X_train_t))]
    plt.plot(XX,A)
    plt.show()
    # Tree= C45()
    # Tree.fit(X_train_t,Y_train_t,[0,1,2,3],[4])
    # print()
    # print(Tree.cal_acc(Y_test,None,X_test))
    # Tree.fit(X_train, Y_train, [0, 1, 2, 3], [4])
    # ans = Tree.cal_acc(Y_test, None, X_test)
    # print(ans)
