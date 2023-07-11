#!/usr/bin/python
# coding=utf-8


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import Learn
import numpy as np
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier



def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0.0] * len(x)
    x = x.tolist()
    y = y.tolist()
    if (x == zero_list) or (y == zero_list):
        return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    # method 2
    # cos = bit_product_sum(x, y) / (np.sqrt(bit_product_sum(x, x)) * np.sqrt(bit_product_sum(y, y)))

    # method 3
    # dot_product, square_sum_x, square_sum_y = 0, 0, 0
    # for i in range(len(x)):
    #     dot_product += x[i] * y[i]
    #     square_sum_x += x[i] * x[i]
    #     square_sum_y += y[i] * y[i]
    # cos = dot_product / (np.sqrt(square_sum_x) * np.sqrt(square_sum_y))

    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内


def pointRecommendataion(xt, xs, yt, ys):
    index = 0
    t = 0
    c1 = 0
    c2 = 0
    c3 = 0
    for i in xt:
        simi = []
        for j in xs:
            simi.append(cosine_similarity(i, j))
        mix = simi.index(max(simi))
        ysp = ys[mix].A
        ytt = yt[index].A
        # print(x[0][1])
        # exit()
        # print(ys[mix],yt[index])
        # print(ys[mix].T[0])
        if ysp[0][3] == ytt[0][3]:
            t += 1
        if ytt[0][0] == ytt[0][3]:
            c1 += 1
        if ytt[0][1] == ytt[0][3]:
            c2 += 1
        if ytt[0][2] == ytt[0][3]:
            c3 += 1
        index += 1
    print(t / (index + 1))
    print(c1 / (index + 1))
    print(c2 / (index + 1))
    print(c3 / (index + 1))


def datasets(s):
    d1 = '/Users/sherwinwang/Documents/Project/winequality-red.csv'
    d2 = '/Users/sherwinwang/Documents/Project/DryBeanDataset/Dry_Bean_Dataset.csv'
    d3 = '/Users/sherwinwang/Documents/Project/abalone/abalone.csv'
    reader = pd.read_csv(d1, sep=';')
    Xdata = reader.iloc[:, :-1]
    iris_dataset = load_iris()

    reader2 = pd.read_csv(d2)
    Xdata2 = reader2.iloc[:, :-1]

    reader3 = pd.read_csv(d3)
    Xdata3 = reader3.iloc[:, 1:]
    # print(reader3)
    # print(reader3.iloc[:, 1:])
    # exit()


    if s == 'iris':
        X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'],
                                                            random_state=2)

    elif s == 'winequality':
        X_train, X_test, y_train, y_test = train_test_split(Xdata, reader['quality'], random_state=6)
    elif s == 'Dry_Bean_Dataset':
        X_train, X_test, y_train, y_test = train_test_split(Xdata2, reader2['Class'], random_state=6)
    elif s == 'abalone':
        X_train, X_test, y_train, y_test = train_test_split(Xdata3, reader3['Sex'], random_state=6)
    else:
        X_train, X_test, y_train, y_test = [], [], [], []

    return X_train, X_test, y_train, y_test


def modelScores(X_train, X_test, y_train, y_test):
    test_pre_li = []
    clf_li = []
    n = 0
    learn = Learn.Learn()
    # list = [1, 2, 3, 5, 6]
    # list1 = [2,3,4,5,7]
    # li = [list,list1]
    # m = np.mat(li)  # 列表转换成NumPy的矩阵
    # #
    # print(m.T)
    # exit()
    for i in range(5):
        clf = learn.reModel(i)
        # xtt, ytt = self.X_Y(self.train[i])
        # X_tt, y_tt = self.X_Y(self.Sn)
        clf.fit(X_train, y_train)
        # print(clf)
        clf_li.append(clf)
        print('Score', clf.score(X_test, y_test))
        if n == 0:
            test = clf.predict_proba(X_test)
            test_pre = clf.predict(X_test)
            n += 1
        else:
            test = np.hstack((test, clf.predict_proba(X_test)))
            test_pre = clf.predict(X_test)
        test_pre_li.append(test_pre)
    test_pre_li.append(y_test)
    test_pre_mat = np.mat(test_pre_li).T
    # print(np.mat(test_pre_li).T)
    # print(np.hstack((test,test_pre_mat)))

    # Create list of column names with the format "colN" (from 1 to N)
    col_names = ['col' + str(i) for i in np.arange(test.shape[1]) + 1]
    # Declare pandas.DataFrame object
    df = pd.DataFrame(data=test, columns=col_names)
    # df_ = df
    # df['type'] = y_test
    # print(df)
    return test, test_pre_mat

def knnRecommendatation(X_t, X_v, y_t, y_v):
    training_accuracy = []
    test_accuracy = []
    # n_neighbors取值从1到10
    neighbors_settings = range(1, 11)
    for n_neighbors in neighbors_settings:
        nbrs_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
        nbrs_clf = nbrs_clf.fit(X_t, y_t)
        training_accuracy.append(nbrs_clf.score(X_t, y_t))  # 记录泛化精度
        test_accuracy.append(nbrs_clf.score(X_v, y_v))

    print(test_accuracy)
    plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
    plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.show()

def randomforestRecommendatation(X_t, X_v, y_t, y_v):
    regressor = RandomForestClassifier()
    rf_clf = regressor.fit(X_t, y_t)
    training_accuracy = rf_clf.score(X_t, y_t)  # 记录泛化精度
    test_accuracy = rf_clf.score(X_v, y_v)
    print(test_accuracy)

    training_accuracy = []
    test_accuracy = []
    # n_neighbors取值从1到10
    # neighbors_settings = range(1, 11)
    # for n_neighbors in neighbors_settings:
    #     nbrs_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    #     nbrs_clf = nbrs_clf.fit(X_t, y_t)
    #     training_accuracy.append(nbrs_clf.score(X_t, y_t))  # 记录泛化精度
    #     test_accuracy.append(nbrs_clf.score(X_v, y_v))
    #
    # print(test_accuracy)
    # plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
    # plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
    # plt.ylabel("Accuracy")
    # plt.xlabel("n_neighbors")
    # plt.legend()
    # plt.show()


def recommendatation(test, test_pre_mat):

    t_type = pd.DataFrame(data=test_pre_mat).iloc[:, -1]
    X_test1, X_validation, y_test1, y_validation = train_test_split(test, test_pre_mat, random_state=5)
    pointRecommendataion(X_validation, X_test1, y_validation, y_test1)

    X_test2, X_validation2, y_test2, y_validation2 = train_test_split(test, t_type, random_state=5)
    knnRecommendatation(X_test2, X_validation2, y_test2, y_validation2)
    randomforestRecommendatation(X_test2, X_validation2, y_test2, y_validation2)


    # y_vt = pd.DataFrame(data=y_validation).iloc[:, -1]

    # print(y_vt.tolist())
    # for cl in clf_li:
    #     print(X_validation.iloc[:,:-3])
    #     print(('Score', cl.score(X_validation, y_vt)))
    # df['type'] = y_test
    # print(df)
    # print(X_validation)
    # print('Score', clf.score(X_test, y_test))
    # print(type(clf.predict_proba(X_test)))

    # scores = cross_val_score(clf, X_tt, y_tt, cv=5)  # cv为迭代次数。
    # print(scores)  # 打印输出每次迭代的度量值（准确度）
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 获取置信区间。（也就是均值和方差）

def main():

    # s = 'iris'
    s = 'winequality'
    s = 'Dry_Bean_Dataset'
    s = 'abalone'
    X_t, X_tt, y_t, y_tt = datasets(s)
    tt, tt_pre_mat = modelScores(X_t, X_tt, y_t, y_tt)
    recommendatation(tt, tt_pre_mat)





if __name__ == '__main__':
    main()