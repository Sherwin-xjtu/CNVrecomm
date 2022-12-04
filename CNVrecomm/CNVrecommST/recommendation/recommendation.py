#!/usr/bin/python
# coding=utf-8


from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
import pandas as pd
import Learn
import numpy as np
from sklearn import neighbors, preprocessing, linear_model
from SimilarityCalculation import Similarity
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, \
    RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, RANSACRegressor, ARDRegression, HuberRegressor, TheilSenRegressor, \
    SGDRegressor, PassiveAggressiveRegressor, Lasso, ElasticNet, Ridge, BayesianRidge, Lars


warnings.filterwarnings('ignore')


def func1(amount, num):
    list1 = []
    total = 0
    for i in range(0, num - 1):
        a = np.random.random(amount)  # 生成 n-1 个随机节点
        list1.append(a.tolist()[0])
    list1.sort()  # 节点排序
    list1.append(amount)  # 设置第 n 个节点为amount，即总金额

    list2 = []
    for i in range(len(list1)):
        if i == 0:
            b = list1[i]  # 第一段长度为第 1 个节点 - 0
        else:
            b = list1[i] - list1[i - 1]  # 其余段为第 n 个节点 - 第 n-1 个节点
        list2.append(b)

    # print(list2)

    # for ele in range(0, len(list2)):
    #     total = total + list2[ele]
    li = list2
    maxnum = max(li)
    return maxnum
    # minnum = min(li)
    # li = [x for x in li if x != maxnum]
    # li.append(maxnum)
    # print(li)

    # print("列表元素之和为: ", total)


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0.0] * len(x)

    x = x.tolist()
    y = y.tolist()
    if (x == zero_list) or (y == zero_list):
        return float(1) if x == y else float(0)
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

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
        # print(ytt)
        # exit()
        if ysp[0][-1] == ytt[0][-1]:
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


class pointRecommendataionClassifier:

    def __init__(self, x_database, y_database):
        self.x_database = x_database
        self.y_database = y_database.tolist()
        self.probabilities = []

    def predict(self, x):

        ysp_li = []
        for i in x:
            simi = []
            for j in self.x_database:
                simi.append(cosine_similarity(i, j))
            probability = max(simi)
            self.probabilities.append(probability)
            mix = simi.index(max(simi))
            ysp = self.y_database[mix]
            ysp_li.append(ysp)
        return ysp_li


class pointRecommendataionClassifier0:

    def __init__(self, x_database, y_database):
        self.x_database = np.array(x_database)
        self.y_database = np.array(y_database)
        self.probabilities = []

    def predict(self, x):

        ysp_li = []
        x_np = np.array(x)

        for i in x_np:
            simi = []
            for j in self.x_database:
                simi.append(cosine_similarity(i, j))
            probability = max(simi)
            self.probabilities.append(probability)
            mix = simi.index(max(simi))
            ysp = self.y_database[mix]
            ysp_li.append(ysp)
        return ysp_li


class pointRecommendataionClassifier1:

    def __init__(self, x_database, y_database):
        Xdata2 = x_database.drop(['pro', 'I', 'J', 'K', 'L', 'M'], axis=1, inplace=False)
        self.min_max_scaler = preprocessing.MinMaxScaler()
        X_train_minmax = self.min_max_scaler.fit_transform(Xdata2)
        self.pro = np.array(x_database['pro'])
        self.x_database = np.array(X_train_minmax)
        self.y_database = np.array(y_database)
        self.probabilities = []

    def predict(self, x):

        ysp_li = []
        x2 = x.drop(['pro', 'I', 'J', 'K', 'L', 'M'], axis=1, inplace=False)
        X_test_minmax = self.min_max_scaler.transform(x2)
        x_np = np.array(X_test_minmax)
        similarity = Similarity()
        for i in x_np:
            simi = []
            for j in range(len(self.x_database)):
                # similar = similarity.Cosine(i, self.x_database[j])*self.pro[j]
                similar = similarity.Cosine(i, self.x_database[j])
                # print(similarity.Cosine(i, self.x_database[j]), self.pro[j], similar)
                simi.append(similar)

            probability = max(simi)
            self.probabilities.append(probability)
            mix = simi.index(max(simi))
            ysp = self.y_database[mix]
            ysp_li.append(ysp)
        return ysp_li


class pointRecommendataionClassifier2:

    def __init__(self, x_database, y_database):
        Xdata2 = x_database.drop(['pro', 'nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv'], axis=1, inplace=False)
        self.min_max_scaler = preprocessing.MinMaxScaler()
        X_train_minmax = self.min_max_scaler.fit_transform(Xdata2)
        self.pro = np.array(x_database['pro'])
        self.x_database = np.array(X_train_minmax)
        self.y_database = np.array(y_database)
        self.probabilities = []

    def predict(self, x):

        ysp_li = []
        x2 = x.drop(['pro', 'nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv'], axis=1, inplace=False)
        X_test_minmax = self.min_max_scaler.transform(x2)
        x_np = np.array(X_test_minmax)
        similarity = Similarity()
        for i in x_np:
            simi = []
            for j in range(len(self.x_database)):
                similar = similarity.Cosine(i, self.x_database[j]) * self.pro[j]
                # similar = similarity.Cosine(i, self.x_database[j])
                # print(similarity.Cosine(i, self.x_database[j]), self.pro[j], similar)
                simi.append(similar)
            data1 = {'simi': simi, 'type': self.y_database.tolist()}
            df = pd.DataFrame(data1)
            df2 = df[df['type'] == 'pbsv']
            pr_pbsv = df2['simi'].tolist()
            df3 = df[df['type'] == 'cutesv']
            pr_cutesv = df3['simi'].tolist()
            df4 = df[df['type'] == 'nanosv']
            pr_nanosv = df4['simi'].tolist()
            df5 = df[df['type'] == 'sniffles']
            pr_sniffles = df5['simi'].tolist()
            df6 = df[df['type'] == 'picky']
            pr_picky = df6['simi'].tolist()
            pr = [np.mean(pr_pbsv), np.mean(pr_cutesv), np.mean(pr_nanosv), np.mean(pr_sniffles), np.mean(pr_picky)]
            # probability = max(pr)
            mix = pr.index(max(pr))
            # print(mix, pr)
            ma = {0: 'pbsv', 1: 'cutesv', 2: 'nanosv', 3: 'sniffles', 4: 'picky'}
            ysp = ma[mix]
            # for index, row in df2.iterrows():
            #     print(row)
            #     exit()

            # List3 = np.multiply(np.array(simi), np.array(self.pro))
            # List3 = List3.tolist()
            #
            # probability = max(List3)
            # self.probabilities.append(probability)
            # mix = simi.index(max(simi))
            # ysp = self.y_database[mix]
            # print(ysp)
            ysp_li.append(ysp)
        return ysp_li


class pointRecommendataionClassifier3:

    def __init__(self, x_database, y_database):
        Xdata2 = x_database.drop(['pro', 'nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv'], axis=1, inplace=False)
        x_toolspro = x_database.iloc[:, -6:-1]
        # print(np.array(x_toolspro))
        self.x_toolspro = np.array(x_toolspro)
        self.min_max_scaler = preprocessing.MinMaxScaler()
        X_train_minmax = self.min_max_scaler.fit_transform(Xdata2)
        self.pro = np.array(x_database['pro'])
        self.x_database = np.array(X_train_minmax)
        self.y_database = np.array(y_database)
        self.probabilities = []

    def predict(self, x, y):

        ysp_li = []
        x2 = x.drop(['pro', 'nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv'], axis=1, inplace=False)
        X_test_minmax = self.min_max_scaler.transform(x2)
        x_np = np.array(X_test_minmax)
        similarity = Similarity()
        # index = 0
        for i in x_np:
            # yt = y[index]
            simi = []
            for j in range(len(self.x_database)):
                similar = similarity.Cosine(i, self.x_database[j]) * self.x_toolspro[j]
                # similar = similarity.Cosine(i, self.x_database[j])
                # print(similarity.Cosine(i, self.x_database[j]), self.x_toolspro[j], similar)

                simi.append(similar)

            nr = pd.DataFrame(np.array(simi))
            nr.columns = ['nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv']
            # nr = nr.iloc[11:12, :]
            print(nr)
            exit()
            nanosv_sum = sum(nr['nanosv'].tolist()) * 0.7

            picky_sum = sum(nr['picky'].tolist()) * 0.6
            sniffles_sum = sum(nr['sniffles'].tolist()) * 0.5
            pbsv_sum = sum(nr['pbsv'].tolist()) * 0.6
            cutesv_sum = sum(nr['cutesv'].tolist()) * 0.7
            pr = [nanosv_sum, picky_sum, sniffles_sum, pbsv_sum, cutesv_sum]
            # print(pr)
            mix = pr.index(max(pr))
            # print(mix, pr)
            ma = {0: 'nanosv', 1: 'picky', 2: 'sniffles', 3: 'pbsv', 4: 'cutesv'}
            ysp = ma[mix]
            # print(ysp, yt)
            # index += 1
            # exit()
            ysp_li.append(ysp)
        return ysp_li


class pointRecommendataionClassifier4:

    def __init__(self, x_database, y_database):
        Xdata2 = x_database.drop(['pro', 'nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv'], axis=1, inplace=False)
        x_toolspro = x_database.iloc[:, -6:-1]
        # print(np.array(x_toolspro))
        self.x_toolspro = np.array(x_toolspro)
        self.min_max_scaler = preprocessing.MinMaxScaler()
        X_train_minmax = self.min_max_scaler.fit_transform(Xdata2)
        self.pro = np.array(x_database['pro'])
        self.x_database = np.array(X_train_minmax)
        self.y_database = np.array(y_database)
        self.probabilities = []

    def predict(self, x, y):

        ysp_li = []
        x2 = x.drop(['pro', 'nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv'], axis=1, inplace=False)
        X_test_minmax = self.min_max_scaler.transform(x2)
        x_np = np.array(X_test_minmax)
        similarity = Similarity()
        # index = 0
        for i in x_np:
            # yt = y[index]
            simi = []
            for j in range(len(self.x_database)):
                similar = similarity.Cosine(i, self.x_database[j])
                # similar = similarity.Cosine(i, self.x_database[j])
                # print(similarity.Cosine(i, self.x_database[j]), self.x_toolspro[j], similar)

                simi.append(similar)

            a = np.array(simi)

            scalar = MinMaxScaler(feature_range=(0, 1))  # 加载函数
            b = scalar.fit_transform(self.x_toolspro)  # 归一化
            # print(a,b)
            # print(b.T)
            c = b.T * simi
            nr = pd.DataFrame(c)
            nr = nr.T
            nr.columns = ['nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv']
            # nr = nr.iloc[11:12, :]
            # print(nr)
            # # exit()
            nanosv_sum = np.median(nr['nanosv'].tolist())

            picky_sum = np.median(nr['picky'].tolist())
            sniffles_sum = np.median(nr['sniffles'].tolist())
            pbsv_sum = np.median(nr['pbsv'].tolist())
            cutesv_sum = np.median(nr['cutesv'].tolist())
            pr = [nanosv_sum, picky_sum, sniffles_sum, pbsv_sum, cutesv_sum]
            # print(pr)
            mix = pr.index(max(pr))
            # print(mix, pr)
            ma = {0: 'nanosv', 1: 'picky', 2: 'sniffles', 3: 'pbsv', 4: 'cutesv'}
            ysp = ma[mix]
            # print(ysp, yt)
            # index += 1
            # exit()
            ysp_li.append(ysp)
        return ysp_li


class pointRecommendataionClassifier5:

    def __init__(self, x_database, y_database):
        Xdata2 = x_database.drop(['pro', 'nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv'], axis=1, inplace=False)
        x_toolspro = x_database.iloc[:, -6:-1]
        x_toolspro['type'] = y_database

        y_nanosv = x_toolspro[x_toolspro['type'] == 'nanosv']
        y_nanosv_avg = y_nanosv[['nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv']].mean(axis=0)

        y_picky = x_toolspro[x_toolspro['type'] == 'picky']
        y_picky_avg = y_picky[['nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv']].mean(axis=0)

        y_sniffles = x_toolspro[x_toolspro['type'] == 'sniffles']
        y_sniffles_avg = y_sniffles[['nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv']].mean(axis=0)

        y_pbsv = x_toolspro[x_toolspro['type'] == 'pbsv']
        y_pbsv_avg = y_pbsv[['nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv']].mean(axis=0)

        y_cutesv = x_toolspro[x_toolspro['type'] == 'cutesv']
        y_cutesv_avg = y_cutesv[['nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv']].mean(axis=0)
        # print(x_toolspro['nanosv'].mean())
        # x_nanosv =

        # print(y_cutesv_avg.tolist())
        y_proavg_dic = {'nanosv': y_nanosv_avg.tolist(), 'picky': y_picky_avg.tolist(),
                        'sniffles': y_sniffles_avg.tolist(), 'pbsv': y_pbsv_avg.tolist(),
                        'cutesv': y_cutesv_avg.tolist()}

        self.y_proavg_df = pd.DataFrame(y_proavg_dic).T
        self.y_proavg_df.columns = ['nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv']

        # print(np.array(x_toolspro))
        self.x_toolspro = np.array(x_toolspro)
        self.min_max_scaler = preprocessing.MinMaxScaler()
        X_train_minmax = self.min_max_scaler.fit_transform(Xdata2)
        self.pro = np.array(x_database['pro'])
        self.x_database = np.array(X_train_minmax)
        self.y_database = np.array(y_database)
        self.probabilities = []

    def predict(self, x, y):

        ysp_li = []
        x2 = x.drop(['pro', 'nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv'], axis=1, inplace=False)
        X_test_minmax = self.min_max_scaler.transform(x2)
        x_np = np.array(X_test_minmax)
        similarity = Similarity()

        index = 0
        for i in x_np:
            yt = y[index]
            index += 1
            print(yt)
            simi = []
            inc = 0
            for j in range(len(self.x_database)):
                similar = similarity.Cosine(i, self.x_database[j]) * self.y_proavg_df
                # similar = similarity.Cosine(i, self.x_database[j])
                # print(similarity.Cosine(i, self.x_database[j]), self.x_toolspro[j], similar)
                if inc == 0:
                    similar_concat = similar
                    inc += 1
                else:
                    similar_concat = pd.concat([similar_concat, similar])
                simi.append(similar)

            print(similar_concat)
            print(similar_concat.mean())
            # print(similar_concat.max())
            exit()
            a = np.array(simi)

            scalar = MinMaxScaler(feature_range=(0, 1))  # 加载函数
            b = scalar.fit_transform(self.x_toolspro)  # 归一化
            # print(a,b)
            # print(b.T)
            c = b.T * simi
            nr = pd.DataFrame(c)
            nr = nr.T
            nr.columns = ['nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv']
            # nr = nr.iloc[11:12, :]
            # print(nr)
            # # exit()
            nanosv_sum = np.median(nr['nanosv'].tolist())

            picky_sum = np.median(nr['picky'].tolist())
            sniffles_sum = np.median(nr['sniffles'].tolist())
            pbsv_sum = np.median(nr['pbsv'].tolist())
            cutesv_sum = np.median(nr['cutesv'].tolist())
            pr = [nanosv_sum, picky_sum, sniffles_sum, pbsv_sum, cutesv_sum]
            # print(pr)
            mix = pr.index(max(pr))
            # print(mix, pr)
            ma = {0: 'nanosv', 1: 'picky', 2: 'sniffles', 3: 'pbsv', 4: 'cutesv'}
            ysp = ma[mix]
            # print(ysp, yt)

            # exit()
            ysp_li.append(ysp)
        return ysp_li


class pointRecommendataionClassifier6:

    def __init__(self, x_database, y_database):

        x_database['type'] = y_database

        data_nanosv = x_database[x_database['type'] == 'nanosv']
        data_nanosv_avg = data_nanosv.mean()
        # print(data_nanosv_avg.tolist())
        # exit()
        data_picky = x_database[x_database['type'] == 'picky']
        data_picky_avg = data_picky.mean()

        data_sniffles = x_database[x_database['type'] == 'sniffles']
        data_sniffles_avg = data_sniffles.mean()

        data_pbsv = x_database[x_database['type'] == 'pbsv']
        data_pbsv_avg = data_pbsv.mean()

        data_cutesv = x_database[x_database['type'] == 'cutesv']
        data_cutesv_avg = data_cutesv.mean()

        data_avg_dic = {'nanosv': data_nanosv_avg.tolist(), 'picky': data_picky_avg.tolist(),
                        'sniffles': data_sniffles_avg.tolist(), 'pbsv': data_pbsv_avg.tolist(),
                        'cutesv': data_cutesv_avg.tolist()}

        self.data_avg_df = pd.DataFrame(data_avg_dic).T

        self.data_avg_df.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'nanosv', 'picky', 'sniffles', 'pbsv',
                                    'cutesv', 'pro']
        Xdata2 = self.data_avg_df.drop(['pro', 'nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv'], axis=1, inplace=False)
        x_toolspro = self.data_avg_df.iloc[:, -6:-1]

        # print(np.array(x_toolspro))
        # self.x_toolspro = np.array(x_toolspro)
        self.x_toolspro = x_toolspro
        self.min_max_scaler = preprocessing.MinMaxScaler()
        X_train_minmax = self.min_max_scaler.fit_transform(Xdata2)
        self.pro = np.array(x_database['pro'])
        self.x_database = np.array(X_train_minmax)
        self.y_database = np.array(y_database)
        self.probabilities = []

    def predict(self, x, y):

        ysp_li = []
        x2 = x.drop(['pro', 'nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv'], axis=1, inplace=False)
        X_test_minmax = self.min_max_scaler.transform(x2)
        x_np = np.array(X_test_minmax)
        similarity = Similarity()

        index = 0
        for i in x_np:
            yt = y[index]
            index += 1
            print(yt)
            simi = []
            inc = 0
            for j in range(len(self.x_database)):

                similar = similarity.Pearson(i, self.x_database[j]) * self.x_toolspro
                print(similarity.Pearson(i, self.x_database[j]))
                print(self.x_toolspro)
                # similar = similarity.Cosine(i, self.x_database[j])
                # print(similarity.Cosine(i, self.x_database[j]), self.x_toolspro[j], similar)
                if inc == 0:
                    similar_concat = similar
                    inc += 1
                else:
                    similar_concat = pd.concat([similar_concat, similar])
                simi.append(similar)

            # print(similar_concat)
            print(similar_concat.mean())
            print(similar_concat.max())
            print(similar_concat.median())
            exit()
            a = np.array(simi)

            scalar = MinMaxScaler(feature_range=(0, 1))  # 加载函数
            b = scalar.fit_transform(self.x_toolspro)  # 归一化
            # print(a,b)
            # print(b.T)
            c = b.T * simi
            nr = pd.DataFrame(c)
            nr = nr.T
            nr.columns = ['nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv']
            # nr = nr.iloc[11:12, :]
            # print(nr)
            # # exit()
            nanosv_sum = np.median(nr['nanosv'].tolist())

            picky_sum = np.median(nr['picky'].tolist())
            sniffles_sum = np.median(nr['sniffles'].tolist())
            pbsv_sum = np.median(nr['pbsv'].tolist())
            cutesv_sum = np.median(nr['cutesv'].tolist())
            pr = [nanosv_sum, picky_sum, sniffles_sum, pbsv_sum, cutesv_sum]
            # print(pr)
            mix = pr.index(max(pr))
            # print(mix, pr)
            ma = {0: 'nanosv', 1: 'picky', 2: 'sniffles', 3: 'pbsv', 4: 'cutesv'}
            ysp = ma[mix]
            # print(ysp, yt)

            # exit()
            ysp_li.append(ysp)
        return ysp_li


class pointRecommendataionClassifier8:

    def __init__(self, x_database, y_database):
        self.train_cols = x_database.columns[:-6]
        self.data_cols = x_database.columns[:]

        self.clfs_name = ['nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv']
        self.clfs = []
        self.user_pros = []

        self.min_max_scaler = preprocessing.MinMaxScaler()

        X_train_minmax = self.min_max_scaler.fit_transform(x_database)
        X_train_minmax = pd.DataFrame(X_train_minmax)

        X_train_minmax.columns = self.data_cols

        for clf in self.clfs_name:
            self.clfs.append(self.linear_model_main(X_train_minmax, self.train_cols, clf))
        # self.pro = np.array(x_database['pro'])

        self.x_database = np.array(X_train_minmax[self.train_cols])
        self.y_database = np.array(y_database)
        self.probabilities = []

    def linear_model_main(self, X_parameters, train_cols, clf):
        # regr = linear_model.LinearRegression()
        regr = ExtraTreesRegressor()
        regr.fit(X_parameters[train_cols], X_parameters[clf])
        # predict_outcome = regr.predict(xt_parameters[train_cols])

        # predictions = {}
        # predictions['intercept'] = regr.intercept_
        # predictions['coefficient'] = regr.coef_
        # predictions['predicted_value'] = predict_outcome

        return regr

    def predict(self, x, y):

        ysp_li = []

        X_test_minmax = self.min_max_scaler.transform(x)
        X_test_minmax = pd.DataFrame(X_test_minmax)
        X_test_minmax.columns = self.data_cols
        user_pros = []
        # print(X_test_minmax)
        # print(X_test_minmax[self.train_cols])
        # for clf in self.clfs:
        #     user_pros.append(clf.predict(X_test_minmax[self.train_cols]))
        #     print(user_pros)
        #     exit()
        X_test_minmax = X_test_minmax.drop(['pro', 'nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv'], axis=1,
                                           inplace=False)
        x_np = np.array(X_test_minmax)
        similarity = Similarity()

        index = 0
        for i in x_np:
            yt = y[index]
            index += 1
            # print(yt)
            simi = []
            inc = 0
            user_pros = []
            for clf in self.clfs:
                user_pros.append(clf.predict(i.reshape(1, -1)).tolist()[0])
            # print(user_pros)
            # for j in range(len(self.x_database)):
            #
            #     similar = similarity.Pearson(i, self.x_database[j]) * np.array(user_pros)
            #     # print(similarity.Pearson(i, self.x_database[j]))
            #
            #     # similar = similarity.Cosine(i, self.x_database[j])
            #     # print(similarity.Cosine(i, self.x_database[j]), self.x_toolspro[j], similar)
            #
            #     simi.append(similar)

            # print(similar_concat)
            # pr = pd.DataFrame(np.array(simi)).max().tolist()

            mix = user_pros.index(max(user_pros))
            # print(mix, pr)
            ma = {0: 'nanosv', 1: 'picky', 2: 'sniffles', 3: 'pbsv', 4: 'cutesv'}
            ysp = ma[mix]
            # print(ysp, yt)

            # exit()
            ysp_li.append(ysp)
        return ysp_li


class pointRecommendataionClassifier7:

    def __init__(self, x_database, y_database):
        self.train_cols = x_database.columns[:-6]
        self.data_cols = x_database.columns[:]

        self.clfs_name = ['nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv']
        self.clfs = []
        self.user_pros = []

        self.min_max_scaler = preprocessing.MinMaxScaler()

        X_train_minmax = self.min_max_scaler.fit_transform(x_database)
        X_train_minmax = pd.DataFrame(X_train_minmax)

        X_train_minmax.columns = self.data_cols

        for clf in self.clfs_name:
            self.clfs.append(self.linear_model_main(X_train_minmax, self.train_cols, clf))
        # self.pro = np.array(x_database['pro'])

        self.x_database = np.array(X_train_minmax[self.train_cols])
        self.y_database = np.array(y_database)
        self.probabilities = []

    def linear_model_main(self, X_parameters, train_cols, clf):
        regr = linear_model.LinearRegression()
        regr = RandomForestRegressor()
        regr.fit(X_parameters[train_cols], X_parameters[clf])
        # predict_outcome = regr.predict(xt_parameters[train_cols])

        # predictions = {}
        # predictions['intercept'] = regr.intercept_
        # predictions['coefficient'] = regr.coef_
        # predictions['predicted_value'] = predict_outcome

        return regr

    def predict(self, x, y):

        ysp_li = []

        X_test_minmax = self.min_max_scaler.transform(x)
        X_test_minmax = pd.DataFrame(X_test_minmax)
        X_test_minmax.columns = self.data_cols
        user_pros = []
        # print(X_test_minmax)
        # print(X_test_minmax[self.train_cols])
        # for clf in self.clfs:
        #     user_pros.append(clf.predict(X_test_minmax[self.train_cols]))
        #     print(user_pros)
        #     exit()
        X_test_minmax = X_test_minmax.drop(['pro', 'nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv'], axis=1,
                                           inplace=False)
        x_np = np.array(X_test_minmax)
        similarity = Similarity()

        index = 0
        for i in x_np:
            yt = y[index]
            index += 1
            # print(yt)
            simi = []
            inc = 0
            user_pros = []
            for clf in self.clfs:
                user_pros.append(clf.predict(i.reshape(1, -1)).tolist()[0])
            # print(user_pros)
            for j in range(len(self.x_database)):
                similar = similarity.Pearson(i, self.x_database[j]) * np.array(user_pros)
                # print(similarity.Pearson(i, self.x_database[j]))

                # similar = similarity.Cosine(i, self.x_database[j])
                # print(similarity.Cosine(i, self.x_database[j]), self.x_toolspro[j], similar)

                simi.append(similar)

            # print(similar_concat)
            pr = pd.DataFrame(np.array(simi)).max().tolist()

            mix = pr.index(max(pr))
            # print(mix, pr)
            ma = {0: 'nanosv', 1: 'picky', 2: 'sniffles', 3: 'pbsv', 4: 'cutesv'}
            ysp = ma[mix]
            # print(ysp, yt)

            # exit()
            ysp_li.append(ysp)
        return ysp_li


class pointRecommendataionClassifier9:

    def __init__(self, x_database, y_database):
        self.train_cols = x_database.columns[:-6]
        self.data_cols = x_database.columns[:]

        self.clfs_name = ['nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv']
        self.clfs = []
        self.user_pros = []

        self.min_max_scaler = preprocessing.MinMaxScaler()

        X_train_minmax = self.min_max_scaler.fit_transform(x_database)
        X_train_minmax = pd.DataFrame(X_train_minmax)

        X_train_minmax.columns = self.data_cols

        for clf in self.clfs_name:
            self.clfs.append(self.linear_model_main(X_train_minmax, y_database, self.train_cols, clf))
        # self.pro = np.array(x_database['pro'])
        self.x_database = np.array(X_train_minmax[self.train_cols])
        self.y_database = np.array(y_database)
        self.probabilities = []


    def modelnameTomodel(self,models, model_name):
        ml = None
        for md in models:
            if md.__name__ == model_name:
                ml = md()
                break
        return ml


    def modelingToselect(self, models, datasets, y_database, clons, tool):
        train_cols = clons
        tols_metrics = dict()

        X_parameters, X_test, y_parameters, y_test = train_test_split(datasets, y_database, test_size=0.3, random_state=6)
        for i in range(len(models)):
            model_name = models[i].__name__
            # 建模、拟合
            regr = models[i]()
            regr.fit(X_parameters[train_cols], X_parameters[tool])
            y_test = X_test[tool]
            y_pred = regr.predict(X_test[train_cols])
            try:
                MSE = metrics.mean_squared_error(y_test, y_pred)
                RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                if i == 0:
                    tols_metrics['MSE'] = [MSE]
                    tols_metrics['RMSE'] = [RMSE]
                    tols_metrics['r2'] = [r2]
                    tols_metrics['model_name'] = [model_name]
                else:
                    tols_metrics['MSE'].append(MSE)
                    tols_metrics['RMSE'].append(RMSE)
                    tols_metrics['r2'].append(r2)
                    tols_metrics['model_name'].append(model_name)
            except:
                if i == 0:
                    tols_metrics['MSE'] = [np.nan]
                    tols_metrics['RMSE'] = [np.nan]
                    tols_metrics['r2'] = [np.nan]
                    tols_metrics['model_name'] = [model_name]
                else:
                    tols_metrics['MSE'].append(np.nan)
                    tols_metrics['RMSE'].append(np.nan)
                    tols_metrics['r2'].append(np.nan)
                    tols_metrics['model_name'].append(model_name)

            # plt.figure(1)
            # # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
            # # plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
            # plt.plot(range(len(y_test)), y_test, label='real test type')
            # plt.plot(range(len(y_test)), y_pred, label='pre test type')
            # plt.figure(2)
            # # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
            # # plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
            # plt.scatter(y_test, y_pred)
            # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
            # plt.xlabel('real type')
            # plt.ylabel('pre type')
            #
            # plt.show()
        tolspd_metrics = pd.DataFrame(tols_metrics)

        ma = 'r2'
        selected_model_name = tolspd_metrics.loc[tolspd_metrics[ma].idxmax()]['model_name']
        selected_model = self.modelnameTomodel(models, selected_model_name)

        return selected_model

    def linear_model_main(self, X_parameters, y_database, train_cols, clf):
        # regr = linear_model.LinearRegression()
        # regr = RandomForestRegressor()

        models = [KNeighborsRegressor, RadiusNeighborsRegressor, GradientBoostingRegressor, AdaBoostRegressor,
                   RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor,
                   # VotingRegressor,
                   DecisionTreeRegressor, ExtraTreeRegressor, SVR, MLPRegressor, GaussianProcessRegressor,
                   LinearRegression, RANSACRegressor, ARDRegression, HuberRegressor, TheilSenRegressor,
                   SGDRegressor, PassiveAggressiveRegressor, Lasso, ElasticNet, Ridge, BayesianRidge, Lars]

        regr = self.modelingToselect(models, X_parameters, y_database, train_cols, clf)

        regr.fit(X_parameters[train_cols], X_parameters[clf])
        # predict_outcome = regr.predict(xt_parameters[train_cols])

        # predictions = {}
        # predictions['intercept'] = regr.intercept_
        # predictions['coefficient'] = regr.coef_
        # predictions['predicted_value'] = predict_outcome

        return regr

    def predict(self, x, y):

        ysp_li = []

        X_test_minmax = self.min_max_scaler.transform(x)
        X_test_minmax = pd.DataFrame(X_test_minmax)
        X_test_minmax.columns = self.data_cols

        X_test_minmax = X_test_minmax.drop(['pro', 'nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv'], axis=1,
                                           inplace=False)
        x_np = np.array(X_test_minmax)
        similarity = Similarity()

        index = 0
        for i in x_np:
            yt = y[index]
            index += 1

            simi = []
            inc = 0
            user_pros = []
            for clf in self.clfs:
                user_pros.append(clf.predict(i.reshape(1, -1)).tolist()[0])
            # print(user_pros)
            for j in range(len(self.x_database)):

                similar = similarity.Pearson(i, self.x_database[j]) * np.array(user_pros)
                # print(similarity.Pearson(i, self.x_database[j]))

                # similar = similarity.Cosine(i, self.x_database[j])
                # print(similarity.Cosine(i, self.x_database[j]), self.x_toolspro[j], similar)

                simi.append(similar)

            # print(similar_concat)
            pr = pd.DataFrame(np.array(simi)).max().tolist()

            mix = pr.index(max(pr))
            # print(mix, pr)
            ma = {0: 'nanosv', 1: 'picky', 2: 'sniffles', 3: 'pbsv', 4: 'cutesv'}
            ysp = ma[mix]
            ysp_li.append(ysp)
        return ysp_li


def datasets(s):
    d1 = '/Users/sherwinwang/Documents/Project/winequality-red.csv'
    d2 = '/Users/sherwinwang/Documents/Project/DryBeanDataset/Dry_Bean_Dataset.csv'
    d3 = '/Users/sherwinwang/Documents/Project/abalone/abalone.csv'
    d4 = '/Users/sherwinwang/Documents/Project/recorededPrecisionFile1.csv'

    if s == 'iris':
        reader = load_iris()

        X_train, X_test, y_train, y_test = train_test_split(reader['data'], reader['target'],
                                                            random_state=2)

    elif s == 'winequality':
        reader = pd.read_csv(d1, sep=';')

        Xdata = reader.iloc[:, :-1]
        X_train, X_test, y_train, y_test = train_test_split(Xdata, reader['quality'], test_size=0.3, random_state=6)

    elif s == 'Dry_Bean_Dataset':
        reader = pd.read_csv(d2)
        num_instances = reader.shape[0]
        pro_li = []
        for i in range(num_instances):
            pro_li.append(func1(1, 3))
        reader['pro'] = pro_li
        Xdata2 = reader.drop('Class', axis=1, inplace=False)
        # print(data_2)
        # exit()
        # Xdata2 = reader.iloc[:, :-1]
        X_train, X_test, y_train, y_test = train_test_split(Xdata2, reader['Class'], test_size=0.3, random_state=6)

    elif s == 'abalone':
        reader = pd.read_csv(d3)
        Xdata3 = reader.iloc[:, 1:]
        X_train, X_test, y_train, y_test = train_test_split(Xdata3, reader['Sex'], test_size=0.3, random_state=6)

    elif s == 'sv':
        reader = pd.read_csv(d4)
        # print(reader.describe())
        # plot all of the columns
        # reader.hist()
        # plt.show()
        import statsmodels.api as sm
        # 指定作为训练变量的列，不含目标列`admit`

        # Index([gre, gpa, prestige_2, prestige_3, prestige_4], dtype=object)

        Xdata3 = reader.iloc[:, :-1]

        X_train, X_test, y_train, y_test = train_test_split(Xdata3, reader['type'], test_size=0.3, random_state=6)

        # logit = sm.Logit(X_train['pbsv'], X_train[train_cols])
        #
        # # 拟合模型
        # result = logit.fit()
        # # 进行预测，并将预测评分存入 predict 列中
        # X_test['predict'] = result.predict(X_test[train_cols])
        #
        #
        # # 预测完成后，predict 的值是介于 [0, 1] 间的概率值
        # # 我们可以根据需要，提取预测结果
        # # 例如，假定 predict > 0.5，则表示会被录取
        # # 在这边我们检验一下上述选取结果的精确度
        # total = 0
        # hit = 0
        # for value in X_test.values:
        #     # 预测分数 predict, 是数据中的最后一列
        #     predict = value[-1]
        #     # 实际录取结果
        #     admit = int(value[-3])
        #
        #     # 假定预测概率大于0.5则表示预测被录取
        #     if predict > 0.5:
        #         total += 1
        #         # 表示预测命中
        #         if admit == 1:
        #             hit += 1
        #
        # # 输出结果
        # print('Total: %d, Hit: %d, Precision: %.2f' % (total, hit, 100.0 * hit / total))
        # exit()

    else:
        X_train, X_test, y_train, y_test, reader = [], [], [], [], None

    return X_train, X_test, y_train, y_test, reader


def modelScores(X_train, X_test, y_train, y_test, rd, n_clfs, sampleid):
    test_pre_li = []
    train_pre_li = []
    all_data_pre_li = []
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
    if sampleid == 'iris':
        all_xdata = rd['data']
        all_type = rd['target']
    elif sampleid == 'winequality':
        all_xdata = rd.iloc[:, :-1]
        all_type = rd.iloc[:, -1]
    elif sampleid == 'Dry_Bean_Dataset':
        all_xdata = rd.iloc[:, :-1]
        all_type = rd.iloc[:, -1]
    elif sampleid == 'abalone':
        all_xdata = rd.iloc[:, 1:]
        all_type = rd['Sex']
    elif sampleid == 'sv':
        all_xdata = rd.iloc[:, :-6]
        all_type = rd.iloc[:, -1]

    for i in range(n_clfs):
        clf = learn.reModel(i)
        # xtt, ytt = self.X_Y(self.train[i])
        # X_tt, y_tt = self.X_Y(self.Sn)
        # X_train['D'] = X_train['D'].astype('float')
        # print(X_train['E'])
        # exit()

        clf.fit(X_train, y_train)
        # print(clf)
        clf_li.append(clf)
        print('trainScore', clf.score(X_train, y_train))
        print('testScore', clf.score(X_test, y_test))
        if n == 0:
            test = clf.predict_proba(X_test)
            test_pre = clf.predict(X_test)

            train = clf.predict_proba(X_train)
            train_pre = clf.predict(X_train)

            all_data = clf.predict_proba(all_xdata)
            all_data_pre = clf.predict(all_xdata)
            n += 1
        else:
            test = np.hstack((test, clf.predict_proba(X_test)))
            test_pre = clf.predict(X_test)

            train = np.hstack((train, clf.predict_proba(X_train)))
            train_pre = clf.predict(X_train)

            all_data = np.hstack((all_data, clf.predict_proba(all_xdata)))
            all_data_pre = clf.predict(all_xdata)

        test_pre_li.append(test_pre)
        train_pre_li.append(train_pre)
        all_data_pre_li.append(all_data_pre)

    test_pre_li.append(y_test)
    train_pre_li.append(y_train)
    all_data_pre_li.append(all_type)
    test_pre_mat = np.mat(test_pre_li).T
    train_pre_mat = np.mat(train_pre_li).T
    all_data_pre_mat = np.mat(all_data_pre_li).T
    # print(np.mat(test_pre_li).T)
    # print(np.hstack((test,test_pre_mat)))

    # Create list of column names with the format "colN" (from 1 to N)
    col_names = ['col' + str(i) for i in np.arange(test.shape[1]) + 1]
    # Declare pandas.DataFrame object
    df = pd.DataFrame(data=test, columns=col_names)
    # df_ = df
    # df['type'] = y_test
    # print(df)
    return test, test_pre_mat, train, train_pre_mat, all_data, all_data_pre_mat


def knnRecommendatation(X_t, X_v, y_t, y_v):
    training_accuracy = []
    test_accuracy = []
    # n_neighbors取值从1到10
    neighbors_settings = range(1, 21)
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
    # print(test_accuracy)
    y2pre = rf_clf.predict(X_v)
    report0 = classification_report(y_v, y2pre)
    print('rf_clf: ', report0)

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


def recommendatation(test, test_pre_mat, tra, tra_pre_mat, n_clfs):
    n_clfs_pres = []
    # X_test1, X_validation, y_test1, y_validation = train_test_split(test, test_pre_mat, random_state=5)
    # pointRecommendataion(X_validation, X_test1, y_validation, y_test1)
    # t_type = pd.DataFrame(data=test_pre_mat).iloc[:, -1]
    X_test2, X_validation2, y_test2, y_validation2 = train_test_split(test, test_pre_mat, test_size=0.3, random_state=5)
    X_test2 = np.concatenate((tra, X_test2))
    y_test2 = np.concatenate((tra_pre_mat, y_test2))
    y_test2_t = pd.DataFrame(data=y_test2).iloc[:, -1]
    y_validation2_t = pd.DataFrame(data=y_validation2).iloc[:, -1]

    for n in range(n_clfs):
        n_clfs_pres.append(pd.DataFrame(data=y_validation2).iloc[:, n])

    t0_type = pd.DataFrame(data=y_validation2).iloc[:, -1]

    knnRecommendatation(X_test2, X_validation2, y_test2_t, y_validation2_t)
    randomforestRecommendatation(X_test2, X_validation2, y_test2_t, y_validation2_t)

    prm = pointRecommendataionClassifier(X_test2, y_test2_t)
    y2pre = prm.predict(X_validation2)
    report0 = classification_report(y_validation2_t, y2pre)
    print('pointRecommendataionClassifier: ', report0)

    for n in range(n_clfs):
        report = classification_report(t0_type, n_clfs_pres[n])
        print(str(n) + 'clf: ' + report)


def recommendatationTest(test, test_pre_mat, n_clfs):
    n_clfs_pres = []
    # X_test1, X_validation, y_test1, y_validation = train_test_split(test, test_pre_mat, random_state=5)
    # pointRecommendataion(X_validation, X_test1, y_validation, y_test1)
    # t_type = pd.DataFrame(data=test_pre_mat).iloc[:, -1]
    X_test2, X_validation2, y_test2, y_validation2 = train_test_split(test, test_pre_mat, test_size=0.3, random_state=5)
    # X_test2 = np.concatenate((tra, X_test2))
    # y_test2 = np.concatenate((tra_pre_mat, y_test2))
    y_test2_t = pd.DataFrame(data=y_test2).iloc[:, -1]
    y_validation2_t = pd.DataFrame(data=y_validation2).iloc[:, -1]

    for n in range(n_clfs):
        n_clfs_pres.append(pd.DataFrame(data=y_validation2).iloc[:, n])

    t0_type = pd.DataFrame(data=y_validation2).iloc[:, -1]

    knnRecommendatation(X_test2, X_validation2, y_test2_t, y_validation2_t)
    randomforestRecommendatation(X_test2, X_validation2, y_test2_t, y_validation2_t)

    prm = pointRecommendataionClassifier(X_test2, y_test2_t)
    y2pre = prm.predict(X_validation2)
    report0 = classification_report(y_validation2_t, y2pre)
    print('pointRecommendataionClassifier: ', report0)

    for n in range(n_clfs):
        report = classification_report(t0_type, n_clfs_pres[n])
        print(str(n) + 'clf: ' + report)

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


def CollaborativeFilteringModel(X_test2, X_validation2, y_test2_t, y_validation2_t):
    prm = pointRecommendataionClassifier9(X_test2, y_test2_t)
    y_validation2_t = np.array(y_validation2_t)
    y2pre = prm.predict(X_validation2, y_validation2_t)
    report0 = classification_report(y_validation2_t, y2pre)
    print('pointRecommendataionClassifier: ', report0)


def main():
    s = 'iris'
    s = 'winequality'
    s = 'Dry_Bean_Dataset'
    # s = 'abalone'
    s = 'sv'
    X_t, X_tt, y_t, y_tt, rd = datasets(s)
    CollaborativeFilteringModel(X_t, X_tt, y_t, y_tt)
    exit()
    n_s = 5

    tt, tt_pre_mat, tra, tra_pre_mat, all_data, all_data_pre_mat = modelScores(X_t, X_tt, y_t, y_tt, rd, n_s, s)
    # recommendatation(tt, tt_pre_mat, tra, tra_pre_mat, n_s)
    recommendatationTest(tt, tt_pre_mat, n_s)


if __name__ == '__main__':
    main()
