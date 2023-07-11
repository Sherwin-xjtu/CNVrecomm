#!/usr/bin/python
# coding=utf-8

import warnings

warnings.filterwarnings('ignore')
from scipy.spatial.distance import pdist
import numpy as np
from scipy.stats import pearsonr


class Similarity:

    def __init__(self):
        pass

    # 欧式距离
    def EuclideanDistance(self, x, y):
        x = np.mat(x)
        # y = np.mat([1, 2, 3, 3, 2, 1])
        y = np.mat(y)
        # np.linalg.norm 用于范数计算，默认是二范数，相当于平方和开根号
        return 1.0 / (1.0 + np.linalg.norm(x - y))


    # print('EuclideanDistance:', EuclideanDistance(x, y))

    # 余弦相似度
    def Cosine(self, x, y):
        x = np.mat(x)
        # y = np.mat([1, 2, 3, 3, 2, 1])
        y = np.mat(y)
        sumData = x * y.T  # 若列为向量则为 x.T * y
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        # print(0.5 * (sumData / denom))
        # 归一化
        return (0.5 + 0.5 * (sumData / denom)).A[0][0]

    # print(Cosine(x, y))
    # print('Cosine:', Cosine(x, y))

    # 皮尔逊相似度
    def Pearson(self, x, y):
        x = np.mat(x)
        # y = np.mat([1, 2, 3, 3, 2, 1])
        y = np.mat(y)
        # 皮尔逊相关系数的取值范围(-1 ~ 1),0.5 + 0.5 * result 归一化(0 ~ 1)
        # print(np.corrcoef(x, y, rowvar=0)[0][1])

        return 0.5 + 0.5 * np.corrcoef(x.astype(float), y.astype(float), rowvar=0)[0][1]

    # print('Pearson:', Pearson(x, y))

    # 修正余弦相似度
    # 修正cosine 减去的是对item i打过分的每个user u，其打分的均值

    def AdjustedCosine(self, x, y, avg):
        x = np.mat(x)
        # y = np.mat([1, 2, 3, 3, 2, 1])
        y = np.mat(y)
        sumData = (x - avg) * (y - avg).T  # 若列为向量则为 x.T * y
        denom = np.linalg.norm(x - avg) * np.linalg.norm(y - avg)
        # print((sumData / denom))
        return 0.5 + 0.5 * (sumData / denom)


    # print('AdjustedCosine:', AdjustedCosine(x, y, avg).A[0][0])

    # 汉明距离
    def hammingDistance(self, x, y):
        x = np.mat(x)
        # y = np.mat([1, 2, 3, 3, 2, 1])
        y = np.mat(y)
        distanceArr = x - y
        print(distanceArr)
        print(np.sum(distanceArr))
        return np.sum(distanceArr == 0)  # 若列为向量则为 shape[0]

    # print('hammingDistance', hammingDistance(x, y))

    # 曼哈顿距离(Manhattan Distance)
    def Manhattan(self, x, y):
        x = np.mat(x)
        # y = np.mat([1, 2, 3, 3, 2, 1])
        y = np.mat(y)
        return np.sum(np.abs(x - y))

    # print('Manhattan', Manhattan(x, y))

    def ChebyshevDistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X, 'chebyshev')
        return d2

    def MinkowskiDistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X, 'minkowski')
        return d2

    def EuclideanDistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X)
        return d2

    def ManhattanDistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X, 'cityblock')
        return d2

    def StandardizedEuclideandistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X, 'seuclidean')
        return d2

    def MahalanobisDistance(self, x, y):
        X = np.vstack([x, y])
        XT = X.T
        d2 = pdist(XT, 'mahalanobis')
        return d2

    def BraycurtisDistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X, 'braycurtis')
        return d2

    def CanberraDistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X, 'canberra')
        return d2

    def CorrelationDistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X, 'correlation')
        return d2

    def CosineDistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X, 'cosine')
        return d2

    def DiceDistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X, 'dice')
        return d2

    def HammingDistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X, 'hamming')
        return d2

    def JaccardDistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X, 'jaccard')
        return d2

    def JensenshannonDistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X, 'jensenshannon')
        return d2

    def KulsinskiDistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X, 'kulsinski')
        return d2

    def MatchingDistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X, 'matching')
        return d2

    def RogerstanimotoDistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X, 'rogerstanimoto')
        return d2

    def RussellraoDistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X, 'russellrao')
        return d2

    def SokalmichenerDistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X, 'sokalmichener')
        return d2

    def SokalsneathDistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X, 'sokalsneath')
        return d2

    def SqeuclideanDistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X, 'sqeuclidean')
        return d2

    def YuleDistance(self, x, y):
        X = np.vstack([x, y])
        d2 = pdist(X, 'yule')
        return d2

    def Pearsonr(self, x, y):
        d2 = pearsonr(x, y)
        return d2[0]

if __name__ == '__main__':


    x = [1, 4, 0]
    # y = np.mat([1, 2, 3, 3, 2, 1])
    y = [8, 9, 2]
    dataC = np.mat([3, 5, 1])
    data = np.vstack([x, y, dataC])
    avg = np.mean(data, axis=0)  #
    Similarity = Similarity()
    print('ChebyshevDistance:', Similarity.ChebyshevDistance(x, y))
    print('MinkowskiDistance:', Similarity.MinkowskiDistance(x, y)[0])
    print('EuclideanDistance:', Similarity.EuclideanDistance(x, y)[0])
    print('ManhattanDistance:', Similarity.ManhattanDistance(x, y)[0])
    print('StandardizedEuclideandistance:', Similarity.StandardizedEuclideandistance(x, y)[0])
    print('MahalanobisDistance:', Similarity.MahalanobisDistance(x, y))
    print('BraycurtisDistance:', Similarity.BraycurtisDistance(x, y)[0])
    print('CanberraDistance:', Similarity.CanberraDistance(x, y)[0])
    print('CorrelationDistance:', Similarity.CorrelationDistance(x, y)[0])
    print('CosineDistance:', Similarity.CosineDistance(x, y)[0])
    print('DiceDistance:', Similarity.DiceDistance(x, y)[0])
    print('HammingDistance:', Similarity.HammingDistance(x, y)[0])
    print('JaccardDistance:', Similarity.JaccardDistance(x, y)[0])
    print('JensenshannonDistance:', Similarity.JensenshannonDistance(x, y)[0])
    print('KulsinskiDistance:', Similarity.KulsinskiDistance(x, y)[0])
    print('MatchingDistance:', Similarity.MatchingDistance(x, y)[0])
    print('RogerstanimotoDistance:', Similarity.RogerstanimotoDistance(x, y)[0])
    print('RussellraoDistance:', Similarity.RussellraoDistance(x, y)[0])
    print('SokalmichenerDistance:', Similarity.SokalmichenerDistance(x, y)[0])
    print('SokalsneathDistance:', Similarity.SokalsneathDistance(x, y)[0])
    print('SqeuclideanDistance:', Similarity.SqeuclideanDistance(x, y)[0])
    print('MahalanobisDistance:', Similarity.MahalanobisDistance(x, y))
    print('YuleDistance:', Similarity.YuleDistance(x, y))
    print('Pearsonr:', Similarity.Pearsonr(x, y))
