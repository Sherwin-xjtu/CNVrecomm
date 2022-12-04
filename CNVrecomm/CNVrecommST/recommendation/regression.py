#!/usr/bin/python
# coding=utf-8
import warnings

import matplotlib.pyplot as mp
import pandas as pd
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
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, RANSACRegressor, ARDRegression, HuberRegressor, TheilSenRegressor, \
    SGDRegressor, PassiveAggressiveRegressor, Lasso, ElasticNet, Ridge, BayesianRidge, Lars
warnings.filterwarnings('ignore')
# 训练集
y = [.27, .16, .06, .036, .044, .04, .022, .017, .022, .014, .017, .02, .019, .017, .011, .01, .03, .05, .066, .09]
ly, n = len(y), 100
x = [[i / ly] for i in range(ly)]
# 待测集
w = [[i / n] for i in range(n)]


# # x轴范围
# max_x = 1
# x, w = [[i[0]*max_x] for i in x], [[i[0]*max_x] for i in w]

def modeling(models):
    for i in range(len(models)):
        print(models[i].__name__)
        # 建模、拟合
        model = models[i]()
        model.fit(x, y)
        # 预测
        z = model.predict(w)
        # 可视化
        mp.subplot(3, 4, i + 1)
        mp.title(models[i].__name__, size=10)
        mp.xticks(())
        mp.yticks(())
        mp.scatter(x, y, s=11, color='g')
        mp.scatter(w, z, s=1, color='r')
    mp.show()


def modelingToevaluating(models, datasets, tool):

    for i in range(len(models)):
        X_parameters, X_test, train_cols = datasets
        # print(X_parameters['A'].max())
        # print(X_parameters.loc[X_parameters['A'].idxmax()])
        # print(type(X_parameters.loc[X_parameters['A'].idxmax()]['nanosv']))
        # exit()
        print(models[i].__name__)
        print(type(models[i].__name__))
        # 建模、拟合
        regr = models[i]()
        regr.fit(X_parameters[train_cols], X_parameters[tool])
        y_test = X_test[tool]
        y_pred = regr.predict(X_test[train_cols])
        try:
            MSE = metrics.mean_squared_error(y_test, y_pred)

            RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            print('MSE:', MSE)

            print('RMSE:', RMSE)
            print('Variance score: %.2f' % r2)
        except:
            print('None')

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

if __name__ == "__main__":
    d4 = '/Users/sherwinwang/Documents/Project/recorededPrecisionFile1.csv'
    reader = pd.read_csv(d4)

    # 指定作为训练变量的列，不含目标列`admit`
    train_cols = reader.columns[:-7]
    tolnas = ['pro', 'nanosv', 'picky', 'sniffles', 'pbsv', 'cutesv']
    toolname = 'nanosv'

    Xdata3 = reader.iloc[:, :-1]

    X_train, X_test, y_train, y_test = train_test_split(Xdata3, reader['type'], test_size=0.3)

    models1 = [KNeighborsRegressor, RadiusNeighborsRegressor,GradientBoostingRegressor, AdaBoostRegressor,
              RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor,
              # VotingRegressor,
              DecisionTreeRegressor, ExtraTreeRegressor,SVR, MLPRegressor, GaussianProcessRegressor]

    datasets = [X_train, X_test,train_cols]
    modelingToevaluating(models1, datasets, toolname)

    models2 = [LinearRegression, RANSACRegressor, ARDRegression, HuberRegressor, TheilSenRegressor, SGDRegressor,
               PassiveAggressiveRegressor, Lasso, ElasticNet, Ridge, BayesianRidge, Lars]
    modelingToevaluating(models2, datasets, toolname)

# modeling([
#     KNeighborsRegressor, RadiusNeighborsRegressor,
#     GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor,
#     # VotingRegressor,
#     DecisionTreeRegressor, ExtraTreeRegressor,
#     SVR, MLPRegressor, GaussianProcessRegressor
# ])
#
# modeling([
#     LinearRegression, RANSACRegressor, ARDRegression, HuberRegressor, TheilSenRegressor,
#     SGDRegressor, PassiveAggressiveRegressor, Lasso, ElasticNet, Ridge, BayesianRidge, Lars
# ])
