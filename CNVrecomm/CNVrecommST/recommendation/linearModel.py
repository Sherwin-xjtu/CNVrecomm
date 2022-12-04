import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import matplotlib
from sklearn import metrics

scatter_matrix


def get_data(file_name):
    data = pd.read_csv(file_name)
    X_parameter = []
    Y_parameter = []
    for single_square_feet, single_price_value in zip(data['square_feet'], data['price']):
        X_parameter.append([float(single_square_feet)])
        Y_parameter.append(float(single_price_value))
    return X_parameter, Y_parameter


def linear_model_main(X_parameters, xt_parameters, train_cols):
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters[train_cols], X_parameters['pbsv'])
    predict_outcome = regr.predict(xt_parameters[train_cols])


    predictions = {}
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_
    predictions['predicted_value'] = predict_outcome

    return predictions


def show_linear_line(X_parameters,X_test,train_cols):
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters[train_cols],X_parameters['pbsv'])
    y_test = X_test['pbsv']
    y_pred = regr.predict(X_test[train_cols])

    MSE = metrics.mean_squared_error(y_test, y_pred)

    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    print('MSE:', MSE)

    print('RMSE:', RMSE)

    # plt.scatter(X_parameters, Y_parameters, color='blue')
    # plt.yticks(())
    # plt.bar(regr.predict(X_test[train_cols]), X_parameters['pbsv'], color='red')

    # plt.scatter(regr.predict(X_test[train_cols]), X_test['pbsv'], color='blue')
    # plt.plot(regr.predict(X_test[train_cols]), X_test['pbsv'], color='red')


    plt.figure(1)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    plt.plot(range(len(y_test)), y_test, label='real test type')
    plt.plot(range(len(y_test)), y_pred, label='pre test type')
    plt.figure(2)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    plt.scatter(y_test, y_pred)

    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')

    plt.xlabel('real type')

    plt.ylabel('pre type')

    # plt.xticks(())

    plt.show()


if __name__ == "__main__":
    d4 = '/Users/sherwinwang/Documents/Project/recorededF1scoreFile.csv'
    # show_linear_line(X,Y)
    reader = pd.read_csv(d4)
    # print(reader.describe())
    # plot all of the columns
    # reader.hist()
    # plt.show()
    import statsmodels.api as sm

    # 指定作为训练变量的列，不含目标列`admit`
    train_cols = reader.columns[:-7]

    # font = {
    #     'family': 'SimHei'
    # }
    # matplotlib.rc('font', **font)
    #
    # scatter_matrix(reader[train_cols],figsize = (10, 10), diagonal = 'kid')
    # plt.show()
    # exit()


    # Index([gre, gpa, prestige_2, prestige_3, prestige_4], dtype=object)

    Xdata3 = reader.iloc[:, :-1]

    X_train, X_test, y_train, y_test = train_test_split(Xdata3, reader['type'], test_size=0.3)

    # logit = sm.Logit(X_train['pbsv'], X_train[train_cols])

    # 拟合模型
    # result = logit.fit()
    # 进行预测，并将预测评分存入 predict 列中
    # X_test['predict'] = result.predict(X_test[train_cols])

    # predictvalue = 700
    show_linear_line(X_train, X_test,train_cols)
    exit()
    result = linear_model_main(X_train, X_test, train_cols)
    print("Intercept value ", result['intercept'])
    print("coefficient", result['coefficient'])
    print("Predicted value: ", result['predicted_value'])