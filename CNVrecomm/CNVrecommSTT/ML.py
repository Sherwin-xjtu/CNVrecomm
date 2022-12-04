#!/usr/bin/python
# coding=utf-8
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
# import xgboost as xgb
import numpy as np
from numpy import *
# from system.Tri_training01.Bagging import Bagging
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import neighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import mixture
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


def ML(X,Y):
    dt_clf = tree.DecisionTreeClassifier()
    rf_clf = RandomForestClassifier()
    svm_clf = SVC()
    nbrs_clf = neighbors.KNeighborsClassifier()
    nb_clf = GaussianNB()
    mls = [dt_clf, rf_clf, svm_clf, nbrs_clf, nb_clf]
    MLS = []
    for ml in mls:
        ml.fit(X,Y)
        scores = cross_val_score(ml, X,Y, cv=5)  # cv为迭代次数。
        MLS.append(ml)
        # print(scores)
    return MLS

""" 获取X,Y """

def X_Y(S, t):
    train = S
    y = train[t]

    x = train.drop([t, 'sample'], 1)
    # if 'chrom' in train.columns:
    #     train.drop(['chrom'], 1)
    # x_train = x.values
    # y_train = y.values.ravel()
    return x, y

def pre_data(data_pd, target):
    # data_pd = pd.read_csv(data_path)
    data_pd = data_pd.fillna(data_pd.mean())
    # data_pd.to_csv('fillna.csv')
    # print(data_pd)
    xt, yt = X_Y(data_pd, target)
    # X_train, X_validation, y_train, y_validation = train_test_split(xt, yt, test_size=0.1, random_state=2, stratify=yt)
    return xt, yt

def check(sample_df, c):

    if sample_df['clf' + str(c)].sum() < 42:
        return 0
    else:
        return 1

if __name__ == '__main__':
    all_data = 'F:/CNVrecommendation/newCalling/ExtractingMetaTargetScarlerML.csv'
    # test_data = 'E:/科研论文/HRD/6.24/frontiers in genetics/revision/test2.csv'
    # test_df = pd.read_csv(test_data)
    #
    # X_train, y_train = pre_data(train_data)
    # X_test, y_test = pre_data(test_data)
    all_data_df = pd.read_csv(all_data)
    Sdf = all_data_df.drop(['Ftype', 'Ptype'], 1)
    Sdf_train, Sdf_test = train_test_split(Sdf, test_size=0.2, stratify=Sdf['Stype'])
    Fdf = all_data_df.drop(['Stype', 'Ptype'], 1)
    Fdf_train, Fdf_test = train_test_split(Fdf, test_size=0.2, stratify=Fdf['Ftype'])
    Pdf = all_data_df.drop(['Ftype', 'Stype'], 1)
    Pdf_train, Pdf_test = train_test_split(Pdf, test_size=0.2, stratify=Pdf['Ptype'])
    # df_train, df_test = train_test_split(all_data_df, test_size=0.2)
    X_train, y_train = pre_data(Sdf_train, 'Stype')
    X_test, y_test = pre_data(Sdf_test, 'Stype')
    mcls = ML(X_train, y_train)
    i = 0
    for ml in mcls:
        # target = ['class 0', 'class 1']
        y_pred = ml.predict(X_test)
        scores = cross_val_score(ml, X_test, y_test, cv=5)  # cv为迭代次数
        print(mean(scores))
        report = classification_report(y_test, y_pred)
        i +=1
        Sdf_test['clf' + str(i)] = y_pred
        print("分类器{0}的性能报告：\n {1}".format(i, report))
    Sdf_test.to_csv('F:/CNVrecommendation/newCalling/clfsResults2.csv', index=False )

    MMGR_results = 'F:/CNVrecommendation/newCalling/results/ExtractingMetaTargetScarler_Spre.csv'
    MMGR_results_df = pd.read_csv(MMGR_results)
    y_label = MMGR_results_df['label']
    y_pred = MMGR_results_df['MMGR_pre']
    report0 = classification_report(y_label, y_pred)

    print("分类器{0}的性能报告：\n {1}".format('MMGR', report0))

    from sklearn.metrics import roc_auc_score



    # samples_id = set(Sdf_test['sample'].tolist())
    # ids = []
    # label = []
    # clf1 = []
    # clf2 = []
    # clf3 = []
    # clf4 = []
    # clf5 = []
    #
    # for id in samples_id:
    #     ids.append(id)
    #     pred = []
    #     sample_df = Sdf_test[Sdf_test['sample']==id]
    #     label.append(sample_df['Stype'])
    #     # for c in range(1, len(mcls)+1):
    #     #     pred.append(check(sample_df, c))
    #     clf1.append(pred[0])
    #     clf2.append(pred[1])
    #     clf3.append(pred[2])
    #     clf4.append(pred[3])
    #     clf5.append(pred[4])
    # preds_dic = {'ids':ids, 'label':label, 'clf1':clf1, 'clf2':clf2, 'clf3':clf3, 'clf4':clf4, 'clf5':clf5}
    # preds_df = pd.DataFrame(preds_dic)
    # out_path = 'F:/CNVrecommendation/newCalling/clfsPreds2.csv'
    # preds_df.to_csv(out_path, index=False)
    # print(preds_df)


