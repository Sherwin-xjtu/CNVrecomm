# a stacked bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import io
from pylab import *

def histogramPlot(tag, data_df):
    result = data_df.groupby(data_df.index).agg(high=(tag + '_mean', 'mean'), low=(tag + '_std', 'mean'))
    plt.bar(result.index, result['high'], color='y', label=tag + '_mean')
    plt.bar(result.index, result['low'], color='c', label=tag + '_std')
    plt.xlim(0, len(data_df['sampleID']))
    # mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
    # 在右侧显示图例
    plt.legend(loc="upper right")
    plt.xlabel('sampleID')
    plt.ylabel('rate')
    plt.title("meta-target-" + tag)
    plt.savefig('F:/CNVrecommendation/newCalling/NecessityExperimentResults/'+ tag + '_histogram.tif', dpi=300)
    plt.show()

def foldPlot(tag, data_df):
    x_axis_data = data_df['sampleID']
    y_axis_data = data_df[tag + 'CV']
    # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    plt.plot(x_axis_data, y_axis_data, label='cv', linestyle='-',
             linewidth=1,  # 折线宽度
             color='g',  # 折线颜色
             marker='o',  # 折线图中添加圆点
             markersize=3,  # 点的大小
             markeredgecolor='r',  # 点的边框色
             markerfacecolor='r',  # 点的填充色
             )

    # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc="upper right")
    plt.xlabel('sampleID')
    plt.ylabel('rate')
    plt.xlim(0, len(data_df['sampleID']))
    plt.ylim(0, max(y_axis_data) + 0.1)
    plt.title("meta-target-" + tag)
    plt.savefig('F:/CNVrecommendation/newCalling/NecessityExperimentResults/'+ tag + '_fold.tif', dpi=300)
    plt.show()

if __name__ == '__main__':
    data = 'F:/CNVrecommendation/newCalling/NecessityExperimentResults/NecessityExperiment.csv'
    data_df = pd.read_csv(data, encoding='gbk', low_memory=False)
    tags = ['sensitivity', 'precision', 'f1_score']
    for tag in tags:
        histogramPlot(tag, data_df)
        foldPlot(tag, data_df)




