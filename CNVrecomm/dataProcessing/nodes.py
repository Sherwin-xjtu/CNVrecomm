import os
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
import pandas as pd
import networkx as nx
import json


def extractSampleNodes(df_data, sample_nodes_path):
    sample_nodes_dic = {}
    sample_nodes = df_data['sample']
    sample_nodes_dic['id'] = list(range(len(sample_nodes.tolist())))
    sample_nodes_dic['sample'] = sample_nodes.tolist()
    sample_nodes_df = pd.DataFrame(sample_nodes_dic)
    sample_nodes_df.to_csv(sample_nodes_path, index=False, sep='\t')


def extractToolNodes(tool_nodes_path):
    tool_nodes_dic = {}
    tools = ['cnMops', 'facets', 'CNVpytor', 'CODEX', 'exomeCopy', 'cnvkit', 'contra']
    # attrs = ['pool', 'algorithm', 'feature', 'segmentation', 'sequencing', 'year', 'language', 'citations']
    tool_nodes_dic['id'] = list(range(len(tools)))
    tool_nodes_dic['tool'] = tools
    tool_nodes_dic['year'] = [2012, 2016, 2011, 2015, 2011, 2016, 2012]
    tool_nodes_dic['citations'] = [198, 402, 691, 63, 35, 542, 158]
    tool_nodes_dic['Q'] = [4, 4, 4, 4, 1, 4, 4]
    tool_nodes_dic['IF'] = [19.160, 19.160, 9.438, 19.160, 0.676, 4.779, 6.931]


    tool_nodes_df = pd.DataFrame(tool_nodes_dic)
    tool_nodes_df.to_csv(tool_nodes_path, index=False)
    df = tool_nodes_df.drop(['id', 'tool'], 1)
    zscore = preprocessing.MinMaxScaler()
    scaler_df = zscore.fit_transform(df)
    df_score = pd.DataFrame(scaler_df, index=df.index, columns=df.columns)
    df_score['id'] = tool_nodes_df['id']
    df_score['tool'] = tool_nodes_df['tool']
    df_score.to_csv('F:/CNVrecommendation/newCalling/toolNodesScarler.csv', index=False)


def extractSampleNodeAttr(df_data, sample_nodes_attr_path):
    df_node_attr = df_data[['purity', 'shortCNV', 'middleCNV', 'largeCNV', 'deletion', 'readLen', 'readDepth',
                            'Stype', 'Ftype', 'Ptype', 'sample']]

    df_node_attr.to_csv(sample_nodes_attr_path, index=False)
    
    for index, row in df_data.iterrows():
        print(row['purity'])
        print(row[:7])

        exit()



def extractNodes(df_data):
    sample_nodes_path = 'F:/CNVrecommendation/newCalling/sample_nodes.csv'
    tool_nodes_path = 'F:/CNVrecommendation/newCalling/tool_nodes.csv'
    sample_nodes_attr_path = 'F:/CNVrecommendation/newCalling/sample_nodes_attr.csv'

    if  not os.path.exists(sample_nodes_path):
        extractSampleNodes(df_data, sample_nodes_path)
    if not os.path.exists(tool_nodes_path):
        extractToolNodes(tool_nodes_path)

    extractSampleNodeAttr(df_data, sample_nodes_attr_path)


def generateNodes(s, len):
    nodes_li = []
    for i in range(len):
        nodes_li.append(s + str(i))
    return nodes_li




def dfStandardScaler(df_data):
    zscore = preprocessing.MinMaxScaler()
    df = df_data.drop(['sample', 'Stype', 'Ftype', 'Ptype'], 1)
    df = df.fillna(df.mean())
    scaler_df = zscore.fit_transform(df)
    df_score = pd.DataFrame(scaler_df, index=df.index, columns=df.columns)
    df_score['Stype'] = df_data['Stype']
    df_score['Ftype'] = df_data['Ftype']
    df_score['Ptype'] = df_data['Ptype']
    df_score['sample'] = df_data['sample']
    df_score.to_csv('F:/CNVrecommendation/newCalling/ExtractingMetaTargetScarler.csv', index=False)

def mian(ExtractingMetaTarget_df,ExtractingMetaTargetScarler):

    if not os.path.exists(ExtractingMetaTargetScarler):
        dfStandardScaler(ExtractingMetaTarget_df)

    ExtractingMetaTargetScarler_df = pd.read_csv(ExtractingMetaTargetScarler, encoding='gbk', low_memory=False)
    extractNodes(ExtractingMetaTargetScarler_df)

if __name__ == '__main__':
    ExtractingMetaTarget = 'F:/CNVrecommendation/newCalling/ExtractingMetaTarget.tsv'
    ExtractingMetaTargetScarler = 'F:/CNVrecommendation/newCalling/ExtractingMetaTargetScarler.csv'
    ExtractingMetaTarget_df= pd.read_csv(ExtractingMetaTarget, encoding='gbk', low_memory=False, sep='\t')
    print(len(ExtractingMetaTarget_df) -1)

    mian(ExtractingMetaTarget_df, ExtractingMetaTargetScarler)
