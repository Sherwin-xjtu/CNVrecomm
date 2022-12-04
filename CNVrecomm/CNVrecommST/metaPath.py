import os
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
import pandas as pd
import numpy as np
import networkx as nx
import json
import torch
from torch_geometric.data import Data, HeteroData
import matplotlib.pyplot as plt
from .recommendation.SimilarityCalculation import Similarity
from torch_geometric.utils import to_networkx

class extractGraphData():

    def extractSampleNodes(self, df_data, sample_nodes_path):
        sample_nodes_dic = {}
        sample_nodes = df_data['sample']
        sample_nodes_dic['id'] = list(range(len(sample_nodes.tolist())))
        sample_nodes_dic['sample'] = sample_nodes.tolist()
        sample_nodes_df = pd.DataFrame(sample_nodes_dic)
        sample_nodes_df.to_csv(sample_nodes_path, index=False, sep='\t')

    def extractToolNodes(self, tool_nodes_path):
        # tool_nodes_dic = {}
        # tools = ['cnMops', 'facets', 'CNVpytor', 'CODEX', 'exomeCopy', 'cnvkit', 'contra']
        # attrs = ['pool', 'algorithm', 'feature', 'segmentation', 'sequencing', 'year', 'language', 'citations']
        # tool_nodes_dic['id'] = list(range(len(tools)))
        # tool_nodes_dic['tool'] = tools
        # tool_nodes_dic['year'] = [2012, 2016, 2011, 2015, 2011, 2016, 2012]
        # tool_nodes_dic['citations'] = [198, 402, 691, 63, 35, 542, 158]
        # tool_nodes_dic['Q'] = [4, 4, 4, 4, 1, 4, 4]
        # tool_nodes_dic['IF'] = [19.160, 19.160, 9.438, 19.160, 0.676, 4.779, 6.931]

        # tool_nodes_df = pd.DataFrame(tool_nodes_dic)
        # tool_nodes_df.to_csv(tool_nodes_path, index=False)
        tool_nodes_df = pd.read_csv(tool_nodes_path, encoding='gbk', low_memory=False)
        # tool_nodes_df['attractiveness'] = tool_nodes_df['year'] + tool_nodes_df['citations'] + tool_nodes_df['IF']
        # tool_nodes_df['matureness'] = tool_nodes_df['Feature'] + tool_nodes_df['MC'] + tool_nodes_df['TC']
        df = tool_nodes_df.drop(['id', 'tool'], 1)
        zscore = preprocessing.MinMaxScaler()
        scaler_df = zscore.fit_transform(df)
        df_score = pd.DataFrame(scaler_df, index=df.index, columns=df.columns)
        df_score['attractiveness'] = df_score['year'] + df_score['citations'] + df_score['IF']
        df_score['matureness'] = df_score['Feature'] + df_score['MC'] + df_score['TC']
        df_score['id'] = tool_nodes_df['id']
        df_score['tool'] = tool_nodes_df['tool']
        df_score.to_csv('F:/CNVrecommendation/newCalling/toolNodesScarler.csv', index=False)

    def dfStandardScaler(self, df_data):
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

    def extractSampleNodeAttr(self, df_data, sample_nodes_attr_path):
        df_node_attr = df_data[['purity', 'shortCNV', 'middleCNV', 'largeCNV', 'deletion', 'readLen', 'readDepth',
                                'Stype', 'Ftype', 'Ptype', 'sample']]

        df_node_attr.to_csv(sample_nodes_attr_path, index=False)

        # for index, row in df_data.iterrows():
        #     print(row['purity'])
        #
        #     exit()

    def extractNodes(self, df_data):
        sample_nodes_path = 'F:/CNVrecommendation/newCalling/sample_nodes.csv'
        # tool_nodes_path = 'F:/CNVrecommendation/newCalling/tool_nodes.csv'

        tool_nodes_path = 'F:/CNVrecommendation/newCalling/toolNodes(new1).csv'
        tool_nodescarler_path = 'F:/CNVrecommendation/newCalling/toolNodesScarler.csv'
        sample_nodes_attr_path = 'F:/CNVrecommendation/newCalling/sample_nodes_attr.csv'

        if not os.path.exists(sample_nodes_path):
            self.extractSampleNodes(df_data, sample_nodes_path)
        if not os.path.exists(tool_nodescarler_path):
            self.extractToolNodes(tool_nodes_path)
        if not os.path.exists(sample_nodes_attr_path):
            self.extractSampleNodeAttr(df_data, sample_nodes_attr_path)


    def dataPre(self):
        ExtractingMetaTarget = 'F:/CNVrecommendation/newCalling/ExtractingMetaTarget.tsv'
        ExtractingMetaTargetScarler = 'F:/CNVrecommendation/newCalling/ExtractingMetaTargetScarler.csv'
        ExtractingMetaTarget_df = pd.read_csv(ExtractingMetaTarget, encoding='gbk', low_memory=False, sep='\t')

        if not os.path.exists(ExtractingMetaTargetScarler):
            self.dfStandardScaler(ExtractingMetaTarget_df)

        ExtractingMetaTargetScarler_df = pd.read_csv(ExtractingMetaTargetScarler, encoding='gbk', low_memory=False)
        self.extractNodes(ExtractingMetaTargetScarler_df)

    def STmetaPath(self,sindex, row, ExtractingMetaTargetScarler_df, tool_nodes_df, tag):
        S_T = []
        ttmpindex = None
        dtnodes_attr = []
        edges_attr = []
        ssnode_attr = [row['purity'], row['shortCNV'], row['middleCNV'], row['largeCNV'], row['deletion'],
                       row['readLen'], row['readDepth']]

        for tindex, row1 in tool_nodes_df.iterrows():
            edge = row[tag + row1['tool']]
            edge_attr = [edge]
            edges_attr.append(edge_attr)
            if row[tag + 'type'] == row1['tool']:
                ttmpindex = tindex

            dtnode_attr = [row1['attractiveness'], row1['matureness']]
            if len(ssnode_attr) - len(dtnode_attr) > 0:
                dtnode_attr += [0 for i in range(len(ssnode_attr) - len(dtnode_attr))]
            elif len(ssnode_attr) - len(dtnode_attr) < 0:
                dtnode_attr += [0 for i in range(len(dtnode_attr) - len(ssnode_attr))]
            dtnodes_attr.append(dtnode_attr)

        data = HeteroData()
        # 初始化结点特征
        data['sample'].x = torch.tensor([ssnode_attr])
        data['tool'].x = torch.tensor(dtnodes_attr)

        # 初始化边索引
        data['sample', 'choose', 'tool'].edge_index = torch.tensor([[0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5, 6]], dtype=torch.long)
        # 初始化边特征
        data['sample', 'choose', 'tool'].edge_attr = torch.tensor(edges_attr, dtype=torch.float)
        data['label'] = torch.tensor(ttmpindex, dtype=torch.long)
        S_T_data = data.to_homogeneous()
        S_T.append(S_T_data)

        return S_T


    def SSTmetaPath(self,ssindex, row, ExtractingMetaTargetScarler_df, tool_nodes_df, tag):
        S_T = []
        ttmpindex = None
        dtnodes_attr = []
        s_t_edges_attr = []
        similar_li = []
        ssw_li = []
        stw_li = []
        rsnodes_attr = []
        s_s_index = []
        s_t_index = []
        ssnode_attr = [row['purity'], row['shortCNV'], row['middleCNV'], row['largeCNV'], row['deletion'],
                       row['readLen'], row['readDepth']]
        ssnode_attr_npy =np.array(row[['purity', 'shortCNV', 'middleCNV', 'largeCNV', 'deletion', 'readLen',
                                       'readDepth']])
        rest_ExtractingMetaTargetScarler_df = ExtractingMetaTargetScarler_df.drop([ssindex])
        similarity = Similarity()
        for rsindex, srow in rest_ExtractingMetaTargetScarler_df.iterrows():
            rsnode_attr = srow[['purity', 'shortCNV', 'middleCNV', 'largeCNV', 'deletion', 'readLen', 'readDepth']]
            rsnode_attr_npy = np.array(rsnode_attr)
            similar = similarity.Pearson(ssnode_attr_npy, rsnode_attr_npy)
            similar_li.append(similar)
            rsnodes_attr.append(rsnode_attr)
            s_s_index.append([ssindex, rsindex])

            for tindex, row1 in tool_nodes_df.iterrows():
                edge = srow[tag + row1['tool']]
                edge_attr = edge
                s_t_edges_attr.append(edge_attr)
                s_t_index.append([rsindex, tindex])
                if ttmpindex == None:
                    if row[tag + 'type'] == row1['tool']:
                        ttmpindex = tindex

                dtnode_attr = [row1['attractiveness'], row1['matureness']]
                if len(ssnode_attr) - len(dtnode_attr) > 0:
                    dtnode_attr += [0 for i in range(len(ssnode_attr) - len(dtnode_attr))]
                elif len(ssnode_attr) - len(dtnode_attr) < 0:
                    dtnode_attr += [0 for i in range(len(dtnode_attr) - len(ssnode_attr))]
                dtnodes_attr.append(dtnode_attr)

        #Random Wandering
        sumSSW = sum(similar_li)
        for ssw in similar_li:
            pro_ss = ssw / sumSSW
            ssw_li.append(pro_ss)

        sumSTW = sum(s_t_edges_attr)
        for stw in s_t_edges_attr:
            pro_st = stw / sumSTW
            stw_li.append(pro_st)

        for i in ssw_li:
            cor_p =


        data = HeteroData()
        # 初始化结点特征
        data['sample'].x = torch.tensor([ssnode_attr])
        data['tool'].x = torch.tensor(dtnodes_attr)

        # 初始化边索引
        data['sample', 'choose', 'tool'].edge_index = torch.tensor([[0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5, 6]], dtype=torch.long)
        # 初始化边特征
        data['sample', 'choose', 'tool'].edge_attr = torch.tensor(edges_attr, dtype=torch.float)
        data['label'] = torch.tensor(ttmpindex, dtype=torch.long)
        S_T_data = data.to_homogeneous()
        S_T.append(S_T_data)

        return S_T



    def getDataList(self, tag):
        self.dataPre()
        ExtractingMetaTargetScarler = 'F:/CNVrecommendation/newCalling/ExtractingMetaTargetScarler.csv'
        tool_nodes_path = 'F:/CNVrecommendation/newCalling/toolNodesScarler.csv'


        ExtractingMetaTargetScarler_df = pd.read_csv(ExtractingMetaTargetScarler, encoding='gbk', low_memory=False)
        tool_nodes_df = pd.read_csv(tool_nodes_path, encoding='gbk', low_memory=False)
        S_T = []
        for sindex, srow in ExtractingMetaTargetScarler_df.iterrows():
            S_T_0 = self.STmetaPath(sindex, srow, ExtractingMetaTargetScarler_df, tool_nodes_df, tag)
            S_T = S_T + S_T_0

        return S_T


if __name__ == '__main__':
    b = extractGraphData()
    S_T = b.getDataList()
    from torch_geometric.loader import DataLoader

    loader = DataLoader(S_T, batch_size=32, shuffle=True)
    for i in loader:
        print(i)
        exit()
    print(S_T[0])
    print(S_T[0].x_dict)
    print(S_T[0].edge_index_dict)
    print(S_T[0].node_types)
    print(S_T[0].edge_types)
    print(S_T[0].edge_attr())
    print(S_T[0].label)
    # 异构转同构
    homogeneous_data = S_T[0].to_homogeneous()
    print(homogeneous_data)
    plt.subplot(224)
    nx.draw(to_networkx(homogeneous_data), with_labels=True)
    plt.show()
