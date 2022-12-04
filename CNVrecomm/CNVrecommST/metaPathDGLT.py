import os
import warnings

import dgl

warnings.filterwarnings('ignore')
from sklearn import preprocessing
import pandas as pd
import numpy as np
import networkx as nx
import json
import torch
# from torch_geometric.data import Data, HeteroData
import matplotlib.pyplot as plt
from recommendation.SimilarityCalculation import Similarity
from torch_geometric.utils import to_networkx


class extractGraphData():

    def extractSampleNodes(self, df_data, sample_nodes_path):
        sample_nodes_dic = {}
        sample_nodes = df_data['sample']
        sample_nodes_dic['id'] = list(range(len(sample_nodes.tolist())))
        sample_nodes_dic['sample'] = sample_nodes.tolist()
        sample_nodes_df = pd.DataFrame(sample_nodes_dic)
        sample_nodes_df.to_csv(sample_nodes_path, index=False)

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
        df = df_data.drop(['sample', 'Stype', 'Ftype', 'Ptype', 'ScnMops', 'PcnMops', 'FcnMops', 'Sfacets', 'Pfacets',
                           'Ffacets', 'SCNVpytor', 'PCNVpytor', 'FCNVpytor', 'SCODEX', 'PCODEX', 'FCODEX', 'SexomeCopy',
                           'PexomeCopy', 'FexomeCopy', 'Scnvkit', 'Pcnvkit', 'Fcnvkit', 'Scontra', 'Pcontra', 'Fcontra'
                           ], 1)
        df = df.fillna(df.mean())
        scaler_df = zscore.fit_transform(df)
        df_score = pd.DataFrame(scaler_df, index=df.index, columns=df.columns)
        df_score['ScnMops'] = df_data['ScnMops']
        df_score['PcnMops'] = df_data['PcnMops']
        df_score['FcnMops'] = df_data['FcnMops']
        df_score['Sfacets'] = df_data['Sfacets']
        df_score['Pfacets'] = df_data['Pfacets']
        df_score['Ffacets'] = df_data['Ffacets']
        df_score['SCNVpytor'] = df_data['SCNVpytor']
        df_score['PCNVpytor'] = df_data['PCNVpytor']
        df_score['FCNVpytor'] = df_data['FCNVpytor']
        df_score['SCODEX'] = df_data['SCODEX']
        df_score['PCODEX'] = df_data['PCODEX']
        df_score['FCODEX'] = df_data['FCODEX']
        df_score['SexomeCopy'] = df_data['SexomeCopy']
        df_score['PexomeCopy'] = df_data['PexomeCopy']
        df_score['FexomeCopy'] = df_data['FexomeCopy']
        df_score['Scnvkit'] = df_data['Scnvkit']
        df_score['Pcnvkit'] = df_data['Pcnvkit']
        df_score['Fcnvkit'] = df_data['Fcnvkit']
        df_score['Scontra'] = df_data['Scontra']
        df_score['Pcontra'] = df_data['Pcontra']
        df_score['Fcontra'] = df_data['Fcontra']
        df_score['Stype'] = df_data['Stype']
        df_score['Ptype'] = df_data['Ptype']
        df_score['Ftype'] = df_data['Ftype']
        df_score['sample'] = df_data['sample']
        df_score.to_csv('F:/CNVrecommendation/exomeData/ExtractingMetaFeatures/ExtractingMetaTargetScarler.csv', index=False)

    def extractSampleNodeAttr(self, df_data, sample_nodes_attr_path):
        df_node_attr = df_data[['purity', 'shortCNV', 'middleCNV', 'largeCNV', 'deletion', 'readLen', 'readDepth',
                                'Stype', 'Ptype', 'Ftype', 'sample']]
        df_node_attr.to_csv(sample_nodes_attr_path, index=False)


    def extractNodes(self, df_data):
        sample_nodes_path = 'F:/CNVrecommendation/exomeData/ExtractingMetaFeatures/sample_nodes.csv'
        # tool_nodes_path = 'F:/CNVrecommendation/newCalling/tool_nodes.csv'

        tool_nodes_path = 'F:/CNVrecommendation/newCalling/toolNodes(new1).csv'
        tool_nodescarler_path = 'F:/CNVrecommendation/newCalling/toolNodesScarler.csv'
        sample_nodes_attr_path = 'F:/CNVrecommendation/exomeData/ExtractingMetaFeatures/sample_nodes_attr.csv'

        if not os.path.exists(sample_nodes_path):
            self.extractSampleNodes(df_data, sample_nodes_path)
        if not os.path.exists(tool_nodescarler_path):
            self.extractToolNodes(tool_nodes_path)
        if not os.path.exists(sample_nodes_attr_path):
            self.extractSampleNodeAttr(df_data, sample_nodes_attr_path)

    def dataPre(self):
        ExtractingMetaTarget = 'F:/CNVrecommendation/exomeData/ExtractingMetaFeatures/ExtractingMetaTarget.tsv'
        ExtractingMetaTargetScarler = 'F:/CNVrecommendation/exomeData/ExtractingMetaFeatures/ExtractingMetaTargetScarler.csv'
        ExtractingMetaTarget_df = pd.read_csv(ExtractingMetaTarget, encoding='gbk', low_memory=False, sep='\t')

        if not os.path.exists(ExtractingMetaTargetScarler):
            self.dfStandardScaler(ExtractingMetaTarget_df)

        ExtractingMetaTargetScarler_df = pd.read_csv(ExtractingMetaTargetScarler, encoding='gbk', low_memory=False)
        self.extractNodes(ExtractingMetaTargetScarler_df)

    def STmetaPath(self, sindex, row, ExtractingMetaTargetScarler_df, tool_nodes_df, tag):
        S_T = []
        label = []
        ttmpindex = None
        dtnodes_attr = []
        edges_attr = []
        samples_id = []
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

        hetero_graph = dgl.heterograph({('sample', 'choose', 'tool'): ([0], [0]),
                                        ('tool', 'choose-by', 'sample'): ([0], [0])})

        hetero_graph.nodes['sample'].data['feature'] = torch.tensor([ssnode_attr])
        # hetero_graph.edges['choose'].data['feature'] = torch.tensor(edges_attr, dtype=torch.float)
        hetero_graph.nodes['tool'].data['feature'] = torch.tensor([dtnodes_attr[ttmpindex]])
        label.append(ttmpindex)
        S_T.append(hetero_graph)
        samples_id.append(sindex)
        return S_T, label, samples_id


    def SSTmetaPath(self, sindex, row, ExtractingMetaTargetScarler_df, tool_nodes_df, tag):
            ttmpindex = None
            dtnodes_attr = []
            s_t_edges_attr = []
            similar_li = []
            ssw_li = []
            stw_li = []
            rsnodes_attr = []
            s_s_index = []
            s_t_index = []
            labels = []
            sst_labels = []
            S_S_T = []
            samples_id = []
            ssnode_attr = [row['purity'], row['shortCNV'], row['middleCNV'], row['largeCNV'], row['deletion'],
                           row['readLen'], row['readDepth']]
            ssnode_attr_npy =np.array(row[['purity', 'shortCNV', 'middleCNV', 'largeCNV', 'deletion', 'readLen',
                                           'readDepth']])
            rest_ExtractingMetaTargetScarler_df = ExtractingMetaTargetScarler_df.drop([sindex])
            similarity = Similarity()
            for rsindex, srow in ExtractingMetaTargetScarler_df.iterrows():
                if sindex == rsindex:
                    continue
                else:
                    for tindex, row1 in tool_nodes_df.iterrows():
                        if row[tag + 'type'] == srow[tag + 'type'] == row1['tool']:
                            ttmpindex = tindex
                            labels.append(ttmpindex)
                            rsnode_attr = srow[
                                ['purity', 'shortCNV', 'middleCNV', 'largeCNV', 'deletion', 'readLen', 'readDepth']]
                            rsnode_attr_npy = np.array(rsnode_attr)
                            similar = similarity.Pearson(ssnode_attr_npy, rsnode_attr_npy)
                            similar_li.append(similar)
                            rsnodes_attr.append(rsnode_attr)
                            s_s_index.append([sindex, rsindex])
                            edge = srow[tag + row1['tool']]
                            edge_attr = edge
                            s_t_edges_attr.append(edge_attr)
                            s_t_index.append([rsindex, tindex])
                            dtnode_attr = [row1['attractiveness'], row1['matureness']]
                            if len(ssnode_attr) - len(dtnode_attr) > 0:
                                dtnode_attr += [0 for i in range(len(ssnode_attr) - len(dtnode_attr))]
                            elif len(ssnode_attr) - len(dtnode_attr) < 0:
                                dtnode_attr += [0 for i in range(len(dtnode_attr) - len(ssnode_attr))]
                            dtnodes_attr.append(dtnode_attr)

            for i in range(len(s_s_index)):
                hetero_graph = dgl.heterograph({('sample', 'choose', 'sample'): ([0], [1]),
                                                ('sample', 'choose-by', 'sample'): ([1], [0]),
                                                ('sample', 'choose', 'tool'): ([1], [0]),
                                                ('tool', 'choose-by', 'sample'): ([0], [1])
                                                })
                hetero_graph.nodes['sample'].data['feature'] = torch.tensor([ssnode_attr, rsnodes_attr[i]])
                # hetero_graph.edges['choose'].data['feature'] = torch.tensor(edges_attr, dtype=torch.float)
                hetero_graph.nodes['tool'].data['feature'] = torch.tensor([dtnodes_attr[i]])
                sst_labels.append(labels[i])
                S_S_T.append(hetero_graph)
                samples_id.append(sindex)

            return S_S_T, sst_labels, samples_id


    def STTmetaPath(self, sindex, row, tool_nodes_df, tag):
            ttmpindex = None
            dtnodes_attr = []
            s_t_edges_attr = []
            t_t_similar_li = []
            rtnodes_attr = []
            s_t_index = []
            t_t_index = []
            labels = []
            stt_labels = []
            S_T_T = []
            samples_id = []
            ssnode_attr = [row['purity'], row['shortCNV'], row['middleCNV'], row['largeCNV'], row['deletion'],
                           row['readLen'], row['readDepth']]
            similarity = Similarity()
            # sample_id = row['sample']
            for tindex, trow in tool_nodes_df.iterrows():
                for rtindex, rtrow in tool_nodes_df.iterrows():
                    if tindex == rtindex:
                        continue
                    else:
                        if row[tag + 'type'] == rtrow['tool']:
                            ttmpindex = rtindex
                            labels.append(ttmpindex)
                            dtnode_attr = [trow['attractiveness'], trow['matureness']]
                            dtnode_attr_npy = np.array(dtnode_attr)
                            rtnode_attr = [rtrow['attractiveness'], rtrow['matureness']]
                            rtnode_attr_npy = np.array(rtnode_attr)
                            similar = similarity.Pearson(dtnode_attr_npy, rtnode_attr_npy)
                            t_t_index.append([tindex, rtindex])
                            t_t_similar_li.append(similar)
                            s_t_index.append([sindex, tindex])
                            edge = row[tag + trow['tool']]
                            edge_attr = edge
                            s_t_edges_attr.append(edge_attr)
                            if len(ssnode_attr) - len(dtnode_attr) > 0:
                                dtnode_attr += [0 for i in range(len(ssnode_attr) - len(dtnode_attr))]
                            elif len(ssnode_attr) - len(dtnode_attr) < 0:
                                dtnode_attr += [0 for i in range(len(dtnode_attr) - len(ssnode_attr))]
                            dtnodes_attr.append(dtnode_attr)
                            if len(ssnode_attr) - len(rtnode_attr) > 0:
                                rtnode_attr += [0 for i in range(len(ssnode_attr) - len(rtnode_attr))]
                            elif len(ssnode_attr) - len(rtnode_attr) < 0:
                                rtnode_attr += [0 for i in range(len(rtnode_attr) - len(ssnode_attr))]
                            rtnodes_attr.append(rtnode_attr)
            for i in range(len(s_t_index)):
                hetero_graph = dgl.heterograph({('sample', 'choose', 'tool'): ([0], [0]),
                                                ('tool', 'choose-by', 'sample'): ([0], [0]),
                                                ('tool', 'choose', 'tool'): ([0], [1]),
                                                ('tool', 'choose-by', 'tool'): ([1], [0])})
                hetero_graph.nodes['sample'].data['feature'] = torch.tensor([ssnode_attr])
                # hetero_graph.edges['choose'].data['feature'] = torch.tensor(edges_attr, dtype=torch.float)
                hetero_graph.nodes['tool'].data['feature'] = torch.tensor([dtnodes_attr[i], rtnodes_attr[i]])
                stt_labels.append(labels[i])
                S_T_T.append(hetero_graph)
                samples_id.append(sindex)
            return S_T_T, stt_labels, samples_id


    def getDataList(self, tag):
        self.dataPre()


        ExtractingMetaTargetScarler = 'F:/CNVrecommendation/exomeData/ExtractingMetaFeatures/ExtractingMetaTargetScarler.csv'
        tool_nodes_path = 'F:/CNVrecommendation/newCalling/toolNodesScarler.csv'

        meta_dgls = []
        meta_dgl_labels = []
        ExtractingMetaTargetScarler_df = pd.read_csv(ExtractingMetaTargetScarler, encoding='gbk', low_memory=False)
        tool_nodes_df = pd.read_csv(tool_nodes_path, encoding='gbk', low_memory=False)
        S_T = []
        S_T_label = []
        S_T_samples_id = []
        for sindex, srow in ExtractingMetaTargetScarler_df.iterrows():
            S_T_0, S_T_label0, S_T_samples_id0 = self.STmetaPath(sindex, srow, ExtractingMetaTargetScarler_df, tool_nodes_df, tag)
            S_T = S_T + S_T_0
            S_T_label = S_T_label + S_T_label0
            S_T_samples_id = S_T_samples_id + S_T_samples_id0

        meta_dgls = S_T
        meta_dgl_labels = S_T_label
        meta_dgl_samples = S_T_samples_id
        return meta_dgls, meta_dgl_labels, meta_dgl_samples


if __name__ == '__main__':
    b = extractGraphData()
    tag = 'S'
    S_T, label = b.getDataList(tag)
    print(S_T, label)
    exit()
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
