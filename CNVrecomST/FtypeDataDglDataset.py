"""A mini synthetic dataset for graph classification benchmark."""
import math, os

import dgl
import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
from CNVrecomST.metaPathDGL import extractGraphData

__all__ = ['FtypeDataDglDataset']

from dgl import load_graphs, save_graphs

from dgl.data import DGLDataset
from torch.utils.data import DataLoader


class FtypeDataDglDataset(DGLDataset):
    def __init__(self, seed=0,save_graph=True, save_mydata='', sampleid=''):

        self.seed = seed
        self.save_graph = save_graph
        self.save_mydata = save_mydata
        self.sampleid = sampleid
        super(FtypeDataDglDataset, self).__init__(name="FtypeDataDglDataset")

    def process(self):

        b = extractGraphData(self.save_mydata)
        tag = 'F'
        graphs_list, label_list, samples_list = b.getDataList(tag)

        self.graphs = graphs_list
        self.labels = torch.tensor(label_list)
        self.samples = torch.tensor(samples_list)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the idx-th sample.

        Parameters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (:class:`dgl.Graph`, Tensor)
            The graph and its label.
        """
        if self._transform is None:
            g = self.graphs[idx]
        else:
            g = self._transform(self.graphs[idx])
        return g, self.labels[idx], self.samples[idx]

    def has_cache(self):

        graph_path = os.path.join(self.save_mydata, '{}_ST_dgl_graph_{}{}.bin'.format(self.sampleid, self.hash, 'FtypeData'))
        if os.path.exists(graph_path):
            return True

        return False

    def save(self):
        """save the graph list and the labels"""
        if self.save_graph:
            graph_path = os.path.join(self.save_mydata, '{}_ST_dgl_graph_{}{}.bin'.format(self.sampleid, self.hash, 'FtypeData'))
            save_graphs(str(graph_path), self.graphs, {'labels': self.labels, 'samples': self.samples})

    def load(self):
        graphs, label_dict = load_graphs(os.path.join(self.save_mydata, '{}_ST_dgl_graph_{}{}.bin'.format(self.sampleid, self.hash, 'FtypeData')))
        self.graphs = graphs
        self.labels = label_dict['labels']
        self.samples = label_dict['samples']

    @property
    def num_classes(self):
        """Number of classes."""
        a_list = self.labels.numpy().tolist()
        # n_classes = len(set(a_list))
        n_classes = max(a_list) + 1
        return n_classes

# test = PtypeDataDglDataset()
# def collate(samples):
#     # 输入`samples` 是一个列表
#     # 每个元素都是一个二元组 (图, 标签)
#     graphs, labels = map(list, zip(*samples))
#     batched_graph = dgl.batch(graphs)
#     return batched_graph, torch.tensor(labels)
#
# # 数据集包含了80张图。每张图有10-20个节点
#
# graph, label = test[0]
# print(graph, label)
# data_loader = DataLoader(test, batch_size=1, shuffle=True,
#                          collate_fn=collate)
# #
# for iter, (bg, label) in enumerate(data_loader):
#     print((bg, label))

# fig, ax = plt.subplots()
# nx.draw(graph.to_networkx(), ax=ax)
# ax.set_title('Class: {:d}'.format(label))
# plt.show()
