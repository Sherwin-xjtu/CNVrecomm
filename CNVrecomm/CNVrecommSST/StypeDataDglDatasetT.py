"""A mini synthetic dataset for graph classification benchmark."""
import math, os

import dgl
import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
from CNVrecomm.CNVrecommSST.metaPathDGLT import extractGraphData

__all__ = ['StypeDataDglDatasetT']

from dgl import load_graphs, save_graphs

from dgl.data import DGLDataset
from torch.utils.data import DataLoader


class StypeDataDglDatasetT(DGLDataset):
    def __init__(self, seed=0,save_graph=True):

        self.seed = seed
        self.save_graph = save_graph
        self.save_mydata = 'F:/CNVrecommendation/exomeData/dglDataset'
        super(StypeDataDglDatasetT, self).__init__(name="StypeDataDglDatasetT")

    def process(self):

        b = extractGraphData()
        tag = 'S'
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

        graph_path = os.path.join(self.save_mydata, 'SST_dgl_graph_{}{}.bin'.format(self.hash, 'StypeData'))
        if os.path.exists(graph_path):
            return True

        return False

    def save(self):
        """save the graph list and the labels"""
        if self.save_graph:
            graph_path = os.path.join(self.save_mydata, 'SST_dgl_graph_{}{}.bin'.format(self.hash, 'StypeData'))
            save_graphs(str(graph_path), self.graphs, {'labels': self.labels, 'samples': self.samples})

    def load(self):
        graphs, label_dict = load_graphs(os.path.join(self.save_mydata, 'SST_dgl_graph_{}{}.bin'.format(self.hash, 'StypeData')))
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
