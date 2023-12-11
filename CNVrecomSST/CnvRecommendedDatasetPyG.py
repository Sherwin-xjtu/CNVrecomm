import dgl
import torch
from torch_geometric.data import InMemoryDataset, download_url, HeteroData
import torch
from torch_geometric.data import Data
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
from metaPath import extractGraphData

class MyOwnDataset(InMemoryDataset):
    # def __init__(self, root, transform=None, pre_transform=None):
    #     super().__init__(root, transform, pre_transform)
    #     self.data, self.slices = torch.load(self.processed_paths[0])
    def __init__(self, root,  transform=None, pre_transform=None):

        super(MyOwnDataset, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return ['cnv_dataset.pt',]

    @property
    def processed_file_names(self):
        return ['cnv_dataset.pt',]
        # #
        # def download(self):
        #     # Download to `self.raw_dir`.
        #     download_url(url, self.raw_dir)



    # def collate(self, samples):
    #     graphs, labels = map(list, zip(*samples))
    #     return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)

    def process(self):
        # Read data into huge `Data` list.
        # Read data into huge `Data` list.

        # edge_index = torch.tensor(
        #     [[0],
        #      [100]], dtype=torch.long
        # )
        # node_attr = torch.tensor(
        #     [[-1, 1, 2], [1, 1, 1]], dtype=torch.float
        # )
        # edge_attr = torch.tensor(
        #     [[0]], dtype=torch.float
        # )
        #
        # # X = torch.tensor([[-1], [0], [1], [3]], dtype=torch.float)
        # Y = torch.tensor([1], dtype=torch.long)
        # Y1 = torch.tensor([0], dtype=torch.long)
        #
        # data = Data(x=node_attr, edge_index=edge_index,edge_attr=edge_attr, y=Y)
        # data1 = Data(x=node_attr, edge_index=edge_index,edge_attr=edge_attr, y=Y1)

        # data_list = [data, data1]
        b = extractGraphData()
        tag = self.processed_paths[0].split('\\')[0][0]
        data_list = b.getDataList(tag)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
