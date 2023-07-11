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

# 这里给出大家注释方便理解
class MyOwnDataset(InMemoryDataset):
    # def __init__(self, root, transform=None, pre_transform=None):
    #     super().__init__(root, transform, pre_transform)
    #     self.data, self.slices = torch.load(self.processed_paths[0])
    def __init__(self, root,  transform=None, pre_transform=None):

        super(MyOwnDataset, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])


    # 返回数据集源文件名
    @property
    def raw_file_names(self):
        return ['cnv_dataset.pt',]

    # 返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['cnv_dataset.pt',]
        # #用于从网上下载数据集
        # def download(self):
        #     # Download to `self.raw_dir`.
        #     download_url(url, self.raw_dir)



    # def collate(self, samples):
    #     # 输入参数samples是一个列表
    #     # 列表里的每个元素是图和标签对，如[(graph1, label1), (graph2, label2), ...]
    #     graphs, labels = map(list, zip(*samples))
    #     return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)

    # 生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        # Read data into huge `Data` list.
        # 这里用于构建data

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
        # # 每个节点的特征：从0号节点开始。。
        # # X = torch.tensor([[-1], [0], [1], [3]], dtype=torch.float)
        # # 每个节点的标签：从0号节点开始-两类0，1
        # Y = torch.tensor([1], dtype=torch.long)
        # Y1 = torch.tensor([0], dtype=torch.long)
        #
        # data = Data(x=node_attr, edge_index=edge_index,edge_attr=edge_attr, y=Y)
        # data1 = Data(x=node_attr, edge_index=edge_index,edge_attr=edge_attr, y=Y1)
        # 放入datalist

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