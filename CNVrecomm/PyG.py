import dgl
import scipy
import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_networkx
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



# data_file_path = './MyOwnDataset/ACM.mat'
# # data['label'] = Y = torch.tensor(0, dtype=torch.long)
# data = scipy.io.loadmat(data_file_path)
# print(list(data.keys()))
# print(data['PvsA'])
data = HeteroData()
# 初始化结点特征
data['sample'].x = torch.tensor( [[-1,1,2],[1,1,1]])
data['tool'].x = torch.tensor( [[1,2],[1,1],[1,1]])
# data['institution'].x = ... # [num_institutions, num_features_institution]
# data['field_of_study'].x = ... # [num_field, num_features_field]
# 初始化边索引
data['sample', 'choose', 'tool'].edge_index = torch.tensor([[0, 0, 0], [0, 1, 2]],dtype=torch.long)
# data['paper', 'cites', 'paper'].edge_index = ... # [2, num_edges_cites]
# data['author', 'writes', 'paper'].edge_index = ... # [2, num_edges_writes]
# data['author', 'affiliated_with', 'institution'].edge_index = ... # [2, num_edges_affiliated]
# data['author', 'has_topic', 'institution'].edge_index = ... # [2, num_edges_topic]
# 初始化边特征
data['sample', 'choose', 'tool'].edge_attr = edge_attr=torch.tensor([[0,0,0], [0,0,0], [0,0,0]])
# data['paper', 'cites', 'paper'].edge_attr = ... # [num_edges_cites, num_features_cites]
# data['author', 'writes', 'paper'].edge_attr = ... # [num_edges_writes, num_features_writes]
# data['author', 'affiliated_with', 'institution'].edge_attr = ... # [num_edges_affiliated, num_features_affiliated]
# data['paper', 'has_topic', 'field_of_study'].edge_attr = ... # [num_edges_topic, num_features_topic]
data['label'] = Y = torch.tensor(0, dtype=torch.long)
node_types, edge_types = data.metadata()
# print(data.is_undirected())
print(data.x_dict)
print(data.edge_index_dict)
print(node_types, edge_types )
# 异构转同构
homogeneous_data = data.to_homogeneous()
print(homogeneous_data)
plt.subplot(224)
nx.draw(to_networkx(homogeneous_data), with_labels=True)
plt.show()
exit()
edge_index=torch.tensor(
    [[1],
    [1]],dtype=torch.long
)
node_attr=torch.tensor(
    [[-1,1,2],[1,1,1]]
)

edge_attr=torch.tensor(
    [[0,0,0]]
)
y=torch.tensor(
    1
)
data=Data(x=node_attr,edge_index=edge_index,edge_attr=edge_attr, y=y)
print(data)

plt.subplot(224)
nx.draw(to_networkx(data), with_labels=True)

plt.show()