import dgl
import numpy as np
import torch
import urllib3
urllib3.disable_warnings()
import dgl.data
import matplotlib.pyplot as plt
import networkx as nx
from torch.utils.data import DataLoader
from dgl.data.dgl_dataset import DGLBuiltinDataset
from dgl.data import TUDataset
# from dgl.data import MiniGCDataset
#
# def collate(samples):
#     # 输入`samples` 是一个列表
#     print(samples[0])
#     print(type(samples[0][0]))
#     # 每个元素都是一个二元组 (图, 标签)
#     graphs, labels = map(list, zip(*samples))
#     batched_graph = dgl.batch(graphs)
#     return batched_graph, torch.tensor(labels)
#
# # 数据集包含了80张图。每张图有10-20个节点
# dataset = MiniGCDataset(80, 10, 20)
# graph, label = dataset[0]
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True,
#                          collate_fn=collate)
#
# for iter, (bg, label) in enumerate(data_loader):
#     print((bg, label))
#     exit()
# fig, ax = plt.subplots()
# nx.draw(graph.to_networkx(), ax=ax)
# ax.set_title('Class: {:d}'.format(label))
# plt.show()



n_users = 1000
n_items = 500
n_follows = 3000
n_clicks = 5000
n_dislikes = 500
n_hetero_features = 10
n_user_classes = 5
n_max_clicks = 10

"""
randint(low=0, high, size)
值在low到high之间，形状由size决定
torch.randint(6,(2,2))

tensor([[5, 1],
        [4, 3]])
"""
follow_src = np.random.randint(0, n_users, n_follows)
follow_dst = np.random.randint(0, n_users, n_follows)
click_src = np.random.randint(0, n_users, n_clicks)
click_dst = np.random.randint(0, n_items, n_clicks)
dislike_src = np.random.randint(0, n_users, n_dislikes)
dislike_dst = np.random.randint(0, n_items, n_dislikes)

hetero_graph = dgl.heterograph({
    ('user', 'follow', 'user'): (follow_src, follow_dst),
    ('user', 'followed-by', 'user'): (follow_dst, follow_src),
    ('user', 'click', 'item'): (click_src, click_dst),
    ('item', 'clicked-by', 'user'): (click_dst, click_src),
    ('user', 'dislike', 'item'): (dislike_src, dislike_dst),
    ('item', 'disliked-by', 'user'): (dislike_dst, dislike_src)})

hetero_graph.nodes['user'].data['feature'] = torch.randn(n_users, n_hetero_features)
hetero_graph.nodes['item'].data['feature'] = torch.randn(n_items, n_hetero_features)
hetero_graph.nodes['user'].data['label'] = torch.randint(0, n_user_classes, (n_users,))
hetero_graph.edges['click'].data['label'] = torch.randint(1, n_max_clicks, (n_clicks,)).float()
# hetero_graph.label = torch.tensor(1, dtype=torch.long)

# 在user类型的节点和click类型的边上随机生成训练集的掩码
hetero_graph.nodes['user'].data['train_mask'] = torch.zeros(n_users, dtype=torch.bool).bernoulli(0.6)
hetero_graph.edges['click'].data['train_mask'] = torch.zeros(n_clicks, dtype=torch.bool).bernoulli(0.6)

print(torch.zeros(n_clicks, dtype=torch.bool))
# tensor([False, False, False,  ..., False, False, False]) torch.Size([5000])
print(hetero_graph.edges['click'].data['train_mask'])
# tensor([False,  True, False,  ...,  True, False,  True]) torch.Size([5000])
