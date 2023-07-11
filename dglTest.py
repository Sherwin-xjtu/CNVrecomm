import networkx as nx
import dgl
import torch
import numpy as np
import torch as th
import scipy.sparse as spp
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

graph_data = {
    ('user', 'follows', 'user') : (torch.tensor([0]), torch.tensor([1])),
    ('user', 'plays', 'game') : (torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 1, 2]))
}
g = dgl.heterograph(graph_data)
print(g)
exit()
hg1 = dgl.to_homogeneous(g)

plt.subplot(224)
nx.draw(hg1.to_networkx(), with_labels=True)

plt.show()
exit()


G = dgl.DGLGraph()
G.add_nodes(3)

# G.ndata['x'] = th.zeros((3, 5))  # init 3 nodes with zero vector(len=5)
# G.nodes[[0, 2]].data['x'] = th.ones((2, 5))
print(G.ndata)
G.add_edges([0, 1], 2)
G.edata['y'] = torch.tensor([0.56,0.9])
print(G)
plt.subplot(224)
nx.draw(G.to_networkx(), with_labels=True)

plt.show()
exit()

## 方式1： 使用两个节点数组构造图
u = torch.tensor([0,1,0,0,0])
v = torch.tensor([1,2,3,4,5])
g1 = dgl.DGLGraph((u,v))

# 如果数组之一是标量，该值自动广播以匹配另一个数组的长度,称为“边缘广播”的功能。
# g1 = dgl.DGLGraph((0,v))

## 方式2: 使用稀疏矩阵进行构造
adj = spp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy()))) ## 传入的参数(data, (row, col))
g2 = dgl.DGLGraph(adj)

## 方式3: 使用networkx
g_nx =nx.petersen_graph()
g3 = dgl.DGLGraph(g_nx)

## 方式4：加边 (没有上面的方法高效)
g4 = dgl.DGLGraph()
g4.add_nodes(10) # 添加节点数量 该方法第二个参数是添加每个节点的特征。
## 加入边
for i in range(1,5): # 一条条边添加
    g4.add_edge(i,0)

src = list(range(5,8));dst = [0]*3 # 使用list批量添加
g4.add_edges(src, dst)
src = torch.tensor([8,9]);dst = torch.tensor([0,0]) # 使用list批量添加
g4.add_edges(src, dst)

plt.subplot(221)
nx.draw(g1.to_networkx(), with_labels=True)
plt.subplot(222)
nx.draw(g2.to_networkx(), with_labels=True)
plt.subplot(223)
nx.draw(g3.to_networkx(), with_labels=True)
plt.subplot(224)
nx.draw(g4.to_networkx(), with_labels=True)

plt.show()