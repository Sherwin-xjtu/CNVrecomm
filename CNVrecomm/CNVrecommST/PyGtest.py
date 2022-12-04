from CnvRecommendedDatasetPyG import MyOwnDataset
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
import torch
import dgl
from torch_geometric.data import Data, Batch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

"""测试"""
dataset1 = MyOwnDataset("MYttttdata")
# img, target = dataset1[0]
# print(img.shape)
# print(target)

from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='data/TUDataset', name='MUTAG')

#
# # plt.subplot(224)
# nx.draw(to_networkx(b[0]), with_labels=True)
#
# plt.show()
# print(b.data.num_features)
#
# print(b.data.num_nodes)
#
# print(b.data.y)

torch.manual_seed(0)
dataset = dataset.shuffle()
train_dataset = dataset[:150]
test_dataset = dataset[150:]

dataset1 = dataset1.shuffle()
train_dataset1 = dataset1[:150]
test_dataset1 = dataset1[150:]

# print(f'Number of training graphs: {len(train_dataset)}')
# print(f'Number of test graphs: {len(test_dataset)}')
#
from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# for step, data in enumerate(train_loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print(data)
#     print()
#
train_loader1 = DataLoader(dataset1, batch_size=64, shuffle=False)

test_loader1 = DataLoader(test_dataset1, batch_size=64, shuffle=False)
#


# for step, data in enumerate(train_loader1):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print(data.edge_index)
#     print()

# for data in train_loader1:  # Iterate in batches over the training dataset.
#
#     # plt.subplot(224)
#     nx.draw(to_networkx(data), with_labels=True)
#
#     plt.show()
#     exit()

# for data in train_loader:  # Iterate in batches over the training dataset.
#     print(data.edge_index)
#     nx.draw(to_networkx(data), with_labels=True)
#     plt.show()
# exit()


from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def train(loader):
    model.train()

    for data in loader:  # Iterate in batches over the training dataset.
        print(data.edge_index)
        print(data)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.

for epoch in range(1, 171):

    train(train_loader1)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')