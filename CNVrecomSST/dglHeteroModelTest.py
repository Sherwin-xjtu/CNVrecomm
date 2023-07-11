import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

def create_graph():
    n_users = 1000
    n_items = 500
    n_follows = 3000
    n_clicks = 5000
    n_dislikes = 500
    n_hetero_features = 10

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

    hetero_graph.nodes['user'].data['feat'] = torch.randn(n_users, n_hetero_features)
    hetero_graph.nodes['item'].data['feat'] = torch.randn(n_items, n_hetero_features)

    label = np.random.randint(0, 5, 1)
    return hetero_graph, torch.LongTensor(label)

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.ndata['feat']
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
            return self.classify(hg)

dataset = [create_graph() for i in range(1000)]

def collate(samples):
    # graphs, labels = map(list, zip(*samples))
    graphs, labels = zip(*samples)
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)
    return batched_graph, batched_labels

dataloader = DataLoader(
    dataset,
    batch_size=1024,
    collate_fn=collate,
    drop_last=False,
    shuffle=True)

etypes = dataset[0][0].etypes
model = HeteroClassifier(10, 20, 5, etypes)
opt = torch.optim.Adam(model.parameters())
for epoch in range(20):
    for batched_graph, labels in dataloader:
        logits = model(batched_graph)
        loss = F.cross_entropy(logits, labels.squeeze(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())
