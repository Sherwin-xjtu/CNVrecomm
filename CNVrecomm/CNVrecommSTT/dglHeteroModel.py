import dgl
import torch
import dgl.nn.pytorch as dglnn
import torch.nn as nn
from dgl.dataloading import GraphDataLoader
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl.function as fn

from CNVrecomm.CNVrecommSTT.PtypeDataDglDatasetT import PtypeDataDglDatasetT
from CNVrecomm.CNVrecommSTT.PtypeDataDglDataset import PtypeDataDglDataset
from CNVrecomm.CNVrecommSTT.FtypeDataDglDataset import FtypeDataDglDataset
from CNVrecomm.CNVrecommSTT.FtypeDataDglDatasetT import FtypeDataDglDatasetT
from CNVrecomm.CNVrecommSTT.StypeDataDglDataset import StypeDataDglDataset
from CNVrecomm.CNVrecommSTT.StypeDataDglDatasetT import StypeDataDglDatasetT
import matplotlib.pyplot as plt
from torch.nn import Linear
import pandas as pd


def collate(samples):
    # graphs, labels = map(list, zip(*samples))
    graphs, labels, samplesId = zip(*samples)
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)
    batched_samplesId = torch.tensor(samplesId)
    return batched_graph, batched_labels, batched_samplesId


# def collate(samples):
#     # graphs, labels = map(list, zip(*samples))
#     graphs, labels = zip(*samples)
#     batched_graph = dgl.batch(graphs)
#     batched_labels = torch.tensor(labels)
#     return batched_graph, batched_labels


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')
        # self.conv2 = dglnn.HeteroGraphConv({
        #     rel: dglnn.GraphConv(hid_feats, out_feats)
        #     for rel in rel_names}, aggregate='sum')

        self.bn1 = nn.BatchNorm1d(out_feats, momentum=1)
        self.lin = Linear(out_feats, 7)
        self.dropout = nn.Dropout(p=0.5)  # dropout训练

    def forward(self, graph, inputs):
        # inputs是节点的特征

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
        h = g.ndata['feature']
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            # 通过平均读出值来计算单图的表征
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
            return self.classify(hg)

def train(model, loader, epoch_loss, opt, epoch):
    for iter, (batched_graph, labels, samples) in enumerate(loader):
        logits = model(batched_graph)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    epoch_loss /= (iter + 1)
    print('Epoch {}, train loss {:.4f}'.format(epoch, epoch_loss))
    return epoch_loss


def validation(model, loader, epoch_loss, epoch):
    model.eval()
    for iter, (batched_graph, labels, samples) in enumerate(loader):  # Iterate in batches over the training dataset.
        logits = model(batched_graph)
        loss = F.cross_entropy(logits, labels)
        epoch_loss += loss.item()
    epoch_loss /= (iter + 1)
    print('Epoch {}, validation loss {:.4f}'.format(epoch, epoch_loss))
    return epoch_loss


def test(model, loader):
    model.eval()
    correct = 0
    for iter, (batched_graph, labels, samples) in enumerate(loader):  # Iterate in batches over the training/test dataset.
        logits = model(batched_graph)
        pred = logits.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == labels).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


def dataPred(model, loader):
    model.eval()
    pred_li = []
    labels_li = []
    samples_li = []
    for iter, (batched_graph, labels, samples) in enumerate(loader):  # Iterate in batches over the training/test dataset.
        logits = model(batched_graph)
        pred = logits.argmax(dim=1)  # Use the class with highest probability.
        pred_li = pred_li + pred.tolist()
        labels_li = labels_li + labels.tolist()
        samples_li = samples_li + samples.tolist()
    return pred_li, labels_li, samples_li

def ensembleResults(result_df):
    MMGR_pre_li = []
    labels_li = []
    samples_li = []

    sample_number = list(set(result_df["samples"].tolist()))
    for ro in range(len(sample_number)):
        sample_df = result_df[result_df["samples"] == ro]
        MMGR_pre = max(sample_df['pre'].tolist(), key=sample_df['pre'].tolist().count)
        MMGR_pre_li.append(MMGR_pre)
        label = max(sample_df['label'].tolist(), key=sample_df['label'].tolist().count)
        labels_li.append(label)
        samples_li.append(ro)
    return MMGR_pre_li, labels_li, samples_li


def dataLoad(tag):
    dataset1 = None
    dataset2 = None
    if tag == 'SSTT':
        dataset1 = StypeDataDglDataset()
        dataset2 = StypeDataDglDatasetT()
    elif tag == 'PSTT':
        dataset1 = PtypeDataDglDataset()
        dataset2 = PtypeDataDglDatasetT()
    elif tag == 'FSTT':
        dataset1 = FtypeDataDglDataset()
        dataset2 = FtypeDataDglDatasetT()
    return dataset1, dataset2

def modelDataLoader(dataset, bats = 80, shu=False, dro=False):
    dataset_loader = DataLoader(dataset, batch_size = bats, shuffle=shu, drop_last=dro,
               collate_fn=collate)
    return dataset_loader

def splitTrainSet(dataset1):
    tv_size = int(len(dataset1) * 0.8)
    test_size = len(dataset1) - tv_size
    tv_dataset, test_dataset = torch.utils.data.random_split(dataset1, [tv_size, test_size])

    train_size = int(len(tv_dataset) * 0.2)
    validation_size = len(tv_dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(tv_dataset, [train_size, validation_size])
    train_loader = modelDataLoader(train_dataset, 80, True, False)
    validation_loader = modelDataLoader(validation_dataset, 80, True, False)
    test_loader = modelDataLoader(test_dataset)
    return train_loader, validation_loader, test_loader

def CNVrecommModelTrain(dataset1, dataset2, tag):
    n_classes = dataset1.num_classes
    # etypes是一个列表，元素是字符串类型的边类型
    etypes = dataset1[0][0].etypes
    train_loader, validation_loader, test_loader = splitTrainSet(dataset1)
    data_loader = modelDataLoader(dataset1, bats=80)
    exomeKg_loader = modelDataLoader(dataset2, bats=80)
    model = HeteroClassifier(7, 64, n_classes, etypes)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    train_epoch_losses = []
    validation_epoch_losses = []
    # epoch_losses = []
    train_acc = []
    validation_acc = []
    test_acc = []
    data_acc = []
    results_li = []
    models_path = 'F:/CNVrecommendation/CNVrecomm/models/'
    train_models = []
    for epoch in range(1, 200):
        train_epoch_loss = 0
        train_epoch_loss = train(model, train_loader, train_epoch_loss, opt, epoch)
        validation_epoch_loss = 0
        validation_epoch_loss = validation(model, validation_loader, validation_epoch_loss, epoch)
        train_epoch_losses.append(train_epoch_loss)
        validation_epoch_losses.append(validation_epoch_loss)
        train_accu = test(model, train_loader)
        validation_accu = test(model, validation_loader)
        test_accu = test(model, test_loader)
        data_accu = test(model, data_loader)
        exomeKg_accu = test(model, exomeKg_loader)
        pre_li, labels_li, samples_li = dataPred(model, data_loader)
        data_acc.append(data_accu)
        train_models.append(model)

        result_dict = {'pre': pre_li, 'label': labels_li, 'samples': samples_li}
        result_df = pd.DataFrame(result_dict)
        MMGR_pre_li, label_li, sample_li = ensembleResults(result_df)
        results_li.append([MMGR_pre_li, label_li, sample_li])


        print(f'Epoch: {epoch:03d}, '
              f'Train Acc: {train_accu:.4f}, '
              f'Validation Acc: {validation_accu:.4f}, '
              f'Test Acc: {test_accu:.4f}, '
              f'data Acc: {data_accu:.4f}, '
              f'exomeKg Acc: {exomeKg_accu:.4f}'
              )
        train_acc.append(train_accu)
        validation_acc.append(validation_accu)
        test_acc.append(test_accu)
    ExtractingMetaTargetScarler = 'F:/CNVrecommendation/newCalling/ExtractingMetaTargetScarler.csv'
    ExtractingMetaTargetScarler_df = pd.read_csv(ExtractingMetaTargetScarler, encoding='gbk', low_memory=False)
    daindex = data_acc.index(max(data_acc))
    print(max(data_acc))
    # save model
    model_path = models_path + "{}{}{}".format(daindex, tag, 'model.pt')
    torch.save(train_models[daindex].state_dict(), model_path)
    ExtractingMetaTargetScarler_df['MMGR_pre'] = results_li[daindex][0]
    ExtractingMetaTargetScarler_df['label'] = results_li[daindex][1]
    ExtractingMetaTargetScarler_df['sample'] = results_li[daindex][2]
    ExtractingMetaTargetScarler_df.to_csv(
        'F:/CNVrecommendation/newCalling/results/ExtractingMetaTargetScarler_{}pre.csv'.format(tag), index=False)
    fig = plt.figure(1, figsize=(7, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    x = [i for i in range(1, 200)]
    p2 = plt.plot(x, train_epoch_losses, color='red', label=u'train_losses')
    p3 = plt.plot(x, validation_epoch_losses, color='blue', label=u'validation_losses')
    plt.legend()
    plt.xlabel(u'iters')
    plt.ylabel(u'loss')
    plt.title('cross entropy averaged over minibatches')
    plt.show()

    fig = plt.figure(2, figsize=(7, 5))
    ax2 = fig.add_subplot(1, 1, 1)
    x = [i for i in range(1, 200)]
    p4 = plt.plot(x, train_acc, color='red', label=u'train_acc')
    p5 = plt.plot(x, validation_acc, color='blue', label=u'validation_acc')
    p6 = plt.plot(x, test_acc, color='green', label=u'test_acc')
    plt.legend()
    plt.xlabel(u'iters')
    plt.ylabel(u'accuracy')
    plt.title('accuracy averaged over minibatches')
    plt.show()


def CNVrecommModelLoader(dataset1, models_path, num, tag):
    n_classes = dataset1.num_classes
    # etypes是一个列表，元素是字符串类型的边类型
    etypes = dataset1[0][0].etypes
    CNVrecomm_model = HeteroClassifier(7, 64, n_classes, etypes)
    model_path = models_path + "{}{}{}".format(num, tag, 'model.pt')
    CNVrecomm_model.load_state_dict(torch.load(model_path))
    # model.eval()
    return CNVrecomm_model


def CNVrecommMainSTT(tag, pattern):

    dataset1, dataset2 = dataLoad(tag)
    CNVrecommResults = None
    if pattern == 'train':
        CNVrecommModelTrain(dataset1, dataset2, tag)
        print('The train process of CNVrecommSTT is end!')
        CNVrecommResults = 'The train process of CNVrecommSTT is end!'
    elif pattern == 'test':
        model_final = 106
        models_path = 'F:/CNVrecommendation/CNVrecomm/models/'
        data_loader = modelDataLoader(dataset1, bats=80)
        exomeKg_loader = modelDataLoader(dataset2, bats=80)
        CNVrecomm_model = CNVrecommModelLoader(dataset1, models_path, model_final, tag)
        data_accu = test(CNVrecomm_model, data_loader)
        exomeKg_accu = test(CNVrecomm_model, exomeKg_loader)
        # pre_li, labels_li, samples_li = dataPred(CNVrecomm_model, data_loader)
        print('CNVrecomm_model{}{}:'.format(106,tag),'data Acc: {}'.format(data_accu), 'exomeKg Acc: {}'.format(exomeKg_accu))
        ExtractingMetaTargetScarler = 'F:/CNVrecommendation/newCalling/ExtractingMetaTargetScarler.csv'
        ExtractingMetaTargetScarler_df = pd.read_csv(ExtractingMetaTargetScarler, encoding='gbk', low_memory=False)
        pre_li, labels_li, samples_li = dataPred(CNVrecomm_model, data_loader)
        result_dict = {'pre': pre_li, 'label': labels_li, 'samples': samples_li}
        result_df = pd.DataFrame(result_dict)
        MMGR_pre_li, label_li, sample_li = ensembleResults(result_df)
        ExtractingMetaTargetScarler_df['MMGR_pre'] = MMGR_pre_li
        ExtractingMetaTargetScarler_df['label'] = label_li
        ExtractingMetaTargetScarler_df['sample'] = sample_li
        ExtractingMetaTargetScarler_df.to_csv(
            'F:/CNVrecommendation/newCalling/results/ExtractingMetaTargetScarler_{}pre.csv'.format(tag), index=False)

        CNVrecommResults = ExtractingMetaTargetScarler_df
    return CNVrecommResults

if __name__ == '__main__':
    meta_tag = 'PSTT'
    CNVrecomm_pattern = 'test'
    CNVrecommMainSTT(meta_tag, CNVrecomm_pattern)






