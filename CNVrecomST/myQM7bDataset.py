from dgl.data import QM7bDataset
qm7b = QM7bDataset()
g, label = qm7b[0]
print(type(qm7b))