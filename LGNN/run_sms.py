import pathpy as pp
import pandas as pd
import os

from loader import load_data

dir = r'Data/'
fold = 'SMS'
fold_path = os.path.join(dir, fold)

loader = load_data(fold_path, True)
data = loader.process()

runss = 50
epochs = 5000

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score, balanced_accuracy_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

def calculate_metrics(y_true, y_pred):
    return {'F1-score-weighted':f1_score(y_true, y_pred, average='weighted'),
            'F1-score-macro':f1_score(y_true, y_pred, average='macro'),
            'F1-score-micro':f1_score(y_true, y_pred, average='micro'),
            'Accuracy':accuracy_score(y_true, y_pred),
            'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
            'Precision-weighted':precision_score(y_true, y_pred, average='weighted'),
            'Precision-macro':precision_score(y_true, y_pred, average='macro'),
            'Precision-micro':precision_score(y_true, y_pred, average='micro'),
            'Recall-weighted':recall_score(y_true, y_pred, average='weighted'),
            'Recall-macro':recall_score(y_true, y_pred, average='macro'),
            'Recall-micro':recall_score(y_true, y_pred, average='micro'),
            'AMI': adjusted_mutual_info_score(y_true, y_pred)}

import torch.nn as nn
import torch.nn.functional as F

import dgl

# Return a list containing features gathered from multiple radius.
import dgl.function as fn
def aggregate_radius(radius, g, z):
    # initializing list to collect message passing result
    z_list = []
    g.ndata['z'] = z
    # pulling message from 1-hop neighbourhood
    g.update_all(fn.copy_src(src='z', out='m'), fn.sum(msg='m', out='z'))
    z_list.append(g.ndata['z'])
    for i in range(radius - 1):
        for j in range(2 ** i):
            #pulling message from 2^j neighborhood
            g.update_all(fn.copy_src(src='z', out='m'), fn.sum(msg='m', out='z'))
        z_list.append(g.ndata['z'])
    return z_list


class LGNNCore(nn.Module):
    def __init__(self, in_feats, out_feats, radius):
        super(LGNNCore, self).__init__()
        self.out_feats = out_feats
        self.radius = radius

        self.linear_prev = nn.Linear(in_feats, out_feats)
        self.linear_deg = nn.Linear(in_feats, out_feats)
        self.linear_radius = nn.ModuleList(
                [nn.Linear(in_feats, out_feats) for i in range(radius)])
        self.linear_fuse = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)

    def forward(self, g, feat_a, feat_b, deg, pm_pd):
        # term "prev"
        prev_proj = self.linear_prev(feat_a)
        # term "deg"
        deg_proj = self.linear_deg(deg * feat_a)

        # term "radius"
        # aggregate 2^j-hop features
        hop2j_list = aggregate_radius(self.radius, g, feat_a)
        # apply linear transformation
        hop2j_list = [linear(x) for linear, x in zip(self.linear_radius, hop2j_list)]
        radius_proj = sum(hop2j_list)

        # term "fuse"
        fuse = self.linear_fuse(torch.mm(pm_pd, feat_b))

        # sum them together
        result = prev_proj + deg_proj + radius_proj + fuse

        # skip connection and batch norm
        n = self.out_feats // 2
        result = torch.cat([result[:, :n], F.relu(result[:, n:])], 1)
        result = self.bn(result)

        return result

class LGNNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, radius):
        super(LGNNLayer, self).__init__()
        self.g_layer = LGNNCore(in_feats, out_feats, radius)
        self.lg_layer = LGNNCore(in_feats, out_feats, radius)

    def forward(self, g, lg, x, lg_x, deg_g, deg_lg, pm_pd):
        next_x = self.g_layer(g, x, lg_x, deg_g, pm_pd)
        pm_pd_y = torch.transpose(pm_pd, 0, 1)
        next_lg_x = self.lg_layer(lg, lg_x, x, deg_lg, pm_pd_y)
        return next_x, next_lg_x

class LGNN(nn.Module):
    def __init__(self, radius):
        super(LGNN, self).__init__()
        self.layer1 = LGNNLayer(1, 16, radius)  # input is scalar feature
        self.layer2 = LGNNLayer(16, 16, radius)  # hidden size is 16
        self.layer3 = LGNNLayer(16, 16, radius)
        self.linear = nn.Linear(16, len(data.y.unique()))  # predice two classes

    def forward(self, g, lg, pm_pd):
        # compute the degrees
        deg_g = g.in_degrees().float().unsqueeze(1)
        deg_lg = lg.in_degrees().float().unsqueeze(1)
        # use degree as the input feature
        x, lg_x = deg_g, deg_lg
        x, lg_x = self.layer1(g, lg, x, lg_x, deg_g, deg_lg, pm_pd)
        x, lg_x = self.layer2(g, lg, x, lg_x, deg_g, deg_lg, pm_pd)
        x, lg_x = self.layer3(g, lg, x, lg_x, deg_g, deg_lg, pm_pd)
        return self.linear(x)

# from torch_geometric.utils.convert import to_networkx

# g = to_networkx(data, to_undirected=True)
# print(g.is_directed())

# from dgl import from_networkx

g = dgl.graph((data.edge_index[0], data.edge_index[1]))
g.ndata['z'] = torch.ones(g.num_nodes(), g.num_nodes())

lg = g.line_graph(backtracking=False)
lg.ndata['z'] = torch.ones(lg.num_nodes(), lg.num_nodes())


# A utility function to convert a scipy.coo_matrix to torch.SparseFloat
def sparse2th(mat):
    value = mat.data
    indices = torch.LongTensor([mat.row, mat.col])
    tensor = torch.sparse.FloatTensor(indices, torch.from_numpy(value).float(), mat.shape)
    return tensor

from sklearn.model_selection import train_test_split
import numpy as np

def train_test_mask(dataset):
    _, _, index_mask, _ = train_test_split(
        np.array(list(range(dataset.num_nodes))),
        np.array(list(range(dataset.num_nodes))),
        test_size=0.3,
        shuffle=True,
        stratify=[i.item() for i in dataset.y]
    )

    train_mask = np.zeros(dataset.num_nodes).astype(bool)
    train_mask[index_mask] = True
    test_mask = np.invert(train_mask)

    return train_mask, test_mask


from torch_geometric.transforms import LineGraph

t = LineGraph(force_directed = True)
line = t(data.clone())

fon = []
hon = []
for i in range(line.num_nodes):
    fon.append(data.edge_index[0][i].item())
    hon.append(i)

for i in range(line.num_nodes):
    fon.append(data.edge_index[1][i].item())
    hon.append(i)

fon = np.array(fon)
hon = np.array(hon)
from scipy.sparse import coo_matrix
coo = coo_matrix((np.ones(len(fon)), (fon, hon)), shape=(data.num_nodes, line.num_nodes))

results = []

for runs in tqdm(range(runss)):

    # train test split
    data.train_mask, data.test_mask = train_test_mask(data)

    # Create the model
    model = LGNN(radius=3)
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    loss_function = torch.nn.CrossEntropyLoss()

    pmpd = sparse2th(coo)

    for _ in tqdm(range(epochs)):

        # Forward
        z = model(g, lg, pmpd)

        loss = loss_function(z[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    _, pred = model(g, lg, pmpd).max(dim=1)

    results.append(calculate_metrics(data.y[data.test_mask], pred[data.test_mask]))


import numpy as np

df= pd.DataFrame(results)

df = pd.DataFrame([df.apply(lambda x: round(x*100,2)).mean(), df.apply(lambda x: round(x*100,2)).std()])

df.to_csv(f'{fold}.results', index=False)