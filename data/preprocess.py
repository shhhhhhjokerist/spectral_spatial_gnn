import dgl
import torch
import numpy as np
import os
import random
# from pygod.utils import load_data
import pandas
import bidict
from dgl.data import FraudAmazonDataset, FraudYelpDataset
from sklearn.model_selection import train_test_split

def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class Dataset:
    def __init__(self, name='tfinance', homo=True, add_self_loop=True, to_bidirectional=False, to_simple=True):
        if name == 'yelp':
            dataset = FraudYelpDataset()
            graph = dataset[0]
            graph.ndata['train_mask'] = graph.ndata['train_mask'].bool()
            graph.ndata['val_mask'] = graph.ndata['val_mask'].bool()
            graph.ndata['test_mask'] = graph.ndata['test_mask'].bool()
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])

        elif name == 'amazon':
            dataset = FraudAmazonDataset()
            graph = dataset[0]
            graph.ndata['train_mask'] = graph.ndata['train_mask'].bool()
            graph.ndata['val_mask'] = graph.ndata['val_mask'].bool()
            graph.ndata['test_mask'] = graph.ndata['test_mask'].bool()
            graph.ndata['mark'] = graph.ndata['train_mask']+graph.ndata['val_mask']+graph.ndata['test_mask']
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask', 'mark'])

        else:
            graph = dgl.load_graphs('../dataset/'+name)[0][0]
            # graph = dgl.load_graphs(name)[0][0]
        graph.ndata['feature'] = graph.ndata['feature'].float()
        graph.ndata['label'] = graph.ndata['label'].long()
        self.name = name
        self.graph = graph
        if add_self_loop:
            self.graph = dgl.add_self_loop(self.graph)
        if to_bidirectional:
            self.graph = dgl.to_bidirected(self.graph, copy_ndata=True)
        if to_simple:
            self.graph = dgl.to_simple(self.graph)

    def split(self, samples=20):
        labels = self.graph.ndata['label']
        n = self.graph.num_nodes()
        if 'mark' in self.graph.ndata:
            index = self.graph.ndata['mark'].nonzero()[:,0].numpy().tolist()
        else:
            index = list(range(n))
        train_masks = torch.zeros([n,20]).bool()
        val_masks = torch.zeros([n,20]).bool()
        test_masks = torch.zeros([n,20]).bool()
        if self.name in ['tolokers', 'questions']:
            train_ratio, val_ratio = 0.5, 0.25
        if self.name in ['tsocial', 'tfinance', 'reddit', 'weibo']:
            train_ratio, val_ratio = 0.4, 0.2
        if self.name in ['amazon', 'yelp', 'elliptic', 'dgraphfin']:  # official split
            train_masks[:,:10] = self.graph.ndata['train_mask'].repeat(10,1).T
            val_masks[:,:10] = self.graph.ndata['val_mask'].repeat(10,1).T
            test_masks[:,:10] = self.graph.ndata['test_mask'].repeat(10,1).T
        else:
            for i in range(10):
                seed = 3407+10*i
                set_seed(seed)
                idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index], train_size=train_ratio, random_state=seed, shuffle=True)
                idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, train_size=int(len(index)*val_ratio), random_state=seed, shuffle=True)
                train_masks[idx_train,i] = 1
                val_masks[idx_valid,i] = 1
                test_masks[idx_test,i] = 1

        for i in range(10):
            pos_index = np.where(labels == 1)[0]
            neg_index = list(set(index) - set(pos_index))
            pos_train_idx = np.random.choice(pos_index, size=2*samples, replace=False)
            neg_train_idx = np.random.choice(neg_index, size=8*samples, replace=False)
            train_idx = np.concatenate([pos_train_idx[:samples], neg_train_idx[:4*samples]])
            train_masks[train_idx, 10+i] = 1
            val_idx = np.concatenate([pos_train_idx[samples:], neg_train_idx[4*samples:]])
            val_masks[val_idx, 10+i] = 1
            test_masks[index, 10+i] = 1
            test_masks[train_idx, 10+i] = 0
            test_masks[val_idx, 10+i] = 0

        self.graph.ndata['train_masks'] = train_masks
        self.graph.ndata['val_masks'] = val_masks
        self.graph.ndata['test_masks'] = test_masks


# preprocess elliptic
def elliptic_preprocess():
    labels = pandas.read_csv('../dataset/elliptic_bitcoin_dataset/elliptic_txs_classes.csv').to_numpy()
    node_features = pandas.read_csv('../dataset/elliptic_bitcoin_dataset/elliptic_txs_features.csv',
                                    header=None).to_numpy()

    node_dict = bidict.bidict()
    # node_dict = None

    for i in range(labels.shape[0]):
        node_dict[i] = labels[i][0]

    new_labels = np.zeros(labels.shape[0]).astype(int)
    marks = labels[:, 1] != 'unknown'
    features = node_features[:, 1:]
    new_labels[labels[:, 1] == '1'] = 1

    train_mask = (features[:, 0] <= 25) & marks
    val_mask = (features[:, 0] > 25) & (features[:, 0] <= 34) & marks
    test_mask = (features[:, 0] > 34) & marks
    print(train_mask.sum(), val_mask.sum(), test_mask.sum())
    edges = pandas.read_csv('../dataset/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv').to_numpy()

    new_edges = np.zeros_like(edges)

    for i in range(edges.shape[0]):
        new_edges[i][0] = node_dict.inv[edges[i][0]]
        new_edges[i][1] = node_dict.inv[edges[i][1]]

    graph = dgl.graph((new_edges[:, 0], new_edges[:, 1]))
    graph.ndata['train_mask'] = torch.tensor(train_mask).bool()
    graph.ndata['val_mask'] = torch.tensor(val_mask).bool()
    graph.ndata['test_mask'] = torch.tensor(test_mask).bool()
    graph.ndata['mark'] = torch.tensor(marks).bool()
    graph.ndata['label'] = torch.tensor(new_labels)
    graph.ndata['feature'] = torch.tensor(features)

    dgl.save_graphs('../dataset/elliptic', [graph])

    # preprocess dgraphfin
    # f = np.load('../dataset/dgraphfin.npz')
    # x = torch.tensor(f['x']).float()
    # y = torch.tensor(f['y']).float()
    # y = (y == 1).int()
    # g = dgl.graph((f['edge_index'][:, 0], f['edge_index'][:, 1]))
    #
    # g.ndata['feature'] = x
    # g.ndata['label'] = y
    #
    # g.ndata['train_mask'] = torch.zeros_like(y).bool()
    # g.ndata['val_mask'] = torch.zeros_like(y).bool()
    # g.ndata['test_mask'] = torch.zeros_like(y).bool()
    #
    # g.ndata['train_mask'][list(f['train_mask'])] = True
    # g.ndata['val_mask'][list(f['valid_mask'])] = True
    # g.ndata['test_mask'][list(f['test_mask'])] = True
    # g.ndata['mark'] = (g.ndata['train_mask'] + g.ndata['val_mask'] + g.ndata['test_mask']).bool()
    # dgl.save_graphs('../dataset/dgraphfin', [g])

    # split dgraphfin and elliptic
    for data_name in ['elliptic']:
        data = Dataset(data_name)
        data.split()
        print(data.graph)
        print(data.graph.ndata['train_masks'].sum(0), data.graph.ndata['val_masks'].sum(0),
              data.graph.ndata['test_masks'].sum(0))
        dgl.save_graphs('../dataset/' + data_name, [data.graph])

# the following code is just for reference and not needed for the experiments if you have downloaded the datasets

# preprocess reddit / amazon
# data = load_data('reddit','data-main')
# graph = dgl.graph((data.edge_index[0,:], data.edge_index[1,:]))
# graph.ndata['feature'] = data.x
# graph.ndata['label'] = data.y
# graph.ndata['label'].sum()
# dgl.save_graphs('reddit', [graph])

# preprocess t-social
# g = dgl.load_graphs('data/tsocial')[0][0]
# g = dgl.to_homogeneous(g, ['feature', 'label', 'train_masks', 'val_masks', 'test_masks'])
# dgl.save_graphs('data/tsocial', [g])

# split all datasets
# for data_name in ['amazon', 'yelp', 'questions', 'tsocial', 'tfinance', 'reddit', 'weibo']:
#     data = Dataset(data_name)
#     data.split()
#     print(data.graph)
#     print(data.graph.ndata['train_masks'].sum(0), data.graph.ndata['val_masks'].sum(0), data.graph.ndata['test_masks'].sum(0))
#     dgl.save_graphs(data_name, [data.graph])

elliptic_preprocess()

