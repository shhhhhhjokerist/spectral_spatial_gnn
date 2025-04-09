from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
import dgl
import numpy as np
import torch

from data.graph_data.graph_construction import amazon_graph, amazon_graph_random_drop_v2, amazon_graph_random_drop_v3


class Dataset:
    def __init__(self, name='tfinance', homo=True, anomaly_alpha=None, anomaly_std=None, sample=True, sample_num=15000):
        self.name = name
        graph = None
        if name == 'tfinance':
            graph, label_dict = load_graphs('dataset/tfinance')
            graph = graph[0]
            graph.ndata['label'] = graph.ndata['label'].argmax(1)

            if anomaly_std:
                graph, label_dict = load_graphs('dataset/tfinance')
                graph = graph[0]
                feat = graph.ndata['feature'].numpy()
                anomaly_id = graph.ndata['label'][:,1].nonzero().squeeze(1)
                feat = (feat-np.average(feat,0)) / np.std(feat,0)
                feat[anomaly_id] = anomaly_std * feat[anomaly_id]
                graph.ndata['feature'] = torch.tensor(feat)
                graph.ndata['label'] = graph.ndata['label'].argmax(1)

            if anomaly_alpha:
                graph, label_dict = load_graphs('dataset/tfinance')
                graph = graph[0]
                feat = graph.ndata['feature'].numpy()
                anomaly_id = list(graph.ndata['label'][:, 1].nonzero().squeeze(1))
                normal_id = list(graph.ndata['label'][:, 0].nonzero().squeeze(1))
                label = graph.ndata['label'].argmax(1)
                diff = anomaly_alpha * len(label) - len(anomaly_id)
                import random
                new_id = random.sample(normal_id, int(diff))
                # new_id = random.sample(anomaly_id, int(diff))
                for idx in new_id:
                    aid = random.choice(anomaly_id)
                    # aid = random.choice(normal_id)
                    feat[idx] = feat[aid]
                    label[idx] = 1  # 0

        elif name == 'tsocial':
            graph, label_dict = load_graphs('dataset/tsocial')
            graph = graph[0]

        elif name == 'yelp':
            dataset = FraudYelpDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
        elif name == 'amazon':
            dataset = FraudAmazonDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
            if sample:
                nodes_num = graph.num_nodes()
                sample_num = 1000
                graph = amazon_graph(True, sample_num, sample_num*200, 0, 1, 0.5, True)
                # graph = amazon_graph_random_drop_v3(True, 1000, 0, 0, None, True)
            # graph.ndata['cat_feat'] = []
            # graph.ndata['neigh_feat'] = []
        elif name == 'amazon_knn':
            graphs, label_dict = dgl.load_graphs('./dataset/amazon/edknn_graph.dgl')
            graph = graphs[0]
            if homo:
                graph = dgl.to_homogeneous(graphs[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
        elif name == 'amazon_cos':
            graphs, label_dict = dgl.load_graphs('./dataset/amazon/cos_graph.dgl')
            graph = graphs[0]
            if homo:
                graph = dgl.to_homogeneous(graphs[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
        elif name == 'amazon_cos':
            graphs, label_dict = dgl.load_graphs('./dataset/amazon/cos_graph.dgl')
            graph = graphs[0]
            if homo:
                graph = dgl.to_homogeneous(graphs[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
        else:
            print('no such dataset')
            exit(1)

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        print(graph)

        self.graph = graph
