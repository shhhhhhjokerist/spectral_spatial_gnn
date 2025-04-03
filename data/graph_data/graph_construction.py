import random

import networkx as nx
import dgl
import numpy
import numpy as np
import scipy.sparse as sp
import torch
from dgl.data import FraudAmazonDataset
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

dataset = FraudAmazonDataset()
graph = dataset[0]
graph.ndata['train_mask'] = graph.ndata['train_mask'].bool()
graph.ndata['val_mask'] = graph.ndata['val_mask'].bool()
graph.ndata['test_mask'] = graph.ndata['test_mask'].bool()
graph.ndata['mark'] = graph.ndata['train_mask'] + graph.ndata['val_mask'] + graph.ndata['test_mask']
graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask', 'mark'])

graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
graph.ndata['feature'] = graph.ndata['feature'].float()

features = graph.ndata['feature']

features_np = features.numpy()

# org graph
original_graph = graph


def euclidean_dis_knn():
    # Euclidean distance knn
    k = 5
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(features_np)

    distances, indices = knn.kneighbors(features_np)

    edges = []
    for i in range(len(indices)):
        for j in indices[i]:
            if i != j:  # 去除自环
                edges.append((i, j))


    u, v = zip(*edges)
    graph_knn = dgl.graph((u, v), num_nodes=graph.num_nodes())

    graph_knn.ndata['feature'] = graph.ndata['feature']
    graph_knn.ndata['label'] = graph.ndata['label']
    graph_knn.ndata['train_mask'] = graph.ndata['train_mask']
    graph_knn.ndata['val_mask'] = graph.ndata['val_mask']
    graph_knn.ndata['test_mask'] = graph.ndata['test_mask']
    graph_knn.ndata['mark'] = graph.ndata['mark']

    return graph_knn

def cos_similarity_knn():
    # cos similarity knn
    cos_sim = cosine_similarity(features.numpy())

    threshold = 0.9
    edges = []

    for i in range(len(cos_sim)):
        for j in range(i+1, len(cos_sim)):
            if cos_sim[i, j] > threshold:
                edges.append((i, j))
                edges.append((j, i))


    u, v = zip(*edges)
    graph_cosine = dgl.graph((u, v), num_nodes=graph.num_nodes())

    graph_cosine.ndata['feature'] = graph.ndata['feature']
    graph_cosine.ndata['label'] = graph.ndata['label']
    graph_cosine.ndata['train_mask'] = graph.ndata['train_mask']
    graph_cosine.ndata['val_mask'] = graph.ndata['val_mask']
    graph_cosine.ndata['test_mask'] = graph.ndata['test_mask']
    graph_cosine.ndata['mark'] = graph.ndata['mark']

    return graph_cosine

def rw_rq():
    walk_length = 3
    num_walks = 5
    start_nodes = torch.tensor([0] * num_walks)
    walks = dgl.sampling.random_walk(graph, start_nodes, length=walk_length)

    edges = []
    for walk in walks:
        edges += [(walk[i], walk[i+1]) for i in range(len(walk) - 1)]

    u, v = zip(*edges)
    rw_graph = dgl.graph((u, v), num_nodes=graph.num_nodes())

    rw_graph.ndata['feature'] = graph.ndata['feature']
    rw_graph.ndata['label'] = graph.ndata['label']
    rw_graph.ndata['train_mask'] = graph.ndata['train_mask']
    rw_graph.ndata['val_mask'] = graph.ndata['val_mask']
    rw_graph.ndata['test_mask'] = graph.ndata['test_mask']
    rw_graph.ndata['mark'] = graph.ndata['mark']

    return rw_graph

def random_graph():
    num_nodes = graph.num_nodes()
    num_edges = 2000  # 边数

    edges_u = torch.tensor([random.randint(0, num_nodes - 1) for _ in range(num_edges)])
    edges_v = torch.tensor([random.randint(0, num_nodes - 1) for _ in range(num_edges)])

    random_graph = dgl.graph((edges_u, edges_v), num_nodes=graph.num_nodes())

    random_graph.ndata['feature'] = graph.ndata['feature']
    random_graph.ndata['label'] = graph.ndata['label']
    random_graph.ndata['train_mask'] = graph.ndata['train_mask']
    random_graph.ndata['val_mask'] = graph.ndata['val_mask']
    random_graph.ndata['test_mask'] = graph.ndata['test_mask']
    random_graph.ndata['mark'] = graph.ndata['mark']

    u, v = random_graph.edges()
    u = u.numpy()
    v = v.numpy()
    adj = random_graph.adj()

    # degree = adj.sum(dim=1).to(torch.float32)
    # degree_inv_sqrt = degree.pow(-0.5)  # 计算度的倒数平方根
    # degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0  # 处理无穷值
    # deg_inv_sqrt = torch.diag(degree_inv_sqrt)
    # laplacian = deg_inv_sqrt @ adj.to_dense() @ deg_inv_sqrt
    #
    # eigenvalues, _ = torch.linalg.eig(laplacian)
    # real_eigenvalues = eigenvalues.real
    # min_eigenvalue = real_eigenvalues.min()

    row_sum = adj.sum(dim=1)
    deg_matrix = torch.diag(row_sum)

    laplacian = deg_matrix - adj.to_dense()
    eigenvalues = torch.linalg.eigvals(laplacian)
    real_eigenvalues = eigenvalues.real
    real_eigenvalues = real_eigenvalues[real_eigenvalues > 0]
    entropy = -torch.sum(real_eigenvalues * torch.log(real_eigenvalues))


    return random_graph


def ba_graph_nx():
    n, m =100, 1
    anomaly_ratio = 0.1
    sigma = 2
    feature_dim = 10

    graph = nx.random_graphs.barabasi_albert_graph(n, m)

    num_anomalies = int(n * anomaly_ratio)
    anomaly_nodes = np.random.choice(graph.nodes, num_anomalies, replace=False)

    features = {}
    for node in graph.nodes():
        if node in anomaly_nodes:
            features[node] = np.random.normal(1, sigma, feature_dim)
        else:
            features[node] = np.random.normal(1, 1, feature_dim)

    nx.set_node_attributes(graph, features, "feature")

    return graph, anomaly_nodes

def ba_graph():
    n, m =100, 10
    anomaly_ratio = 0.1
    sigma = 2
    feature_dim = 10

    graph = dgl.BAShapeDataset(n, 3)

    return graph

def rd_graph(num_nodes=500, num_edges=1000, anomaly_ratio=0.2, hetero_ratio=0.2, homo_edge_anomaly_ratio=0.5, sigma_anomaly=2, feature_dim=1):
    # 参数设置
    # num_nodes = 500  # 总节点数
    # num_edges = 1000  # 总边数
    # anomaly_ratio = 0.2  # 异常节点占比
    # hetero_ratio = 0.2  # 异质边比例
    # sigma_anomaly = 2  # 异常节点的标准差
    # feature_dim = 1

    # 生成节点索引
    num_anomalies = int(num_nodes * anomaly_ratio)
    anomaly_nodes = set(random.sample(range(num_nodes), num_anomalies))  # 随机选取异常节点
    normal_nodes = set(range(num_nodes)) - anomaly_nodes

    # 生成节点特征
    features = []
    for node in range(num_nodes):
        if node in anomaly_nodes:
            # features[node] = np.random.normal(1, sigma_anomaly, size=10)  # 10维特征
            features.append(np.random.normal(1, sigma_anomaly, size=feature_dim))
        else:
            # features[node] = np.random.normal(1, 1, size=10)
            features.append(np.random.normal(1, 1, size=feature_dim))

    # 生成边
    num_hetero_edges = int(num_edges * hetero_ratio)  # 异质边数
    num_homo_edges = num_edges - num_hetero_edges  # 同质边数
    edges = set()

    # 生成异质边（正常-异常）
    while len(edges) < num_hetero_edges:
        u = random.choice(list(normal_nodes))
        v = random.choice(list(anomaly_nodes))
        if (u, v) not in edges and (v, u) not in edges:
            edges.add((u, v))

    # 生成同质边（异常-异常）
    target_anomaly_edges = int(num_homo_edges * homo_edge_anomaly_ratio)
    for i in [0, target_anomaly_edges]:
        if len(anomaly_nodes) >= 2:
            u, v = random.sample(anomaly_nodes, 2)
            edges.add((u, v))
        else:
            continue

    # 生成同质边（正常-正常）
    while len(edges) < num_edges:
        u, v = random.sample(normal_nodes, 2)
        edges.add((u, v))

    new_edges = edges.copy()
    for u, v in edges:
        new_edges.add((v, u))
    edges = new_edges

    rd_g = dgl.graph(list(edges), num_nodes=num_nodes)
    rd_g.ndata['feature'] = torch.tensor(np.array(features))
    return rd_g


def amazon_graph(sample=False, sample_num_nodes=500, num_edges=1000, anomaly_ratio=0.2, hetero_ratio=0.2, homo_edge_anomaly_ratio=0.5, homo=True, graph=None):
    # dataset = FraudAmazonDataset()
    # graph = dataset[0]
    if graph is None:
        if homo:
            graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
            graph = dgl.add_self_loop(graph)
        else:
            graph = dataset[0]

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()

        if sample:
            num_nodes = graph.num_nodes()
            sampled_nodes = np.random.choice(num_nodes, sample_num_nodes, replace=False)
            graph = dgl.node_subgraph(graph, sampled_nodes)

    num_nodes = graph.num_nodes()

    labels = graph.ndata['label']
    pos_nodes = set(np.where(labels == 1)[0])
    neg_nodes = set(np.where(labels == 0)[0])

    # 随机选择节点 控制异常节点数量
    #####



    # 生成节点索引
    num_anomalies = int(num_nodes * anomaly_ratio)
    anomaly_nodes = set(random.sample(range(num_nodes), num_anomalies))  # 随机选取异常节点
    normal_nodes = set(range(num_nodes)) - anomaly_nodes


    # 生成边 控制异构边数量
    num_hetero_edges = int(num_edges * hetero_ratio)  # 异质边数
    num_homo_edges = num_edges - num_hetero_edges  # 同质边数
    edges = set()

    # 生成异质边（正常-异常）
    while len(edges) < num_hetero_edges:
        u = random.choice(list(neg_nodes))
        v = random.choice(list(pos_nodes))
        if (u, v) not in edges and (v, u) not in edges:
            edges.add((u, v))

    # 生成同质边（异常-异常）
    target_anomaly_edges = int(num_homo_edges * homo_edge_anomaly_ratio)
    for i in [0, target_anomaly_edges]:
        if len(pos_nodes) >= 2:
            u, v = random.sample(pos_nodes, 2)
            edges.add((u, v))
        else:
            continue

    # 生成同质边（正常-正常）
    while len(edges) < num_edges:
        u, v = random.sample(neg_nodes, 2)
        edges.add((u, v))

    new_edges = edges.copy()
    for u, v in edges:
        new_edges.add((v, u))
    edges = new_edges

    g = dgl.graph(list(edges), num_nodes=num_nodes)

    g.ndata['feature'] = graph.ndata['feature']
    g.ndata['label'] = graph.ndata['label']
    g.ndata['train_mask'] = graph.ndata['train_mask']
    g.ndata['val_mask'] = graph.ndata['val_mask']
    g.ndata['test_mask'] = graph.ndata['test_mask']
    return g


def amazon_graph_random_drop(sample=True, sample_num_nodes=1000, homo_pos_drop_num=100, homo_neg_drop_num=1000, heter_drop_ratio=0, graph=None, homo=True):
    if graph is None:
        if homo:
            graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
            graph = dgl.add_self_loop(graph)
        else:
            # graph = dataset[0]
            graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()

        if sample:
            num_nodes = graph.num_nodes()
            sampled_nodes = np.random.choice(num_nodes, sample_num_nodes, replace=False)
            graph = dgl.node_subgraph(graph, sampled_nodes)


    edges = graph.edges()

    labels = graph.ndata['label']
    pos_nodes = set(np.where(labels == 1)[0])
    neg_nodes = set(np.where(labels == 0)[0])

    homo_pos_edges = set()
    homo_neg_edges = set()
    heter_edges = set()

    us = np.array(edges[0])
    vs = np.array(edges[1])

    for (u, v) in zip(us, vs):
        if u in pos_nodes:
            if v in pos_nodes:
                homo_pos_edges.add((u, v))
            else:
                heter_edges.add((u, v))
        else:
            if v in pos_nodes:
                heter_edges.add((u, v))
            else:
                homo_neg_edges.add((u, v))


    homo_edge_num = len(homo_pos_edges)+len(homo_neg_edges)
    heter_edge_num = len(heter_edges)

    # homo_drop_num = homo_edge_num * homo_drop_ratio
    # homo_pos_drop_num = int(homo_drop_num * homo_pos_drop_ratio)
    # homo_neg_drop_num = int(homo_drop_num - homo_pos_drop_num)
    heter_drop_num = int(heter_edge_num * heter_drop_ratio)

    edges_to_remove = random.sample(homo_pos_edges, homo_pos_drop_num)
    for (u, v) in edges_to_remove:
        homo_pos_edges.discard((u, v))
        homo_pos_edges.discard((v, u))

        u = random.choice(list(neg_nodes))
        v = random.choice(list(pos_nodes))
        if (u, v) not in edges and (v, u) not in edges:
            heter_edges.add((u, v))


    edges_to_remove = random.sample(homo_neg_edges, homo_neg_drop_num)
    for (u, v) in edges_to_remove:
        homo_neg_edges.discard((u, v))
        homo_neg_edges.discard((v, u))

        u = random.choice(list(neg_nodes))
        v = random.choice(list(pos_nodes))
        if (u, v) not in edges and (v, u) not in edges:
            heter_edges.add((u, v))

    edges_to_remove = random.sample(heter_edges, heter_drop_num)
    for (u, v) in edges_to_remove:
        heter_edges.discard((u, v))
        heter_edges.discard((v, u))

        if random.random() > 0.5:
            u = random.choice(list(pos_nodes))
            v = random.choice(list(pos_nodes))
            if (u, v) not in edges and (v, u) not in edges:
                homo_pos_edges.add((u, v))
        else:
            u = random.choice(list(neg_nodes))
            v = random.choice(list(neg_nodes))
            if (u, v) not in edges and (v, u) not in edges:
                homo_neg_edges.add((u, v))

    edges = list(homo_pos_edges | homo_neg_edges | heter_edges)

    g = dgl.graph(edges, num_nodes=graph.num_nodes())
    g.ndata['feature'] = graph.ndata['feature']
    g.ndata['label'] = graph.ndata['label']
    g.ndata['train_mask'] = graph.ndata['train_mask']
    g.ndata['val_mask'] = graph.ndata['val_mask']
    g.ndata['test_mask'] = graph.ndata['test_mask']
    return g


def amazon_graph_random_drop_v2(sample=True, sample_num_nodes=1000, homo_drop_ratio=0, heter_drop_ratio=0, graph=None, homo=True):
    if graph is None:
        if homo:
            graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
            graph = dgl.add_self_loop(graph)
        else:
            # graph = dataset[0]
            graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()

        if sample:
            num_nodes = graph.num_nodes()
            sampled_nodes = np.random.choice(num_nodes, sample_num_nodes, replace=False)
            graph = dgl.node_subgraph(graph, sampled_nodes)


    edges = graph.edges()

    labels = graph.ndata['label']
    pos_nodes = set(np.where(labels == 1)[0])
    neg_nodes = set(np.where(labels == 0)[0])

    homo_edges = set()
    heter_edges = set()

    us = np.array(edges[0])
    vs = np.array(edges[1])

    for (u, v) in zip(us, vs):
        if u in pos_nodes:
            if v in pos_nodes:
                homo_edges.add((u, v))
            else:
                heter_edges.add((u, v))
        else:
            if v in pos_nodes:
                heter_edges.add((u, v))
            else:
                homo_edges.add((u, v))


    homo_edge_num = len(homo_edges)
    heter_edge_num = len(heter_edges)

    homo_drop_num = int(homo_edge_num * homo_drop_ratio)
    heter_drop_num = int(heter_edge_num * heter_drop_ratio)

    edges_to_remove = random.sample(homo_edges, homo_drop_num)
    for (u, v) in edges_to_remove:
        homo_edges.discard((u, v))
        homo_edges.discard((v, u))

    edges_to_remove = random.sample(heter_edges, heter_drop_num)
    for (u, v) in edges_to_remove:
        heter_edges.discard((u, v))
        heter_edges.discard((v, u))

    edges = list(homo_edges | heter_edges)

    g = dgl.graph(edges, num_nodes=graph.num_nodes())
    g.ndata['feature'] = graph.ndata['feature']
    g.ndata['label'] = graph.ndata['label']
    g.ndata['train_mask'] = graph.ndata['train_mask']
    g.ndata['val_mask'] = graph.ndata['val_mask']
    g.ndata['test_mask'] = graph.ndata['test_mask']
    return g


def amazon_graph_random_drop_v3(sample=True, sample_num_nodes=1000, homo_drop_ratio=0, heter_drop_ratio=0, graph=None, homo=True):
    """
    generate new edge after dropping
    if drop homo generate heter
    else ^
    """
    if graph is None:
        if homo:
            graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
            graph = dgl.add_self_loop(graph)
        else:
            graph = dataset[0]
            # graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
            return graph

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()

        if sample:
            num_nodes = graph.num_nodes()
            sampled_nodes = np.random.choice(num_nodes, sample_num_nodes, replace=False)
            graph = dgl.node_subgraph(graph, sampled_nodes)


    edges = graph.edges()

    labels = graph.ndata['label']
    pos_nodes = set(np.where(labels == 1)[0])
    neg_nodes = set(np.where(labels == 0)[0])

    homo_edges = set()
    heter_edges = set()

    us = np.array(edges[0])
    vs = np.array(edges[1])

    for (u, v) in zip(us, vs):
        if u in pos_nodes:
            if v in pos_nodes:
                homo_edges.add((u, v))
            else:
                heter_edges.add((u, v))
        else:
            if v in pos_nodes:
                heter_edges.add((u, v))
            else:
                homo_edges.add((u, v))


    homo_edge_num = len(homo_edges)
    heter_edge_num = len(heter_edges)

    homo_drop_num = int(homo_edge_num * homo_drop_ratio)
    heter_drop_num = int(heter_edge_num * heter_drop_ratio)

    edges_to_remove = random.sample(homo_edges, homo_drop_num)
    for (u, v) in edges_to_remove:
        homo_edges.discard((u, v))
        homo_edges.discard((v, u))



    edges_to_remove = random.sample(heter_edges, heter_drop_num)
    for (u, v) in edges_to_remove:
        heter_edges.discard((u, v))
        heter_edges.discard((v, u))

    edges = list(homo_edges | heter_edges)

    g = dgl.graph(edges, num_nodes=graph.num_nodes())
    g.ndata['feature'] = graph.ndata['feature']
    g.ndata['label'] = graph.ndata['label']
    g.ndata['train_mask'] = graph.ndata['train_mask']
    g.ndata['val_mask'] = graph.ndata['val_mask']
    g.ndata['test_mask'] = graph.ndata['test_mask']
    return g


def amazon_temporal_graph():
    graph = dataset[0]

#
# dgl.save_graphs('./amazon/org_graph.dgl', original_graph)
# dgl.save_graphs('./amazon/edknn_graph.dgl', euclidean_dis_knn())
# dgl.save_graphs('./amazon/cos_graph.dgl', cos_similarity_knn())
# dgl.save_graphs('./amazon/rw_graph.dgl', random_graph())
# ba_graph_nx()
# rd_graph()
# amazon_graph(True)
# amazon_graph_random_drop()
