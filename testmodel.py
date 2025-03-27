import pickle

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from dgl.data import CoraGraphDataset


class GCN(nn.Module):
    def __init__(self,
                 g,  # DGL的图对象
                 in_feats,  # 输入特征的维度
                 n_hidden,  # 隐层的特征维度
                 n_classes,  # 类别数
                 n_layers,  # 网络层数
                 activation,  # 激活函数
                 dropout  # dropout系数
                 ):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # 输入层
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # 隐层
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # 输出层
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)

        # 计算预测中类别0和类别1的占比
        # logits中每一行是一个样本的logits，使用softmax将其转化为概率
        prob = torch.nn.functional.softmax(logits, dim=1)  # 获取类别0和类别1的概率
        prob_0 = prob[:, 0]  # 类别0的概率
        prob_1 = prob[:, 1]  # 类别1的概率

        # 计算预测中类别0和类别1的占比
        num_0 = torch.sum(prob_0 > prob_1).item()
        num_1 = torch.sum(prob_1 > prob_0).item()

        total = len(prob_0)
        ratio_0 = num_0 / total
        ratio_1 = num_1 / total

        print(f'Ratio of class 0: {ratio_0:.4f}')
        print(f'Ratio of class 1: {ratio_1:.4f}')

        # 返回准确率
        return correct.item() * 1.0 / len(labels)

def similaritySample(g, threshold=0):
    num_edge = g.number_of_edges()
    src, dst = g.out_edges('homo', g.nodes())
    new_src, new_dst = [], []
    nums = []
    for i, (s, d) in enumerate(zip(src, dst)):
        x1 = g.ndata['feat'][s]
        x2 = g.ndata['feat'][d]
        dist = torch.norm(x1 - x2, dim=-1)  # 计算每对节点的距离
        if dist >= threshold:
            new_src.append(s)
            new_dst.append(d)
        else:
            nums.append(i)

    for num in reversed(nums):
        g.remove_edges(num, 'homo')

    new_num_edge = g.number_of_edges()
    g = dgl.add_self_loop(g)
    print('before:{} after:{}'.format(num_edge, new_num_edge))
    return g

def similaritySample2(g, threshold=0.5):
    num_edge = g.number_of_edges()
    src, dst = g.out_edges(g.nodes())
    # new_src, new_dst = [], []
    nums = []
    for i, (s, d) in enumerate(zip(src, dst)):
        x1 = g.ndata['feature'][s]
        x2 = g.ndata['feature'][d]
        dist = torch.norm(x1 - x2, dim=-1)  # 计算每对节点的距离
        # # 归一化特征
        # x1_normalized = (x1 - x1.min()) / (x1.max() - x1.min())
        # x2_normalized = (x2 - x2.min()) / (x2.max() - x2.min())
        #
        # # 计算归一化后的欧氏距离
        # dist = torch.norm(x1_normalized - x2_normalized, dim=-1)

        if dist >= threshold:
            nums.append(i)
        # else:
            # new_src.append(s)
            # new_dst.append(d)

    for num in reversed(nums):
        g.remove_edges(num)

    new_num_edge = g.number_of_edges()
    g = dgl.add_self_loop(g)
    print('before:{} after:{}'.format(num_edge, new_num_edge))
    return g

def train(dataset='yelp', n_epochs=100, lr=1e-2, weight_decay=5e-4, n_hidden=16, n_layers=1, activation=F.relu, dropout=0.5):
    # data = CoraGraphDataset()
    # g = data[0]  # 图的所有信息，包含2078个节点，每个节点1433维，所有节点可分为7类。10556条边。
    if dataset == 'yelp':
        data = dgl.load_graphs('./dataset/yelp.dgl')
        g = data[0][0]

        g = similaritySample(g)

        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']  # 0~139为训练节点
        val_mask = g.ndata['val_mask']  # 140~539为验证节点
        test_mask = g.ndata['test_mask']  # 1708-2707为测试节点
        in_feats = features.shape[1]
        n_classes = data.num_classes

        model = GCN(g,
                    in_feats,
                    n_hidden,
                    n_classes,
                    n_layers,
                    activation,
                    dropout)

        loss_fcn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)
        best_val_acc = 0
        for epoch in range(n_epochs):
            model.train()
            logits = model(features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = evaluate(model, features, labels, val_mask)
            print("Epoch {} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, loss.item(), acc))
            if acc > best_val_acc:
                best_val_acc = acc
                torch.save(model.state_dict(), 'models/best_model.pth')

        model.load_state_dict(torch.load("models/best_model.pth"))
        acc = evaluate(model, features, labels, test_mask)
        print("Test accuracy {:.2%}".format(acc))
    elif dataset == 'elliptic':
        data = dgl.load_graphs('./dataset/elliptic')
        g = data[0][0]


        g = similaritySample2(g, 3)

        features = g.ndata['feature']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']  # 0~139为训练节点
        val_mask = g.ndata['val_mask']  # 140~539为验证节点
        test_mask = g.ndata['test_mask']  # 1708-2707为测试节点
        in_feats = features.shape[1]
        # n_classes = data.num_classes
        n_classes = 2

        model = GCN(g,
                    in_feats,
                    n_hidden,
                    n_classes,
                    n_layers,
                    activation,
                    dropout)

        loss_fcn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)
        best_val_acc = 0
        for epoch in range(n_epochs):
            model.train()
            logits = model(features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = evaluate(model, features, labels, val_mask)
            print("Epoch {} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, loss.item(), acc))
            if acc > best_val_acc:
                best_val_acc = acc
                torch.save(model.state_dict(), 'models/best_model.pth')

        model.load_state_dict(torch.load("models/best_model.pth"))
        acc = evaluate(model, features, labels, test_mask)
        print("Test accuracy {:.2%}".format(acc))




if __name__ == '__main__':
    train('elliptic')