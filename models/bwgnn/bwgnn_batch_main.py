import dgl
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
from bwgnn_dataset import Dataset
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from bwgnn_models import *
from sklearn.model_selection import train_test_split
from dgl.dataloading import NeighborSampler, DataLoader, MultiLayerFullNeighborSampler


def train(model, g, args):
    features = g.ndata['feature']
    labels = g.ndata['label']
    index = np.arange(len(labels))

    # 划分训练、验证、测试集
    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels.numpy(), stratify=labels.numpy(),
                                                            train_size=args.train_ratio, random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67, random_state=2, shuffle=True)

    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    val_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)

    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1

    print('Train/Valid/Test samples:', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())

    # 采样器 (邻居采样，采样2层，每层最多采 10 个邻居)
    # sampler = NeighborSampler([10, 10])
    sampler = MultiLayerFullNeighborSampler(2)

    # DataLoader 进行 mini-batch 训练
    # train_dataloader = DataLoader(
    #     g, idx_train, sampler, batch_size=1024, shuffle=True, drop_last=False, num_workers=0
    # )
    idx_train = torch.tensor(idx_train, dtype=torch.long)
    train_dataloader = DataLoader(
        g, idx_train, sampler, use_ddp=False, batch_size=64, shuffle=True, drop_last=False, num_workers=0
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_f1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0.
    best_model = None

    weight = (1 - labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('Cross Entropy Weight:', weight)

    time_start = time.time()

    for e in range(args.epoch):
        model.train()
        total_loss = 0.

        # for input_nodes, output_nodes, blocks in train_dataloader:
        for step, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            batch_features = features[input_nodes]  # 获取输入节点的特征
            batch_labels = labels[output_nodes]    # 仅获取输出节点的标签

            logits = model(blocks, batch_features)  # 传入 blocks 计算
            loss = F.cross_entropy(logits, batch_labels, weight=torch.tensor([1., weight]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 评估模型
        model.eval()
        with torch.no_grad():
            logits = model(g, features)
            probs = logits.softmax(1)
            f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
            preds = np.zeros_like(labels.numpy())
            preds[probs[:, 1] > thres] = 1

            trec = recall_score(labels[test_mask].numpy(), preds[test_mask])
            tpre = precision_score(labels[test_mask].numpy(), preds[test_mask])
            tmf1 = f1_score(labels[test_mask].numpy(), preds[test_mask], average='macro')
            tauc = roc_auc_score(labels[test_mask].numpy(), probs[test_mask][:, 1].numpy())

            if best_f1 < f1:
                best_f1 = f1
                final_trec = trec
                final_tpre = tpre
                final_tmf1 = tmf1
                final_tauc = tauc
                best_model = model.state_dict()

        print('Epoch {}, Loss: {:.4f}, Val MF1: {:.4f}, (Best {:.4f})'.format(e, total_loss, f1, best_f1))

    time_end = time.time()
    print('Time cost: ', time_end - time_start, 's')
    print(
        'Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_trec * 100, final_tpre * 100, final_tmf1 * 100,
                                                                   final_tauc * 100))

    return final_tmf1, final_tauc


# 计算最佳 F1 阈值
def get_best_f1(labels, probs):
    best_f1, best_thres = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels.numpy())
        preds[probs[:, 1] > thres] = 1
        mf1 = f1_score(labels.numpy(), preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thres = thres
    return best_f1, best_thres


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument("--dataset", type=str, default="amazon", help="Dataset name")
    parser.add_argument("--train_ratio", type=float, default=0.01, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1, help="1 for Homo BWGNN, 0 for Hetero BWGNN")
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs")
    parser.add_argument("--run", type=int, default=5, help="Number of runs")

    args = parser.parse_args()
    print(args)

    dataset_name = args.dataset
    homo = args.homo
    order = args.order
    h_feats = args.hid_dim

    graph = Dataset(dataset_name, homo, sample=False).graph
    in_feats = graph.ndata['feature'].shape[1]
    num_classes = 2

    if args.run == 1:
        model = BWGNN(in_feats, h_feats, num_classes, graph, d=order) if homo else BWGNN_Hetero(in_feats, h_feats,
                                                                                                num_classes, graph,
                                                                                                d=order)
        train(model, graph, args)
    else:
        final_mf1s, final_aucs = [], []
        for _ in range(args.run):
            model = BWGNN(in_feats, h_feats, num_classes, graph, d=order) if homo else BWGNN_Hetero(in_feats, h_feats,
                                                                                                    num_classes, graph,
                                                                                                    d=order)
            mf1, auc = train(model, graph, args)
            final_mf1s.append(mf1)
            final_aucs.append(auc)

        final_mf1s = np.array(final_mf1s)
        final_aucs = np.array(final_aucs)
        print('MF1-mean: {:.2f}, MF1-std: {:.2f}, AUC-mean: {:.2f}, AUC-std: {:.2f}'.format(
            100 * np.mean(final_mf1s), 100 * np.std(final_mf1s),
            100 * np.mean(final_aucs), 100 * np.std(final_aucs)))
