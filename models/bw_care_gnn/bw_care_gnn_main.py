import argparse
import time
import random

import dgl
import numpy as np
import torch
from dgl.data import FraudAmazonDataset
from sklearn.model_selection import train_test_split
from torch import nn
from torch.autograd import Variable

from bw_care_gnn_model import InterAgg, OneLayerCARE_BW
from models.caregnn.caregnn_layers import IntraAgg
from models.caregnn.caregnn_utils import *


def to_adj(relation, labels):
    (us, vs) = relation
    us = us.numpy()
    vs = vs.numpy()

    reviews = defaultdict(set)
    for i in range(len(labels)):
        reviews[i] = set()

    for (u, v) in zip(us, vs):
        # if u not in reviews.keys():
        # reviews[u] = set()
        reviews[u].add(v)
    return reviews

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BWGNN')
    # bw
    parser.add_argument("--dataset", type=str, default="amazon",
                        help="Dataset for this model (yelp/amazon/tfinance/tsocial)")
    parser.add_argument("--train_ratio", type=float, default=0.01, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    # parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    # parser.add_argument("--run", type=int, default=5, help="Running times")

    # care
    # parser.add_argument('--data', type=str, default='amazon', help='The dataset name. [yelp, amazon]')
    parser.add_argument('--model', type=str, default='CARE', help='The model name. [CARE, SAGE]')
    parser.add_argument('--inter', type=str, default='GNN',
                        help='The inter-relation aggregator type. [Att, Weight, Mean, GNN]')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size 1024 for yelp, 256 for amazon.')

    # hyper-parameters
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--lambda_1', type=float, default=2, help='Simi loss weight.')
    parser.add_argument('--lambda_2', type=float, default=1e-3, help='Weight decay (L2 loss weight).')
    parser.add_argument('--emb-size', type=int, default=64, help='Node embedding size at the last layer.')
    parser.add_argument('--num-epochs', type=int, default=31, help='Number of epochs.')
    parser.add_argument('--test-epochs', type=int, default=3, help='Epoch interval to run test set.')
    parser.add_argument('--under-sample', type=int, default=1, help='Under-sampling scale.')
    parser.add_argument('--step-size', type=float, default=2e-2, help='RL action step size')
    parser.add_argument('--run', type=float, default=5, help='run times')

    # other args
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')

    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset
    homo = args.homo
    order = args.order
    h_feats = args.hid_dim
    # graph = Dataset(dataset_name, homo).graph
    # in_feats = graph.ndata['feature'].shape[1]
    num_classes = 2

    final_f1s, final_accs, final_recalls, final_aucs, final_aps = [], [], [], [], []
    for tt in range(args.run):
        graph = FraudAmazonDataset()[0]
        graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        feat_data = graph.ndata['feature']
        in_feats = feat_data.shape[1]
        labels = graph.ndata['label']
        labels = np.array(labels)

        homo_edges = graph.edges()
        homo = to_adj(homo_edges, labels)

        np.random.seed(args.seed)
        random.seed(args.seed)

        index = list(range(3305, len(labels)))
        idx_train, idx_test, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                                test_size=0.60, random_state=2, shuffle=True)

        # split pos neg sets for under-sampling
        train_pos, train_neg = pos_neg_split(idx_train, y_train)

        # initialize model input
        features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
        feat_data = normalize(feat_data)
        features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
        if args.cuda:
            features.cuda()

        adj_lists = [homo]

        intra1 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
        inter1 = InterAgg(features, feat_data.shape[1], args.emb_size, adj_lists, [intra1], inter=args.inter,
                         step_size=args.step_size, cuda=args.cuda, num_classes=2, graph=graph, d=order, batch=False)

        weight = (1 - labels).sum().item() / labels.sum().item()
        model = OneLayerCARE_BW(2, inter1, args.lambda_1, weight)

        if args.cuda:
            model.cuda()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.lambda_2)

        performance_log = []

        for epoch in range(args.num_epochs):
            # randomly under-sampling negative nodes for each epoch
            # sampled_idx_train = undersample(train_pos, train_neg, scale=1)
            # random.shuffle(sampled_idx_train)

            # send number of batches to model to let the RLModule know the training progress
            num_batches = int(len(idx_train) / args.batch_size) + 1
            if args.model == 'CARE':
                inter1.batch_num = num_batches

            loss = 0.0

            start_time = time.time()
            # batch_label = labels[np.array(idx_train)]
            optimizer.zero_grad()
            if args.cuda:
                loss = model.loss(idx_train, Variable(torch.cuda.LongTensor(y_train)))
            else:
                loss = model.loss(idx_train, Variable(torch.LongTensor(y_train)))
            loss.backward()
            optimizer.step()
            end_time = time.time() - start_time

            print(f'Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {end_time}s')

            # testing the model for every $test_epoch$ epoch
            if epoch % args.test_epochs == 0:
                if args.model == 'SAGE':
                    test_sage(idx_test, y_test, model, args.batch_size)
                else:
                    # gnn_auc, label_auc, gnn_recall, label_recall = test_care(idx_test, y_test, gnn_model, args.batch_size)
                    gnn_auc, label_auc, gnn_recall, label_recall, gnn_f1, gnn_acc, gnn_ap = test_care(idx_test, y_test, model, args.batch_size)
                    performance_log.append([gnn_auc, label_auc, gnn_recall, label_recall])

        final_f1s.append(gnn_f1)
        final_accs.append(gnn_acc)
        final_recalls.append(gnn_recall)
        final_aucs.append(gnn_auc)
        final_aps.append(gnn_ap)

    final_f1s = np.array(final_f1s)
    final_accs = np.array(final_accs)
    final_recalls = np.array(final_recalls)
    final_aucs = np.array(final_aucs)
    final_aps = np.array(final_aps)
    print('MF1-mean: {:.2f}, MF1-std: {:.2f}, ACC-mean: {:.2f}, ACC-std: {:.2f}, '
          'RECALL-mean: {:.2f}, RECALL-std: {:.2f}, AUC-mean: {:.2f}, AUC-std: {:.2f}, '
          'AP-mean: {:.2f}, AP-std: {:.2f}'.format(100 * np.mean(final_f1s),
                                                   100 * np.std(final_f1s),
                                                   100 * np.mean(final_accs),
                                                   100 * np.std(final_accs),
                                                   100 * np.mean(final_recalls),
                                                   100 * np.std(final_recalls),
                                                   100 * np.mean(final_aucs),
                                                   100 * np.std(final_aucs),
                                                   100 * np.mean(final_aps),
                                                   100 * np.std(final_aps),))
