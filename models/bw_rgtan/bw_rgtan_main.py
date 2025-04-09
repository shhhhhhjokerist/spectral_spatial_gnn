import argparse
import time

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from dgl.dataloading import MultiLayerFullNeighborSampler, DataLoader
from sklearn.model_selection import train_test_split

from bw_rgtan_model import bw_rgtan_model
from models.bwgnn.bwgnn_dataset import Dataset
from models.bwgnn.bwgnn_models import BWGNN, BWGNN_Hetero


def train(graph, args, cat_features, neigh_features):
    device = args.device
    graph = graph.to(device)

    feature = graph.ndata['feature']
    labels = graph.ndata['label']

    index = list(range(len(labels)))

    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                            train_size=args.train_ratio,
                                                            random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=2, shuffle=True)
    

    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()

    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1


    sampler = MultiLayerFullNeighborSampler(2)  # 2 layers for GNN
    dataloader = DataLoader(graph, torch.where(train_mask)[0].tolist(), sampler, device=device, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # cat_features = args['cat_feat']
    cat_feat = {col: torch.from_numpy(feature[col].values).long().to(
        device) for col in cat_features}
    # neigh_features = args['neigh_feat']
    nei_feat = None
    if isinstance(neigh_features, pd.DataFrame):  # otherwise []
        # if null it is []
        nei_feat = {col: torch.from_numpy(neigh_features[col].values).to(torch.float32).to(
            device) for col in neigh_features.columns}
    nei_att_head = args.nei_att_heads

    # model = bw_rgtan_model(in_feats=in_feats, hidden_dim=h_feats, n_classes=num_classes, heads=[4]*args.n_layers, activation=torch.nn.PReLU(), graph=graph, h_feats=h_feats, d=order, batch=True,  n_layers=args.n_layers, drop=args.dropout, device=device, gated=args.gated, ref_df=feature[idx_train], cat_features=cat_feat, neigh_features=nei_feat, nei_att_head=nei_att_head).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.
    best_model = None

    weight = (1 - labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)
    time_start = time.time()

    for e in range(args.epoch):
        model.train()
        for input_nodes, output_nodes, mfg in dataloader:
            batch_features = mfg.ndata['feature']
            batch_labels = labels[output_nodes]

            optimizer.zero_grad()

            # Forward pass
            logits = model(batch_features)

            # Calculate loss using cross-entropy
            loss = F.cross_entropy(logits, batch_labels, weight=torch.tensor([1., weight]))
            loss.backward()
            optimizer.step()
    model.eval()








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BW_RGTAN')
    parser.add_argument("--dataset", type=str, default="amazon",
                        help="Dataset for this model (yelp/amazon/tfinance/tsocial)")
    parser.add_argument("--train_ratio", type=float, default=0.01, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=5, help="Running times")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--device", type=str, default='cpu', help="cuda")
    parser.add_argument("--nei_att_heads", type=int, default=5, help="")
    parser.add_argument("--n_layers", type=int, default=2, help="")
    parser.add_argument("--dropout", type=list, default=[0.2, 0.1], help="")
    parser.add_argument("--gated", type=bool, default=True, help="")

    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset
    homo = args.homo
    order = args.order
    h_feats = args.hid_dim
    graph = Dataset(dataset_name, homo, sample=False).graph
    in_feats = graph.ndata['feature'].shape[1]
    num_classes = 2
    # args['cat_feat'] = graph.ndata['cat_feat']
    # args['neigh_feat'] = graph.ndata['neigh_feat']
    cat_feat = []
    neigh_feat = []



    if args.run == 1:
        train(graph, args, cat_feat, neigh_feat)
    else:
        final_mf1s, final_aucs = [], []
        for tt in range(args.run):
            mf1, auc = train(graph, args, cat_feat, neigh_feat)
            final_mf1s.append(mf1)
            final_aucs.append(auc)

        final_mf1s = np.array(final_mf1s)
        final_aucs = np.array(final_aucs)
        print('MF1-mean: {:.2f}, MF1-std: {:.2f}, AUC-mean: {:.2f}, AUC-std: {:.2f}'.format(100 * np.mean(final_mf1s),
                                                                                            100 * np.std(final_mf1s),
                                                               100 * np.mean(final_aucs), 100 * np.std(final_aucs)))
