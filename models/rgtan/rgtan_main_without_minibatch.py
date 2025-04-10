import os
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.dataloading import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from scipy.io import loadmat
from tqdm import tqdm
from . import *
from .rgtan_lpa import load_lpa_subtensor
from .rgtan_model import RGTAN


def rgtan_main(feat_df, graph, train_idx, val_idx, test_idx, labels, args, cat_features, neigh_features: pd.DataFrame, nei_att_head):
    device = args['device']
    graph = graph.to(device)

    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    cat_feat = {col: torch.from_numpy(feat_df[col].values).long().to(device) for col in cat_features}

    neigh_padding_dict = {}
    nei_feat = {}
    if isinstance(neigh_features, pd.DataFrame):
        nei_feat = {col: torch.from_numpy(neigh_features[col].values).float().to(device) for col in neigh_features.columns}

    labels_tensor = torch.from_numpy(labels.values).long().to(device)

    # 准备训练/验证索引
    trn_ind = torch.tensor(train_idx, dtype=torch.long).to(device)
    val_ind = torch.tensor(val_idx, dtype=torch.long).to(device)

    # 构建模型
    model = RGTAN(
        in_feats=feat_df.shape[1],
        hidden_dim=args['hid_dim'] // 4,
        n_classes=2,
        heads=[4] * args['n_layers'],
        activation=nn.PReLU(),
        n_layers=args['n_layers'],
        drop=args['dropout'],
        device=device,
        gated=args['gated'],
        ref_df=feat_df,
        cat_features=cat_feat,
        neigh_features=nei_feat,
        nei_att_head=nei_att_head
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    scheduler = MultiStepLR(optimizer, milestones=[4000, 12000], gamma=0.3)
    loss_fn = nn.CrossEntropyLoss()

    earlystoper = early_stopper(patience=args['early_stopping'], verbose=True)

    # Full-batch training loop
    for epoch in range(args['max_epochs']):
        model.train()

        # 获取子图（全图或者仅邻居采样）
        sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        train_dataloader = NodeDataLoader(graph, trn_ind, sampler, device=device, batch_size=len(trn_ind),
                                          shuffle=False, drop_last=False, num_workers=0)
        input_nodes, seeds, blocks = next(iter(train_dataloader))
        blocks = [block.to(device) for block in blocks]

        batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
            num_feat, cat_feat, nei_feat, neigh_padding_dict, labels_tensor,
            seeds, input_nodes, device, blocks
        )

        mask = batch_labels == 2
        batch_labels = batch_labels[~mask]
        output = model(blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)[~mask]

        loss = loss_fn(output, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            model.eval()
            val_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
            val_dataloader = NodeDataLoader(graph, val_ind, val_sampler, device=device, batch_size=len(val_ind),
                                            shuffle=False, drop_last=False, num_workers=0)
            input_nodes, seeds, blocks = next(iter(val_dataloader))
            blocks = [block.to(device) for block in blocks]

            batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                num_feat, cat_feat, nei_feat, neigh_padding_dict, labels_tensor,
                seeds, input_nodes, device, blocks
            )

            mask = batch_labels == 2
            batch_labels = batch_labels[~mask]
            output = model(blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)[~mask]

            val_loss = loss_fn(output, batch_labels)
            val_score = torch.softmax(output, dim=1)[:, 1].cpu().numpy()

            print(f"[Epoch {epoch}] Train loss: {loss.item():.4f}, Val loss: {val_loss.item():.4f}, "
                  f"Val AP: {average_precision_score(batch_labels.cpu().numpy(), val_score):.4f}, "
                  f"Val AUC: {roc_auc_score(batch_labels.cpu().numpy(), val_score):.4f}")

            earlystoper.earlystop(val_loss.item(), model)
            if earlystoper.is_earlystop:
                print("Early stopping triggered.")
                break

    print("Best val_loss is: {:.7f}".format(earlystoper.best_cv))

    # ====== Test ======
    test_ind = torch.tensor(test_idx, dtype=torch.long).to(device)
    test_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
    test_dataloader = NodeDataLoader(graph, test_ind, test_sampler, device=device, batch_size=len(test_ind),
                                     shuffle=False, drop_last=False, num_workers=0)
    b_model = earlystoper.best_model.to(device)
    b_model.eval()

    with torch.no_grad():
        input_nodes, seeds, blocks = next(iter(test_dataloader))
        blocks = [block.to(device) for block in blocks]

        batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
            num_feat, cat_feat, nei_feat, neigh_padding_dict, labels_tensor,
            seeds, input_nodes, device, blocks
        )

        mask = batch_labels == 2
        batch_labels = batch_labels[~mask]
        output = b_model(blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)[~mask]
        test_score = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
        test_pred = torch.argmax(output, dim=1).cpu().numpy()
        y_true = batch_labels.cpu().numpy()

        print("Test AUC:", roc_auc_score(y_true, test_score))
        print("Test f1:", f1_score(y_true, test_pred, average="macro"))
        print("Test AP:", average_precision_score(y_true, test_score))


def loda_rgtan_data(dataset: str, test_size: float):
    # prefix = "./antifraud/data/"
    prefix = "data/"
    if dataset == 'S-FFSD':
        cat_features = ["Target", "Location", "Type"]

        
        df = pd.read_csv(prefix + "S-FFSDneofull.csv")
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        #####
        neigh_features = []
        #####
        data = df[df["Labels"] <= 2]
        data = data.reset_index(drop=True)
        out = []
        alls = []
        allt = []
        pair = ["Source", "Target", "Location", "Type"]
        for column in pair:
            src, tgt = [], []
            edge_per_trans = 3
            for c_id, c_df in tqdm(data.groupby(column), desc=column):
                c_df = c_df.sort_values(by="Time")
                df_len = len(c_df)
                sorted_idxs = c_df.index
                src.extend([sorted_idxs[i] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
                tgt.extend([sorted_idxs[i+j] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
            alls.extend(src)
            allt.extend(tgt)
        alls = np.array(alls)
        allt = np.array(allt)
        g = dgl.graph((alls, allt))
        cal_list = ["Source", "Target", "Location", "Type"]
        for col in cal_list:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].apply(str).values)
        feat_data = data.drop("Labels", axis=1)
        labels = data["Labels"]

        #######
        g.ndata['label'] = torch.from_numpy(
            labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        #######

        graph_path = prefix+"graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])
        index = list(range(len(labels)))

        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=0.6,
                                                                random_state=2, shuffle=True)
        feat_neigh = pd.read_csv(
            prefix + "S-FFSD_neigh_feat.csv")
        print("neighborhood feature loaded for nn input.")
        neigh_features = feat_neigh

    elif dataset == 'yelp':
        cat_features = []
        neigh_features = []
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        # load the preprocessed adj_lists
        with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=test_size,
                                                                random_state=2, shuffle=True)
        src = []
        tgt = []
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)
        src = np.array(src)
        tgt = np.array(tgt)
        g = dgl.graph((src, tgt))
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])

        try:
            feat_neigh = pd.read_csv(
                prefix + "yelp_neigh_feat.csv")
            print("neighborhood feature loaded for nn input.")
            neigh_features = feat_neigh
        except:
            print("no neighbohood feature used.")

    elif dataset == 'amazon':
        cat_features = []
        neigh_features = []
        data_file = loadmat(prefix + 'Amazon.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        # load the preprocessed adj_lists
        with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(3305, len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                                test_size=test_size, random_state=2, shuffle=True)
        src = []
        tgt = []
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)
        src = np.array(src)
        tgt = np.array(tgt)
        g = dgl.graph((src, tgt))
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])
        try:
            feat_neigh = pd.read_csv(
                prefix + "amazon_neigh_feat.csv")
            print("neighborhood feature loaded for nn input.")
            neigh_features = feat_neigh
        except:
            print("no neighbohood feature used.")

    return feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features

args = parse_args()

feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features = rgtan_graph('amazon', 0.2, True)
train_idx, val_idx = train_test_split(train_idx, stratify=labels[train_idx], test_size=0.1,
                                                        random_state=2, shuffle=True)
rgtan_main(feat_data, g, train_idx, test_idx, labels, args, cat_features, neigh_features, nei_att_head=args['nei_att_heads'][args['dataset']])

