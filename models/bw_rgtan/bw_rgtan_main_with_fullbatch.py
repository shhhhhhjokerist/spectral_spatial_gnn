import dgl
import torch
import torch.nn.functional as F
import numpy
import argparse
import time

from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix

from sklearn.model_selection import train_test_split

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

from models.bwgnn.bwgnn_dataset import Dataset
from models.bw_rgtan.bw_rgtan_models import BWGNN, BWGNN_Hetero, BW_RGTAN_AGG
from models.rgtan import early_stopper
from models.rgtan.rgtan_lpa import load_lpa_subtensor
from models.bw_rgtan.bw_rgtan_models import RGTAN


def train(args):
    # init bwgnn 
    dataset_name = args.dataset
    homo = args.homo
    order = args.order
    h_feats = args.hid_dim
    graph = Dataset(dataset_name, homo, sample=False).graph
    in_feats = graph.ndata['feature'].shape[1]
    num_classes = 2

    if homo:
        model_bw = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
    else:
        model_bw = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)

    features = graph.ndata['feature']
    labels = graph.ndata['label']
    index = list(range(len(labels)))
    if dataset_name == 'amazon':
        index = list(range(3305, len(labels)))

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
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    optimizer_bw = torch.optim.Adam(model_bw.parameters(), lr=0.01)
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.

    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)



    cat_features = []
    neigh_features = []
    nei_att_head = args.nei_att_head

    # init rgtan
    device = args.device
    graph = graph.to(device)
    oof_predictions = torch.from_numpy(
        np.zeros([len(features), 2])).float().to(device)
    test_predictions = torch.from_numpy(
        np.zeros([len(features), 2])).float().to(device)

    y_target = labels[idx_train].values
    # num_feat = torch.from_numpy(features.values).float().to(device)
    num_feat = features
    cat_feat = {col: torch.from_numpy(features[col].values).long().to(
        device) for col in cat_features}

    neigh_padding_dict = {}
    nei_feat = None
    if isinstance(neigh_features, pd.DataFrame):  # otherwise []
        # if null it is []
        nei_feat = {col: torch.from_numpy(neigh_features[col].values).to(torch.float32).to(
            device) for col in neigh_features.columns}
        
    # y = labels
    # labels = torch.from_numpy(y.values).long().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # # 假设你已经准备好了 idx_train, idx_valid, idx_test
    trn_ind = torch.from_numpy(np.array(idx_train)).long().to(device)
    val_ind = torch.from_numpy(np.array(idx_valid)).long().to(device)

    # 构建数据加载器
    train_sampler = MultiLayerFullNeighborSampler(args.n_layers)
    train_dataloader = DataLoader(graph,
                                trn_ind,
                                train_sampler,
                                device=device,
                                use_ddp=False,
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=False,
                                num_workers=0)

    val_sampler = MultiLayerFullNeighborSampler(args.n_layers)
    val_dataloader = DataLoader(graph,
                                val_ind,
                                val_sampler,
                                use_ddp=False,
                                device=device,
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=False,
                                num_workers=0)

    # 初始化模型
    model_rgtan = RGTAN(in_feats=features.shape[1],
                hidden_dim=args.hid_dim//4,
                # hidden_dim=args.hid_dim,
                n_classes=2,
                heads=[4]*args.n_layers,
                activation=nn.PReLU(),
                n_layers=args.n_layers,
                drop=args.dropout,
                device=device,
                gated=args.gated,
                ref_df=features,
                cat_features=cat_feat,
                neigh_features=nei_feat,
                nei_att_head=nei_att_head).to(device)

    lr = args.lr * np.sqrt(args.batch_size/1024)
    optimizer_rgtan = optim.Adam(model_rgtan.parameters(), lr=lr, weight_decay=args.wd)
    lr_scheduler = MultiStepLR(optimizer=optimizer_rgtan, milestones=[4000, 12000], gamma=0.3)
    earlystoper = early_stopper(patience=args.early_stopping, verbose=True)


    model_agg = BW_RGTAN_AGG(input_dim=h_feats*2, hidden_dim=args.hid_dim)
    # optimizer_agg = torch.optim.Adam(model_agg.parameters(), lr=0.001)
    # loss_agg = nn.CrossEntropyLoss()

    time_start = time.time()
    # 正式开始训练
    for epoch in range(args.epoch):

        model_bw.train()
        logits, h_bw = model_bw(features)
        loss_bw = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))
        # optimizer_bw.zero_grad()
        loss_bw.backward()
        # optimizer_bw.step()
        model_bw.eval()
        probs = logits.softmax(1)
        f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
        preds = numpy.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        trec = recall_score(labels[test_mask], preds[test_mask])
        tpre = precision_score(labels[test_mask], preds[test_mask])
        tmf1 = f1_score(labels[test_mask], preds[test_mask], average='macro')
        tauc = roc_auc_score(labels[test_mask], probs[test_mask][:, 1].detach().numpy())

        if best_f1 < f1:
            best_f1 = f1
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
        print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(epoch, loss, f1, best_f1))


        train_loss_list = []
        model_rgtan.train()
        model_agg.train()
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                num_feat, cat_feat, nei_feat, neigh_padding_dict, labels, seeds, input_nodes, device, blocks)

            blocks = [block.to(device) for block in blocks]
            train_batch_logits, out = model_rgtan(blocks, batch_inputs, lpa_labels, h_bw[seeds], batch_work_inputs, batch_neighstat_inputs)

            # mask = batch_labels == 2
            # train_batch_logits= train_batch_logits[~mask]
            # batch_labels = batch_labels[~mask]
            #
            # train_loss = loss_fn(train_batch_logits, batch_labels)
            # optimizer_rgtan.zero_grad()
            # train_loss.backward()
            # optimizer_rgtan.step()
            # lr_scheduler.step()
            # train_loss_list.append(train_loss.cpu().detach().numpy())

            mask = batch_labels == 2
            out= out[~mask]
            batch_labels = batch_labels[~mask]

            loss_rgtan = loss_fn(out, batch_labels)

            loss = loss_rgtan +loss_bw
            optimizer_rgtan.zero_grad()
            loss_rgtan.backward()
            optimizer_rgtan.step()
            lr_scheduler.step()
            train_loss_list.append(loss_rgtan.cpu().detach().numpy())

            # output = model_agg(h_bw[seeds], h_rgtan)
            # loss = loss_agg(output, labels[seeds])
            # loss.backward()
            # optimizer_agg.step()

            if step % 10 == 0:
                # tr_batch_pred = (torch.argmax(train_batch_logits, dim=1) == batch_labels).float().mean()
                # score = torch.softmax(train_batch_logits, dim=1)[:, 1].cpu().numpy()

                tr_batch_pred = torch.sum(torch.argmax(train_batch_logits.clone(
                ).detach(), dim=1) == batch_labels) / batch_labels.shape[0]
                score = torch.softmax(train_batch_logits.clone().detach(), dim=1)[
                        :, 1].cpu().numpy()
                try:
                    print('Epoch:{:03d}|Batch:{:04d}, Train Loss:{:.4f}, AP:{:.4f}, Acc:{:.4f}, AUC:{:.4f}'.format(
                        epoch, step, np.mean(train_loss_list),
                        average_precision_score(batch_labels.cpu().numpy(), score),
                        tr_batch_pred.item(),
                        roc_auc_score(batch_labels.cpu().numpy(), score)))
                except:
                    pass

        # 验证
        val_loss_list, val_acc_list, val_all_list = 0, 0, 0
        model_rgtan.eval()
        with torch.no_grad():
            for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                    num_feat, cat_feat, nei_feat, neigh_padding_dict, labels, seeds, input_nodes, device, blocks)

                blocks = [block.to(device) for block in blocks]
                val_batch_logits, out = model_rgtan(blocks, batch_inputs, lpa_labels, h_bw[seeds], batch_work_inputs, batch_neighstat_inputs)

                oof_predictions[seeds] = out
                mask = batch_labels == 2
                out = out[~mask]
                batch_labels = batch_labels[~mask]

                val_loss_list += loss_fn(out, batch_labels)
                val_batch_pred = (torch.argmax(out, dim=1) == batch_labels).float().sum()
                val_acc_list += val_batch_pred
                val_all_list += batch_labels.shape[0]

                if step % 10 == 0:
                    score = torch.softmax(val_batch_logits, dim=1)[:, 1].cpu().numpy()
                    try:
                        print('Epoch:{:03d}|Val Batch:{:04d}, Val Loss:{:.4f}, AP:{:.4f}, Acc:{:.4f}, AUC:{:.4f}'.format(
                            epoch, step, val_loss_list/val_all_list,
                            average_precision_score(batch_labels.cpu().numpy(), score),
                            val_batch_pred.item() / batch_labels.shape[0],
                            roc_auc_score(batch_labels.cpu().numpy(), score)))
                    except:
                        pass

        earlystoper.earlystop(val_loss_list/val_all_list, model_rgtan)
        if earlystoper.is_earlystop:
            print("Early stopping triggered!")
            break

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_trec*100,
                                                                     final_tpre*100, final_tmf1*100, final_tauc*100))


    print("Best val loss: {:.6f}".format(earlystoper.best_cv))

    # 测试
    test_ind = torch.from_numpy(np.array(idx_test)).long().to(device)
    test_sampler = MultiLayerFullNeighborSampler(args.n_layers)
    test_dataloader = DataLoader(graph,
                                test_ind,
                                test_sampler,
                                use_ddp=False,
                                device=device,
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=False,
                                num_workers=0)

    b_model = earlystoper.best_model.to(device)
    b_model.eval()
    with torch.no_grad():
        for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
            batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                num_feat, cat_feat, nei_feat, neigh_padding_dict, labels, seeds, input_nodes, device, blocks)

            blocks = [block.to(device) for block in blocks]
            test_batch_logits = b_model(blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
            test_predictions[seeds] = test_batch_logits
            test_batch_pred = (torch.argmax(test_batch_logits, dim=1) == batch_labels).float().mean()

            if step % 10 == 0:
                print('Test batch {:04d}, accuracy: {:.4f}'.format(step, test_batch_pred.item()))

    

    return final_tmf1, final_tauc


# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument("--dataset", type=str, default="amazon",
                        help="Dataset for this model (yelp/amazon/tfinance/tsocial)")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    parser.add_argument("--epoch", type=int, default=1, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")

    parser.add_argument("--device", type=str, default='cpu', help="cuda")
    parser.add_argument("--n_layers", type=int, default=2, help="nhop")
    parser.add_argument("--batch_size", type=int, default=64, help="batchsize")
    parser.add_argument("--dropout", type=list, default=[0.2, 0.1], help="dropout")
    parser.add_argument("--gated", type=bool, default=True, help="gated")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--wd", type=int, default=1, help="weight decay")
    parser.add_argument("--early_stopping", type=int, default=50, help="early_stopping")
    parser.add_argument("--nei_att_head", type=int, default=5, help="nei_att_head")

    

    

    args = parser.parse_args()
    print(args)

    if args.run == 1:
        train(args)

    else:
        final_mf1s, final_aucs = [], []
        for tt in range(args.run):
            mf1, auc = train(args)
            final_mf1s.append(mf1)
            final_aucs.append(auc)
        final_mf1s = np.array(final_mf1s)
        final_aucs = np.array(final_aucs)
        print('MF1-mean: {:.2f}, MF1-std: {:.2f}, AUC-mean: {:.2f}, AUC-std: {:.2f}'.format(100 * np.mean(final_mf1s),
                                                                                            100 * np.std(final_mf1s),
                                                               100 * np.mean(final_aucs), 100 * np.std(final_aucs)))