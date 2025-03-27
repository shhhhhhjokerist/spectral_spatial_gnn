import dgl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.data import FraudAmazonDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score

from data.graph_data.graph_construction import amazon_graph
# 导入前面定义的 SpectralGCN 和相关函数
from spectral_spacial_layer2 import *


homo = True
sample = True
sample_num_nodes = 1000
random_edge = True

# ==================== 1. 加载 Cora 数据集 ====================
dataset = FraudAmazonDataset()
data = dataset[0]  # 获取单个图数据

if homo:
    data = dgl.to_homogeneous(data, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])

# ==================== 2. 构造训练数据 ====================
if sample:
    num_nodes = data.num_nodes()
    sampled_nodes = np.random.choice(num_nodes, sample_num_nodes, replace=False)
    data = dgl.node_subgraph(data, sampled_nodes)

if random_edge:
    data_spectral = amazon_graph(sample, sample_num_nodes, sample_num_nodes*100, 0.2, 0.8, 0.5, True, data)
    data_spatial = amazon_graph(sample, sample_num_nodes, sample_num_nodes*100, 0.2, 0.2, 0.5, True, data)

    edge_spectral = data_spectral.edges()
    edge_spectral = torch.stack(edge_spectral, dim=0)
    edge_spatial = data_spatial.edges()
    edge_spatial = torch.stack(edge_spatial, dim=0)
else:
    edges = data.edges()  # 边信息
    edge_index = torch.stack(edges, dim=0)
    edge_spectral = edge_index
    edge_spatial = edge_index

X = data.ndata['feature']  # 节点特征 [N, num_features]

labels = data.ndata['label']  # 标签
num_nodes, num_features = X.shape
num_classes = 2

# # 转换邻接矩阵
# adj = to_scipy_sparse_matrix(edge_spectral, num_nodes=num_nodes)
# # edges = np.array(edge_index)
# laplacian = compute_normalized_laplacian(adj)

# 计算谱分解
k = 50  # 选择前 k 个特征值
# lambdas, U = compute_spectral_decomposition(laplacian, k)
#
# # 转换为 PyTorch 格式
# lambdas = lambdas.to(torch.float32)
# U = U.to(torch.float32)

# ==================== 3. 训练模型 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 转换数据格式
X, labels = X.to(device), labels.to(device)

# 初始化模型
model = HybridGNN(num_features, 64, num_classes, k, X, edge_spectral).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)


train_masks = data.ndata['train_mask'].bool()
val_masks = data.ndata['val_mask'].bool()
test_masks = data.ndata['test_mask'].bool()

weight = (1-labels[train_masks]).sum().item() / labels[train_masks].sum().item()
print('cross entropy weight: ', weight)


# 训练循环
num_epochs = 500
patience = 100  # 设定早停的容忍度
best_val_auc = 0  # 记录最佳的验证集AUC
early_stopping_counter = 0  # 记录连续未提升的轮数
best_model_state = None  # 用于存储最佳模型状态

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    out = model(X, edge_spectral, edge_spatial)

    # 计算训练损失
    loss = F.cross_entropy(out[train_masks], labels[train_masks], weight=torch.tensor([1., weight]).to(device))
    loss.backward()
    optimizer.step()

    # 计算训练集的准确率
    pred_train = out.argmax(dim=1)
    acc_train = (pred_train[train_masks] == labels[train_masks]).float().mean().item()

    # 计算训练集的 AUC 和 F1
    y_true_train = labels[train_masks].cpu().numpy()
    y_pred_train = pred_train[train_masks].cpu().numpy()
    y_proba_train = out[train_masks].softmax(dim=1)[:, 1].cpu().detach().numpy()

    f1_train = f1_score(y_true_train, y_pred_train, average="macro")
    auc_train = roc_auc_score(y_true_train, y_proba_train)

    # 计算验证集的指标
    model.eval()
    with torch.no_grad():
        out_val = model(X, edge_spectral, edge_spatial)
        pred_val = out.argmax(dim=1)
        acc_val = (pred_val[val_masks] == labels[val_masks]).float().mean().item()

        y_true_val = labels[val_masks].cpu().numpy()
        y_pred_val = pred_val[val_masks].cpu().numpy()
        y_proba_val = out[val_masks].softmax(dim=1)[:, 1].cpu().detach().numpy()

        f1_val = f1_score(y_true_val, y_pred_val, average="macro")
        auc_val = roc_auc_score(y_true_val, y_proba_val)

    # 早停逻辑
    if auc_val > best_val_auc:
        best_val_auc = auc_val
        best_model_state = model.state_dict()  # 保存最佳模型权重
        early_stopping_counter = 0  # 重置计数
    else:
        early_stopping_counter += 1

    # 打印训练和验证集的指标
    if epoch % 10 == 0:
        print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}, '
              f'Train Acc: {acc_train:.4f}, AUC: {auc_train:.4f}, F1: {f1_train:.4f}, '
              f'Val Acc: {acc_val:.4f}, Val AUC: {auc_val:.4f}, Val F1: {f1_val:.4f}')

    # 早停：如果验证 AUC 在 `patience` 轮内没有提升，则停止训练
    if early_stopping_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
    print(f"Early stopping counter: {early_stopping_counter}")

# 载入最佳模型
if best_model_state:
    model.load_state_dict(best_model_state)

# ============ 计算最终测试集表现 ============
model.eval()
with torch.no_grad():
    out = model(X, edge_spectral, edge_spatial)
    pred = out.argmax(dim=1)
    acc_test = (pred[test_masks] == labels[test_masks]).float().mean().item()

    # 计算测试集的 AUC 和 F1-score
    y_true_test = labels[test_masks].cpu().numpy()
    y_pred_test = pred[test_masks].cpu().numpy()
    y_proba_test = out[test_masks].softmax(dim=1)[:, 1].cpu().detach().numpy()

    f1_test = f1_score(y_true_test, y_pred_test, average="macro")
    auc_test = roc_auc_score(y_true_test, y_proba_test)

print(f'Best Val AUC: {best_val_auc:.4f}')
print(f'Test Accuracy: {acc_test:.4f}, Test AUC: {auc_test:.4f}, Test F1: {f1_test:.4f}')


data_test = dataset[0]
data_test = dgl.to_homogeneous(data_test, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
num_nodes = data_test.num_nodes()
sampled_nodes = np.random.choice(num_nodes, sample_num_nodes, replace=False)
data_test = dgl.node_subgraph(data_test, sampled_nodes)
edge_test = data_test.edges()
edge_test = torch.stack(edge_test, dim=0)
feature_test = data_test.ndata['feature'].to(device)
labels = data_test.ndata['label'].to(device)

adj = to_scipy_sparse_matrix(edge_test, num_nodes=feature_test.shape[0])
laplacian = compute_normalized_laplacian(adj)
model.spectral_gnn.lambdas, model.spectral_gnn.U = compute_spectral_decomposition(laplacian, model.spectral_gnn.k)
model.spectral_gnn.lambdas = model.spectral_gnn.lambdas.to(device)
model.spectral_gnn.U = model.spectral_gnn.U.to(device)

model.eval()
with torch.no_grad():
    out = model(feature_test, edge_test, edge_test)
    pred = out.argmax(dim=1)
    acc_test = (pred == labels).float().mean().item()

    # 计算测试集的 AUC 和 F1-score
    y_true_test = labels.cpu().numpy()
    y_pred_test = pred.cpu().numpy()
    y_proba_test = out.softmax(dim=1)[:, 1].cpu().detach().numpy()

    f1_test = f1_score(y_true_test, y_pred_test, average="macro")
    auc_test = roc_auc_score(y_true_test, y_proba_test)

print(f'Best Val AUC: {best_val_auc:.4f}')
print(f'Test Accuracy: {acc_test:.4f}, Test AUC: {auc_test:.4f}, Test F1: {f1_test:.4f}')
