import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import torch.optim as optim
from dgl.data import FraudAmazonDataset
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.linalg import eig

from data.graph_data.graph_construction import amazon_graph
# 导入前面定义的 SpectralGCN 和相关函数

def compute_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    N = adj.shape[0]
    I = sp.eye(N)
    degrees = np.array(adj.sum(axis=1)).flatten()
    # degrees_inv_sqrt = np.power(degrees, -0.5)
    # degrees_inv_sqrt = np.where(degrees > 0, np.power(degrees, -0.5), 0)
    epsilon = 1e-10
    degrees_inv_sqrt = np.power(degrees + epsilon, -0.5)
    degrees_inv_sqrt[np.isinf(degrees_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(degrees_inv_sqrt)

    L = I - D_inv_sqrt @ adj @ D_inv_sqrt
    return L


def compute_spectral_decomposition(L):
    L = L.toarray()
    lambdas, U = eig(L)
    return torch.tensor(lambdas, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)


class SpectralGNNLayer(nn.Module):
    def __init__(self, in_features, out_features, A, use_normalized=True):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            A: 图的邻接矩阵，形状为 (N, N)
            use_normalized: 是否采用归一化拉普拉斯
        """
        super(SpectralGNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.N = A.shape[0]

        L = compute_normalized_laplacian(A)

        eigvals, U = compute_spectral_decomposition(L)
        self.register_buffer('U', U)

        # 每个谱分量都对应一个从 in_features 到 out_features 的变换矩阵
        # 参数形状为 (N, in_features, out_features)
        self.Theta = nn.Parameter(torch.randn(self.N, in_features, out_features))

        # 可选：增加偏置参数（对所有节点共享）
        # self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        """
        Args:
            x: 输入节点特征，形状为 (N, in_features)
        Returns:
            输出节点特征，形状为 (N, out_features)
        """
        # 1. 图傅里叶变换：将时域特征映射到谱域
        # x_hat = U^T x ，形状为 (N, in_features)
        x_hat = self.U.t() @ x

        # 2. 在谱域上应用滤波器
        # 对于每个频率 f，对应一个 in_features -> out_features 的线性变换：
        # y_hat[f] = x_hat[f] @ Theta[f]
        # 使用 batch 维度的矩阵乘法：
        x_hat_unsq = x_hat.unsqueeze(1)  # 形状变为 (N, 1, in_features)
        y_hat = torch.bmm(x_hat_unsq, self.Theta).squeeze(1)  # 形状 (N, out_features)

        # 3. 逆图傅里叶变换：将谱域特征转换回时域
        # y = U y_hat
        y = self.U @ y_hat

        # 4. 加上偏置
        # y = y + self.bias

        return y


class DeepSpectralGNN(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes, A, num_layers=3, dropout=0.5):
        """
        构建一个深层谱域 GNN，用于图级二分类任务。

        Args:
            in_features: 每个节点的输入特征维度
            hidden_features: 隐藏层特征维度
            num_classes: 中间层最后输出的特征数，之后经过全连接层转换为最终类别（此处一般设为较小值）
            A: 图的邻接矩阵
            num_layers: 谱域卷积层数（包括输入层和输出层）
            dropout: Dropout 率
        """
        super(DeepSpectralGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # 构造多层谱域卷积，每层均使用相同图结构 A
        self.layers = nn.ModuleList()
        # 第一层：将输入特征映射到隐藏特征
        self.layers.append(SpectralGNNLayer(in_features, hidden_features, A))
        # 中间层：隐藏到隐藏
        for _ in range(num_layers - 2):
            self.layers.append(SpectralGNNLayer(hidden_features, hidden_features, A))
        # 最后一层：映射到指定的 num_classes（这里可以看作一个中间分类表示）
        self.layers.append(SpectralGNNLayer(hidden_features, hidden_features, A))

        # 图级表示聚合：这里采用全局均值池化（也可以尝试 max pooling、attention pooling 等）
        # 接下来通过全连接层得到二分类输出
        self.fc = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        """
        Args:
            x: 输入节点特征，形状为 (N, in_features)
        Returns:
            输出二分类的 logit（单个标量）
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 最后一层不加非线性，保持原始谱域变换结果传入全连接层
            if i != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # x 的形状为 (N, num_classes)，进行全局均值池化（图级特征聚合）
        # x = torch.mean(x, dim=0)  # (num_classes,)
        logits = self.fc(x)  # (1,)
        return logits.squeeze()










homo = True
sample = False
sample_num_nodes = 1000
random_edge = False

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
    # data_spectral = amazon_graph(sample, sample_num_nodes, sample_num_nodes*50, 0.2, 0.8, 0.5, True, data)
    data_spectral = data

    # data_spatial = amazon_graph(sample, sample_num_nodes, sample_num_nodes*50, 0.2, 0.2, 0.5, True, data)
    data_spatial = data_spectral

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
adj = to_scipy_sparse_matrix(edge_spectral, num_nodes=num_nodes)


k = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 转换数据格式
X, labels = X.to(device), labels.to(device)

# 初始化模型
model = DeepSpectralGNN(in_features=X.shape[1], hidden_features=64, num_classes=num_classes, A=adj, num_layers=3, dropout=0.5).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)


train_masks = data.ndata['train_mask'].bool()
val_masks = data.ndata['val_mask'].bool()
test_masks = data.ndata['test_mask'].bool()

weight = (1-labels[train_masks]).sum().item() / labels[train_masks].sum().item()
print('cross entropy weight: ', weight)


# 训练循环
num_epochs = 500
patience = 500  # 设定早停的容忍度
best_val_auc = 0  # 记录最佳的验证集AUC
early_stopping_counter = 0  # 记录连续未提升的轮数
best_model_state = None  # 用于存储最佳模型状态

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    out = model(X)

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
        out_val = model(X)
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
    out = model(X)
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

# adj = to_scipy_sparse_matrix(edge_test, num_nodes=feature_test.shape[0])
# laplacian = compute_normalized_laplacian(adj)
# model.spectral_gnn.lambdas, model.spectral_gnn.U = compute_spectral_decomposition(laplacian)
# model.spectral_gnn.lambdas = model.spectral_gnn.lambdas.to(device)
# model.spectral_gnn.U = model.spectral_gnn.U.to(device)

model.eval()
with torch.no_grad():
    out = model(feature_test)
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
