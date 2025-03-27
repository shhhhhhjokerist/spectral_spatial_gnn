import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh  # 计算特征值分解
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.nn import GCNConv
from scipy.linalg import eig


class SpatialGCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        """
        经典图域 GCN
        :param num_features: 输入特征维度
        :param hidden_dim: 隐藏层维度
        :param num_classes: 输出类别数
        """
        super(SpatialGCN, self).__init__()

        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        """
        前向传播
        :param x: [N, num_features] 节点特征
        :param edge_index: PyG 格式的边索引
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class SpectralGCN(nn.Module):
    def __init__(self, num_features, num_classes, k=10):
        """
        直接使用谱分解的GNN
        :param num_features: 输入特征维度
        :param num_classes: 输出类别数
        :param k: 选取前 k 个特征值（用于降维）
        """
        super(SpectralGCN, self).__init__()

        self.k = k
        self.fc = nn.Linear(num_features, num_classes)  # 线性变换层

        # 初始化可学习滤波器参数 g(λ)
        self.filter_params = nn.Parameter(torch.randn(k))

    def forward(self, x, edge_index):
        """
        前向传播
        :param x: [N, num_features] 节点特征
        :param edge_index: PyG 格式的邻接矩阵索引
        """
        # 计算稀疏邻接矩阵
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=x.shape[0])
        laplacian = compute_normalized_laplacian(adj)

        # 计算特征分解
        lambdas, U = compute_spectral_decomposition(laplacian, self.k)
        lambdas = lambdas.to(x.device)
        U = U.to(x.device)

        # 计算 g(Λ) * X
        lambda_filter = self.filter_params * lambdas  # g(λ) 作用于 λ
        lambda_diag = torch.diag(lambda_filter)
        spectral_x = (U @ (lambda_diag @ (U.T @ x)))  # U g(Λ) U^T X

        # 转换为 PyTorch Tensor 并通过全连接层
        # spectral_x = torch.tensor(spectral_x, dtype=torch.float32)
        out = self.fc(spectral_x)

        return F.log_softmax(out, dim=1)


# 计算归一化拉普拉斯矩阵
def compute_normalized_laplacian(adj):
    """
    计算归一化拉普拉斯矩阵 L = I - D^(-1/2) A D^(-1/2)
    :param adj: scipy.sparse 格式的邻接矩阵
    :return: 归一化拉普拉斯矩阵 (scipy.sparse.csr_matrix)
    """
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


# 计算谱分解（选取前 k 个特征值和特征向量）
def compute_spectral_decomposition(L, k):
    """
    计算拉普拉斯矩阵的前 k 个特征值和特征向量
    :param L: scipy.sparse.csr_matrix 归一化拉普拉斯矩阵
    :param k: 选取的前 k 个特征值
    :return: (前 k 个特征值, 对应的特征向量矩阵)
    """
    L = L.toarray()
    # lambdas, U = eigsh(L, k=k, which='SM')  # 计算最小的 k 个特征值
    # lambdas, U = eigsh(L, k=L.shape[0]-1, which='SM')  # 计算最小的 k 个特征值
    lambdas, U = eig(L)
    return torch.tensor(lambdas, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)


class HybridGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, k=10):
        """
        结合谱域 GNN（SpectralGCN）和图域 GNN（SpatialGCN）
        """
        super(HybridGNN, self).__init__()

        self.spectral_gnn = SpectralGCN(num_features, num_classes, k)
        self.spatial_gnn = SpatialGCN(num_features, hidden_dim, num_classes)

        self.fc = nn.Linear(num_classes, num_classes)  # 预测层

    def forward(self, x, edge_index):
        spectral_out = self.spectral_gnn(x, edge_index)  # 谱域 GNN 输出
        spatial_out = self.spatial_gnn(x, edge_index)  # 图域 GNN 输出

        fusion_out = spectral_out + spatial_out  # 每一层特征相加
        return self.fc(fusion_out)  # 经过全连接层后输出