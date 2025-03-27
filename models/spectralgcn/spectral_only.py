import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.linalg import eig


class SpectralGCN(nn.Module):
    def __init__(self, num_features, num_classes, k, x, edge_spectral, num_bins=15):
        super(SpectralGCN, self).__init__()

        self.k = k
        self.num_bins = num_bins

        self.filter_params = nn.Parameter(torch.randn(num_bins))

        adj = to_scipy_sparse_matrix(edge_spectral, num_nodes=x.shape[0])
        laplacian = compute_normalized_laplacian(adj)

        lambdas, U = compute_spectral_decomposition(laplacian, self.k)
        self.lambdas = lambdas.to(x.device)
        self.U = U.to(x.device)

        self.lambdas_min = self.lambdas.min().item()
        self.lambdas_max = self.lambdas.max().item()
        self.lambdas_normalized = 2 * (self.lambdas - self.lambdas_min) / (self.lambdas_max - self.lambdas_min)


    def forward(self, x, edge_index):
        lambdas = self.lambdas_normalized

        bin_width = 2 / (self.num_bins - 1)  # 计算 bin 的宽度
        indices = (lambdas / bin_width).long().clamp(0, self.num_bins - 1)  # 计算 bin 索引

        lambda_filter = self.filter_params[indices]

        lambda_diag = torch.diag(lambda_filter)
        spectral_x = self.U @ (lambda_diag @ (self.U.T @ x))

        return spectral_x


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


def compute_spectral_decomposition(L, k):
    L = L.toarray()
    lambdas, U = eig(L)
    return torch.tensor(lambdas, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)


