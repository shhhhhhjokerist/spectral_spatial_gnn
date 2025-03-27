import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops


# 计算欧氏距离
def compute_euclidean_distance(x, edge_index):
    node_a, node_b = edge_index
    # 取出两个节点的特征
    a_features = x[node_a]  # 获取节点a的特征
    b_features = x[node_b]  # 获取节点b的特征
    # 计算欧氏距离
    dist = torch.norm(a_features - b_features, dim=1)  # 计算每对节点的距离
    return dist


# 过滤低相似度的边
def filter_edges_by_similarity(x, edge_index, threshold=0.5):
    # 计算相邻节点的欧氏距离
    dist = compute_euclidean_distance(x, edge_index)
    # 过滤掉相似度低于阈值的边
    mask = dist < threshold
    filtered_edge_index = edge_index[:, mask]  # 选择相似度大于阈值的边
    return filtered_edge_index


class SampledGCN(nn.Module):
    def __init__(self, in_channels, out_channels, threshold=0.5):
        super(SampledGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.threshold = threshold
        self.fc = nn.Linear(out_channels, 2)

    def forward(self, x, edge_index):
        # 过滤低相似度的边
        filtered_edge_index = filter_edges_by_similarity(x, edge_index, self.threshold)

        # 加上自环
        filtered_edge_index, _ = add_self_loops(filtered_edge_index, num_nodes=x.size(0))

        # 使用新的邻接矩阵进行卷积计算
        x = self.conv1(x, filtered_edge_index)

        # 通过分类层，输出每个节点的分类结果
        # out = self.fc(x)


        return F.log_softmax(x, dim=1)


# 测试模型
if __name__ == "__main__":
    # 假设有 4 个节点，每个节点的特征是 3 维
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                     dtype=torch.float)  # 节点特征 (4, 3)

    # 假设的邻接边（edge_index），每列是一个边的起始节点和终止节点
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)  # 4个节点，4条边

    # 定义采样GCN模型
    model = SampledGCN(in_channels=3, out_channels=2, threshold=2.0)  # 输入特征维度为 3，输出特征维度为 2

    # 前向传播
    out = model(x, edge_index)

    print("Output node features:", out)
