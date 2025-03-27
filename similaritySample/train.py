import pickle

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

from similaritySample.SSGNN import SampledGCN

#载入数据
# dataset = Planetoid(root='./dataset/Cora', name='Cora')
# data = dataset[0]
data = pickle.load(open('../dataset/yelp.dat', 'rb'))



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SampledGCN(data.x.shape[1], 2).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#模型训练
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)    #模型的输入有节点特征还有边特征,使用的是全部数据
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])   #损失仅仅计算的是训练集的损失
    loss.backward()
    optimizer.step()
    print(loss)
#测试：
model.eval()
test_predict = model(data.x, data.edge_index)[data.test_mask]
max_index = torch.argmax(test_predict, dim=1)
test_true = data.y[data.test_mask]
correct = 0
count = 0
for i in range(len(max_index)):
    if max_index[i] == test_true[i]:
        correct += 1
    if 0 == max_index[i]:
        count += 1
print('测试集准确率为：{}%'.format(correct*100/len(test_true)))
print('count：{}%'.format(count/len(test_true)))