import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable


from operator import itemgetter
import math

from models.bwgnn.bwgnn_models import calculate_theta2, PolyConv, PolyConvBatch


class InterAgg(nn.Module):
    def __init__(self, features, feature_dim,
                 embed_dim, adj_lists, intraggs,
                 inter='GNN', step_size=0.02, cuda=True,
                 num_classes=2, graph=None, d=2, batch=False):
        super(InterAgg, self).__init__()

        self.features = features
        self.dropout = 0.6
        self.adj_lists = adj_lists
        self.intra_agg1 = intraggs[0]
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.inter = inter
        self.step_size = step_size
        self.cuda = cuda
        self.intra_agg1.cuda = cuda

        self.RL = True

        # number of batches for current epoch, assigned during training
        self.batch_num = 0

        # initial filtering thresholds
        self.thresholds = [0.5, 0.5]

        # the activation function used by attention mechanism
        self.leakyrelu = nn.LeakyReLU(0.2)

        # parameter used to transform node embeddings before inter-relation aggregation
        self.weight = nn.Parameter(torch.FloatTensor(self.feat_dim, self.embed_dim))
        init.xavier_uniform_(self.weight)

        # weight parameter for each relation used by CARE-Weight
        self.alpha = nn.Parameter(torch.FloatTensor(self.embed_dim, 3))
        init.xavier_uniform_(self.alpha)

        # parameters used by attention layer
        self.a = nn.Parameter(torch.FloatTensor(2 * self.embed_dim, 1))
        init.xavier_uniform_(self.a)

        # label predictor for similarity measure
        self.label_clf = nn.Linear(self.feat_dim, 2)

        # initialize the parameter logs
        self.weights_log = []
        self.thresholds_log = [self.thresholds]
        self.relation_score_log = []

        self.g = graph.to('cuda:0' if cuda else 'cpu')

        self.thetas = calculate_theta2(d=d)
        self.spe_conv = []
        for i in range(len(self.thetas)):
            if not batch:
                self.spe_conv.append(PolyConv(embed_dim, embed_dim, self.thetas[i], lin=False))
            else:
                self.spe_conv.append(PolyConvBatch(embed_dim, embed_dim, self.thetas[i], lin=False))
        self.linear = nn.Linear(feature_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.linear3 = nn.Linear(embed_dim, feature_dim)
        self.act = nn.ReLU()

        self.agg_weight = [0, 1]


    def forward(self, nodes, labels, train_flag=True):
        to_neighs = []
        for adj_list in self.adj_lists:
            to_neighs.append([set(adj_list[int(node)]) for node in nodes])

        # find unique nodes and their neighbors used in current batch
        # unique_nodes = set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]),
        #                          set.union(*to_neighs[2], set(nodes)))
        # unique_nodes = set.union(set.union(*to_neighs[0], set(nodes)))
        unique_nodes = set(range(0, len(self.adj_lists[0])))

        # calculate label-aware scores
        if self.cuda:
            batch_features = self.features(torch.cuda.LongTensor(list(unique_nodes)))
        else:
            batch_features = self.features(torch.LongTensor(list(unique_nodes)))

        # print(batch_features.device)  # 查看 batch_features 所在的设备
        # for param in self.label_clf.parameters():
        #     print(param.device)  # 查看 label_clf 的参数在哪个设备上
        # self.label_clf.to(batch_features.device)

        batch_scores = self.label_clf(batch_features)
        id_mapping = {node_id: index for node_id, index in zip(unique_nodes, range(len(unique_nodes)))}

        # the label-aware scores for current batch of nodes
        center_scores = batch_scores[itemgetter(*nodes)(id_mapping), :]

        # get neighbor node id list for each batch node and relation
        r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
        # r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
        # r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]

        # assign label-aware scores to neighbor nodes for each batch node and relation
        r1_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r1_list]
        # r2_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r2_list]
        # r3_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r3_list]

        # count the number of neighbors kept for aggregation for each batch node and relation
        r1_sample_num_list = [math.ceil(len(neighs) * self.thresholds[0]) for neighs in r1_list]
        # r2_sample_num_list = [math.ceil(len(neighs) * self.thresholds[1]) for neighs in r2_list]
        # r3_sample_num_list = [math.ceil(len(neighs) * self.thresholds[2]) for neighs in r3_list]

        # intra-aggregation steps for each relation
        # Eq. (8) in the paper
        r1_feats, r1_scores = self.intra_agg1.forward(nodes, r1_list, center_scores, r1_scores, r1_sample_num_list)
        # r2_feats, r2_scores = self.intra_agg2.forward(nodes, r2_list, center_scores, r2_scores, r2_sample_num_list)
        # r3_feats, r3_scores = self.intra_agg3.forward(nodes, r3_list, center_scores, r3_scores, r3_sample_num_list)





        h = self.linear(batch_features)

        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(batch_features), 0]).cuda()
        for conv in self.spe_conv:
            h = conv(self.g, h)
            # h_final = torch.cat([h_final, h0], -1)
        h = self.linear3(h)

        spe_feats = h[nodes]



        # concat the intra-aggregated embeddings from each relation
        spe_spa_feats = torch.cat((r1_feats, spe_feats), dim=0)


        # get features or embeddings for batch nodes
        if self.cuda and isinstance(nodes, list):
            index = torch.LongTensor(nodes).cuda()
        else:
            index = torch.LongTensor(nodes)
        self_feats = self.features(index)

        # number of nodes in a batch
        n = len(nodes)




        # spe-spa aggregation steps
        # Eq. (9) in the paper
        if self.inter == 'Att':
            # 1) CARE-Att Inter-relation Aggregator
            combined, attention = att_inter_agg(len(self.adj_lists), self.leakyrelu, self_feats, spe_spa_feats, self.embed_dim,
                                                self.weight, self.a, n, self.dropout, self.training, self.cuda)
        elif self.inter == 'Weight':
            # 2) CARE-Weight Inter-relation Aggregator
            combined = weight_inter_agg(len(self.adj_lists), self_feats, spe_spa_feats, self.embed_dim, self.weight, self.alpha, n, self.cuda)
            gem_weights = F.softmax(torch.sum(self.alpha, dim=0), dim=0).tolist()
            if train_flag:
                print(f'Weights: {gem_weights}')
        elif self.inter == 'Mean':
            # 3) CARE-Mean Inter-relation Aggregator
            combined = mean_inter_agg(len(self.adj_lists), self_feats, spe_spa_feats, self.embed_dim, self.weight, n, self.cuda)
        elif self.inter == 'GNN':
            # 4) CARE-GNN Inter-relation Aggregator
            combined = threshold_inter_agg(len(self.adj_lists), self_feats, spe_spa_feats, self.embed_dim, self.weight, self.agg_weight, n, self.cuda)

        # # the reinforcement learning module
        # if self.RL and train_flag:
        #     relation_scores, rewards, thresholds, stop_flag = RLModule([r1_scores],
        #                                                                self.relation_score_log, labels, self.thresholds,
        #                                                                self.batch_num, self.step_size)
        #     self.thresholds = thresholds
        #     self.RL = stop_flag
        #     self.relation_score_log.append(relation_scores)
        #     self.thresholds_log.append(self.thresholds)

        return combined, center_scores


def RLModule(scores, scores_log, labels, thresholds, batch_num, step_size):
    """
    The reinforcement learning module.
    It updates the neighbor filtering threshold for each relation based
    on the average neighbor distances between two consecutive epochs.
    :param scores: the neighbor nodes label-aware scores for each relation
    :param scores_log: a list stores the relation average distances for each batch
    :param labels: the batch node labels used to select positive nodes
    :param thresholds: the current neighbor filtering thresholds for each relation
    :param batch_num: numbers batches in an epoch
    :param step_size: the RL action step size
    :return relation_scores: the relation average distances for current batch
    :return rewards: the reward for given thresholds in current epoch
    :return new_thresholds: the new filtering thresholds updated according to the rewards
    :return stop_flag: the RL terminal condition flag
    """

    relation_scores = []
    stop_flag = True

    # only compute the average neighbor distances for positive nodes
    pos_index = (labels == 1).nonzero().tolist()
    pos_index = [i[0] for i in pos_index]

    # compute average neighbor distances for each relation
    for score in scores:
        pos_scores = itemgetter(*pos_index)(score)
        neigh_count = sum([1 if isinstance(i, float) else len(i) for i in pos_scores])
        pos_sum = [i if isinstance(i, float) else sum(i) for i in pos_scores]
        relation_scores.append(sum(pos_sum) / neigh_count)

    if len(scores_log) % batch_num != 0 or len(scores_log) < 2 * batch_num:
        # do not call RL module within the epoch or within the first two epochs
        rewards = [0, 0, 0]
        new_thresholds = thresholds
    else:
        # update thresholds according to average scores in last epoch
        # Eq.(5) in the paper
        previous_epoch_scores = [sum(s) / batch_num for s in zip(*scores_log[-2 * batch_num:-batch_num])]
        current_epoch_scores = [sum(s) / batch_num for s in zip(*scores_log[-batch_num:])]

        # compute reward for each relation and update the thresholds according to reward
        # Eq. (6) in the paper
        rewards = [1 if previous_epoch_scores[i] - s >= 0 else -1 for i, s in enumerate(current_epoch_scores)]
        new_thresholds = [thresholds[i] + step_size if r == 1 else thresholds[i] - step_size for i, r in enumerate(rewards)]

        # avoid overflow
        new_thresholds = [0.999 if i > 1 else i for i in new_thresholds]
        new_thresholds = [0.001 if i < 0 else i for i in new_thresholds]

        print(f'epoch scores: {current_epoch_scores}')
        print(f'rewards: {rewards}')
        print(f'thresholds: {new_thresholds}')

    # TODO: add terminal condition

    return relation_scores, rewards, new_thresholds, stop_flag


def att_inter_agg(num_relations, att_layer, self_feats, neigh_feats, embed_dim, weight, a, n, dropout, training, cuda):
    return


def weight_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, alpha, n, cuda):
    return


def mean_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, n, cuda):
    return



def threshold_inter_agg(num_relations, self_feats, spe_spa_feats, embed_dim, weight, agg_weight, n, cuda):
    """
    CARE-GNN inter-relation aggregator
    Eq. (9) in the paper
    :param num_relations: number of relations in the graph
    :param self_feats: batch nodes features or embeddings
    :param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
    :param embed_dim: the dimension of output embedding
    :param weight: parameter used to transform node embeddings before inter-relation aggregation
    :param threshold: the neighbor filtering thresholds used as aggregating weights
    :param n: number of nodes in a batch
    :param cuda: whether use GPU
    :return: inter-relation aggregated node embeddings
    """

    # transform batch node embedding and neighbor embedding in each relation with weight parameter
    center_h = torch.mm(self_feats, weight)
    neigh_h = torch.mm(spe_spa_feats, weight)

    # initialize the final neighbor embedding
    if cuda:
        aggregated = torch.zeros(size=(n, embed_dim)).cuda()
    else:
        aggregated = torch.zeros(size=(n, embed_dim))

    # add weighted neighbor embeddings in each relation together
    for r in range(2):
        aggregated += neigh_h[r * n:(r + 1) * n, :] * agg_weight[r]

    # sum aggregated neighbor embedding and batch node embedding
    # feed them to activation function
    combined = F.relu(center_h + aggregated)

    return combined


class OneLayerCARE_BW(nn.Module):
    """
    The CARE-GNN model in one layer
    """

    def __init__(self, num_classes, inter1, lambda_1, weight):
        """
        Initialize the CARE-GNN model
        :param num_classes: number of classes (2 in our paper)
        :param inter1: the inter-relation aggregator that output the final embedding
        """
        super(OneLayerCARE_BW, self).__init__()
        self.inter1 = inter1
        self.xent = nn.CrossEntropyLoss(weight=torch.tensor([1., weight]))

        # the parameter to transform the final embedding
        self.weight = nn.Parameter(torch.FloatTensor(inter1.embed_dim, num_classes))
        init.xavier_uniform_(self.weight)
        self.lambda_1 = lambda_1

    def forward(self, nodes, labels, train_flag=True):
        embeds1, label_scores = self.inter1(nodes, labels, train_flag)
        scores = torch.mm(embeds1, self.weight)
        return scores, label_scores

    def to_prob(self, nodes, labels, train_flag=True):
        gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
        gnn_prob = nn.functional.softmax(gnn_scores, dim=1)
        label_prob = nn.functional.softmax(label_scores, dim=1)
        return gnn_prob, label_prob

    def loss(self, nodes, labels, train_flag=True):
        gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
        # Simi loss, Eq. (4) in the paper
        label_loss = self.xent(label_scores, labels.squeeze())
        # GNN loss, Eq. (10) in the paper
        gnn_loss = self.xent(gnn_scores, labels.squeeze())
        # the loss function of CARE-GNN, Eq. (11) in the paper
        final_loss = gnn_loss + self.lambda_1 * label_loss
        return final_loss
