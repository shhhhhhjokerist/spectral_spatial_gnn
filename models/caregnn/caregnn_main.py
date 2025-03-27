import time
import os
import random
import argparse

import dgl
from dgl.data import FraudAmazonDataset
from sklearn.model_selection import train_test_split

from caregnn_utils import *
from caregnn_model import *
from caregnn_layers import *
from data.graph_data.graph_construction import amazon_graph, amazon_graph_random_drop_v3

# from graphsage import *

def to_adj(relation, labels):
	(us, vs) = relation
	us = us.numpy()
	vs = vs.numpy()

	reviews = defaultdict(set)
	for i in range(len(labels)):
		reviews[i] = set()

	for (u, v) in zip(us, vs):
		# if u not in reviews.keys():
		# 	reviews[u] = set()
		reviews[u].add(v)
	return reviews


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
	Training CARE-GNN
	Paper: Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
	Source: https://github.com/YingtongDou/CARE-GNN
"""

parser = argparse.ArgumentParser()

# dataset and model dependent args
parser.add_argument('--data', type=str, default='amazon', help='The dataset name. [yelp, amazon]')
parser.add_argument('--model', type=str, default='CARE', help='The model name. [CARE, SAGE]')
parser.add_argument('--inter', type=str, default='GNN', help='The inter-relation aggregator type. [Att, Weight, Mean, GNN]')
parser.add_argument('--batch-size', type=int, default=256, help='Batch size 1024 for yelp, 256 for amazon.')

# hyper-parameters
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--lambda_1', type=float, default=2, help='Simi loss weight.')
parser.add_argument('--lambda_2', type=float, default=1e-3, help='Weight decay (L2 loss weight).')
parser.add_argument('--emb-size', type=int, default=64, help='Node embedding size at the last layer.')
parser.add_argument('--num-epochs', type=int, default=31, help='Number of epochs.')
parser.add_argument('--test-epochs', type=int, default=3, help='Epoch interval to run test set.')
parser.add_argument('--under-sample', type=int, default=1, help='Under-sampling scale.')
parser.add_argument('--step-size', type=float, default=2e-2, help='RL action step size')
parser.add_argument('--run', type=float, default=5, help='run times')

# other args
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(f'run on {args.data}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

final_f1s, final_accs, final_recalls, final_aucs, final_aps = [], [], [], [], []
for tt in range(args.run):
	# load graph, feature, and label
	# data =  amazon_graph(True, sample_num, sample_num*200, 0, 1, 0.5, True)
	# graph = amazon_graph_random_drop_v3(False, 1000, 0, 0, None, False)
	graph = FraudAmazonDataset()[0]
	graph = dgl.add_self_loop(graph, etype='net_upu')
	graph = dgl.add_self_loop(graph, etype='net_usu')
	graph = dgl.add_self_loop(graph, etype='net_uvu')
	# [homo, relation1, relation2, relation3], feat_data, labels = load_data(args.data)
	labels = graph.ndata['label']
	labels = np.array(labels)
	feat_data = graph.ndata['feature']
	relation1 = graph.edges(etype='net_upu')
	relation2 = graph.edges(etype='net_usu')
	relation3 = graph.edges(etype='net_uvu')

	relation1 = to_adj(relation1, labels)
	relation2 = to_adj(relation2, labels)
	relation3 = to_adj(relation3, labels)


	graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
	homo = graph.edges()

	# train_test split
	np.random.seed(args.seed)
	random.seed(args.seed)
	if args.data == 'yelp':
		index = list(range(len(labels)))
		idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=0.60,
																random_state=2, shuffle=True)
	elif args.data == 'amazon':  # amazon
		# 0-3304 are unlabeled nodes
		index = list(range(3305, len(labels)))
		idx_train, idx_test, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
																test_size=0.60, random_state=2, shuffle=True)

	# split pos neg sets for under-sampling
	train_pos, train_neg = pos_neg_split(idx_train, y_train)

	# initialize model input
	features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
	feat_data = normalize(feat_data)
	features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
	if args.cuda:
		features.cuda()

	# set input graph
	if args.model == 'SAGE':
		adj_lists = homo
	else:
		adj_lists = [relation1, relation2, relation3]

	print(f'Model: {args.model}, Inter-AGG: {args.inter}, emb_size: {args.emb_size}.')

	# build one-layer models
	if args.model == 'CARE':
		intra1 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
		intra2 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
		intra3 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
		inter1 = InterAgg(features, feat_data.shape[1], args.emb_size, adj_lists, [intra1, intra2, intra3], inter=args.inter,
						  step_size=args.step_size, cuda=args.cuda)
	# elif args.model == 'SAGE':
	# 	agg1 = MeanAggregator(features, cuda=args.cuda)
	# 	enc1 = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg1, gcn=True, cuda=args.cuda)

	if args.model == 'CARE':
		gnn_model = OneLayerCARE(2, inter1, args.lambda_1)
	# elif args.model == 'SAGE':
	# 	# the vanilla GraphSAGE model as baseline
	# 	enc1.num_samples = 5
	# 	gnn_model = GraphSage(2, enc1)

	if args.cuda:
		gnn_model.cuda()

	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr, weight_decay=args.lambda_2)
	times = []
	performance_log = []

	# train the model
	for epoch in range(args.num_epochs):
		# randomly under-sampling negative nodes for each epoch
		sampled_idx_train = undersample(train_pos, train_neg, scale=1)
		rd.shuffle(sampled_idx_train)

		# send number of batches to model to let the RLModule know the training progress
		num_batches = int(len(sampled_idx_train) / args.batch_size) + 1
		if args.model == 'CARE':
			inter1.batch_num = num_batches

		loss = 0.0
		epoch_time = 0

		# mini-batch training
		for batch in range(num_batches):
			start_time = time.time()
			i_start = batch * args.batch_size
			i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
			batch_nodes = sampled_idx_train[i_start:i_end]
			batch_label = labels[np.array(batch_nodes)]
			optimizer.zero_grad()
			if args.cuda:
				loss = gnn_model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))
			else:
				loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))
			loss.backward()
			optimizer.step()
			end_time = time.time()
			epoch_time += end_time - start_time
			loss += loss.item()

		print(f'Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {epoch_time}s')

		# testing the model for every $test_epoch$ epoch
		if epoch % args.test_epochs == 0:
			if args.model == 'SAGE':
				test_sage(idx_test, y_test, gnn_model, args.batch_size)
			else:
				# gnn_auc, label_auc, gnn_recall, label_recall = test_care(idx_test, y_test, gnn_model, args.batch_size)
				gnn_auc, label_auc, gnn_recall, label_recall, gnn_f1, gnn_acc, gnn_ap = test_care(idx_test, y_test, gnn_model, args.batch_size)
				performance_log.append([gnn_auc, label_auc, gnn_recall, label_recall])

	final_f1s.append(gnn_f1)
	final_accs.append(gnn_acc)
	final_recalls.append(gnn_recall)
	final_aucs.append(gnn_auc)
	final_aps.append(gnn_ap)

final_f1s = np.array(final_f1s)
final_accs = np.array(final_accs)
final_recalls = np.array(final_recalls)
final_aucs = np.array(final_aucs)
final_aps = np.array(final_aps)
print('MF1-mean: {:.2f}, MF1-std: {:.2f}, ACC-mean: {:.2f}, ACC-std: {:.2f}, '
	  'RECALL-mean: {:.2f}, RECALL-std: {:.2f}, AUC-mean: {:.2f}, AUC-std: {:.2f}, '
	  'AP-mean: {:.2f}, AP-std: {:.2f}'.format(100 * np.mean(final_f1s),
												100 * np.std(final_f1s),
												100 * np.mean(final_accs),
												100 * np.std(final_accs),
												100 * np.mean(final_recalls),
												100 * np.std(final_recalls),
												100 * np.mean(final_aucs),
												100 * np.std(final_aucs),
												100 * np.mean(final_aps),
												100 * np.std(final_aps),))


