import dgl


graphs, _ =dgl.load_graphs('./amazon/org_graph.dgl')
graph = graphs[0]


feature = graph.ndata['feature']
label = graph.ndata['label']
train_mask = graph.ndata['train_mask']
val_mask = graph.ndata['val_mask']
test_mask = graph.ndata['test_mask']
mark = graph.ndata['mark']

# adj = graph.adj(scipy_fmt='csr')

print(graph.edges())






