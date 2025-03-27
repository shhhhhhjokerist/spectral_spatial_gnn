# number of heter edge
import numpy as np

from data.graph_data.graph_construction import *
from pygsp import *
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec

s = np.random.randn(1000) + 1
s2 = np.zeros_like(s)
random.seed(123)
anomaly_id = []
for i in range(len(s)):
    if random.random() < 0.01:
        s2[i] = np.random.randn() * 1 + 1
        anomaly_id.append(i)
    else:
        s2[i] = s[i]

def convert_color(aid, x):
    co = np.zeros([len(x), 4])
    for i in aid:
        co[i][0]=65/255
        co[i][1]=105/255
        co[i][2]=225/255
    return co


def convert_shape(aid, x, scale=5, mins=30, maxs=120):
    # co = np.zeros([G.N, 4])
    x2 = x.copy()
    for i in aid:
        x2[i] = x[i]*scale
        x2[i] = max(x2[i],mins)
        x2[i] = min(x2[i],maxs)
    return x2


def plot_diag(G, sa, bar_scale=4):
    f = []
    x = np.linspace(0,2-2/bar_scale, bar_scale)
    for i in range(3):
        width = 2/(bar_scale*4)
        c = np.dot(G.U.transpose(), sa[i])
        S = np.zeros_like(c)
        for j in range(G.N):
            S[j] = sum((c * c)[:j + 1] / sum(c * c))
        M = np.zeros(bar_scale)
        for j in range(G.N):
            idx = min(int(G.e[j] / 0.5), bar_scale-1)
            M[idx] += c[j]**2
        M = M/sum(M)
        plt.plot(G.e, S, linewidth=3)
        f1 = plt.bar(x + width*(i+1), M, width=width)
        f.append(f1)
    return f


def plotg(G, gs, xs=0, ys=0, ft=17):
    ax1 = plt.subplot(gs[0,ys])
    # random.seed(123)
    sa = np.zeros([3, G.N])
    txt = ['μ=1, σ=1', 'μ=0.5, σ=1', 'μ=1, σ=2']
    sa[0] = np.random.randn(G.N)+1
    sa[1] = np.random.randn(G.N)+0.5
    sa[2] = np.random.randn(G.N)*2+0.5
    f = plot_diag(G, sa)

    if ys==0:
        fig.legend(handles=f, labels=txt,
                   loc='upper center', bbox_to_anchor=(0.5, 0.99),
                   ncol=3, fontsize=ft + 2)
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.yticks(fontsize=ft)
        plt.xlabel('Barabasi–Albert graph', fontsize=ft+2)

    else:
        plt.yticks([])
        plt.xlabel('Minnesota road graph', fontsize=ft+2)
    plt.xticks(np.arange(0, 2.001, step=0.5), fontsize=ft)


fig = plt.figure(figsize=(10, 5), dpi=500)

gs = gridspec.GridSpec(1, 2)
gs.update(left=0.09, right=0.98, top=0.85, bottom=0.16, wspace=0.05, hspace=0)

def cal_M(graph, i):
    edges = graph.edges()
    adj_matrix = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()), dtype=int)

    us, vs = edges

    for u, v in zip(us.tolist(), vs.tolist()):
        adj_matrix[u, v] = 1
        adj_matrix[v, u] = 1  # 如果是无向图

    features = graph.ndata['feature']

    psg_graph = graphs.Graph(adj_matrix)

    psg_graph.set_coordinates()
    psg_graph.compute_laplacian('normalized')
    psg_graph.compute_fourier_basis()


    bar_scale = 4
    f = []
    x = np.linspace(0, 2 - 2/bar_scale, bar_scale)

    # for


    width = 1/(bar_scale*4)
    c = np.dot(psg_graph.U.transpose(), features)
    S = np.zeros_like(c)
    for j in range(psg_graph.N):
        S[j] = sum((c * c)[:j + 1] / sum(c * c))
    S = np.mean(S, axis=1)

    M = np.zeros(bar_scale)
    for j in range(psg_graph.N):
        idx = min(int(psg_graph.e[j] / 0.5), bar_scale-1)
        # M[idx] += c[j]**2
        M[idx] += np.sum(c[j] ** 2)  # 解决报错

    M = M / sum(M)


    plt.plot(psg_graph.e, S, linewidth=3)
    f1 = plt.bar(x + width * (i + 1), M, width=width)
    return S, M


def random_graph_frequency_exp():
    # M = []
    # graph = rd_graph(500, 1000, 0.0, 0.2, 0.5, 2, 1)
    # M.append(cal_M(graph))\
    graph = rd_graph(500, 2000, 0.1, 0, 0.5, 2, 30)
    features = graph.ndata['feature']
    # M.append(cal_M(graph))
    cal_M(graph, 0)
    graph = rd_graph(500, 2000, 0.1, 0.2, 0.5, 2, 30)
    graph.ndata['feature'] = features
    # M.append(cal_M(graph))
    cal_M(graph, 1)
    graph = rd_graph(500, 2000, 0.1, 0.4, 0.5, 2, 30)
    graph.ndata['feature'] = features
    # M.append(cal_M(graph))
    cal_M(graph, 2)
    graph = rd_graph(500, 2000, 0.1, 0.6, 0.5, 2, 30)
    graph.ndata['feature'] = features
    # M.append(cal_M(graph))
    cal_M(graph, 3)
    graph = rd_graph(500, 2000, 0.1, 0.8, 0.5, 2, 30)
    graph.ndata['feature'] = features
    # M.append(cal_M(graph))
    cal_M(graph, 4)
    graph = rd_graph(500, 2000, 0.1, 1, 0.5, 2, 30)
    graph.ndata['feature'] = features
    # M.append(cal_M(graph))
    cal_M(graph, 5)

def amazon_graph_frequency_exp():
    # graph = amazon_graph(True, 1000, 30000, 0.2, 0, 0.2, True)
    # cal_M(graph, 0)
    # print('1')
    # graph = amazon_graph(True, 500, 30000, 0.2, 0.2, 0.2, True, graph)
    # cal_M(graph, 1)
    # print('2')
    # graph = amazon_graph(True, 500, 30000, 0.2, 0.4, 0.2, True, graph)
    # cal_M(graph, 2), graph
    # print('3'), graph
    # graph = amazon_graph(True, 500, 30000, 0.2, 0.6, 0.2, True, graph)
    # cal_M(graph, 3), graph
    # print('4'), graph
    # graph = amazon_graph(True, 500, 30000, 0.2, 0.8, 0.2, True, graph)
    # cal_M(graph, 4)
    # print('5')
    # graph = amazon_graph(True, 500, 30000, 0.2, 1, 0.2, True, graph)
    # cal_M(graph, 5)

    graph = amazon_graph(True, 1000, 30000, 0.2, 0.5, 0, True)
    cal_M(graph, 0)
    print('1')
    graph = amazon_graph(True, 1000, 30000, 0.2, 0.5, 0.2, True, graph)
    cal_M(graph, 1)
    print('2')
    graph = amazon_graph(True, 1000, 30000, 0.2, 0.5, 0.4, True, graph)
    cal_M(graph, 2), graph
    print('3'), graph
    graph = amazon_graph(True, 1000, 30000, 0.2, 0.5, 0.6, True, graph)
    cal_M(graph, 3), graph
    print('4'), graph
    graph = amazon_graph(True, 1000, 30000, 0.2, 0.5, 0.8, True, graph)
    cal_M(graph, 4)
    print('5')
    graph = amazon_graph(True, 1000, 30000, 0.2, 0.5, 1, True, graph)
    cal_M(graph, 5)


def amazon_graph_frequency_exp2():
    graph_org = amazon_graph_random_drop(True, 1000, 0, 0, 0, None, True)

    graph = amazon_graph_random_drop(True, 1000, 0, 0, 0, graph_org, True)
    cal_M(graph, 0)
    graph = amazon_graph_random_drop(True, 1000, 0, 0, 0.2, graph_org, True)
    cal_M(graph, 1)
    graph = amazon_graph_random_drop(True, 1000, 0, 0, 0.4, graph_org, True)
    cal_M(graph, 2)
    graph = amazon_graph_random_drop(True, 1000, 0, 0, 0.6, graph_org, True)
    cal_M(graph, 3)
    graph = amazon_graph_random_drop(True, 1000, 0, 0, 0.8, graph_org, True)
    cal_M(graph, 4)
    graph = amazon_graph_random_drop(True, 1000, 0, 0, 1, graph_org, True)
    cal_M(graph, 5)


def amazon_graph_frequency_exp2_v2():
    graph_org = amazon_graph_random_drop_v2(True, 1000, 0, 0, None, True)

    graph = amazon_graph_random_drop_v2(True, 1000, 0, 0, graph_org, True)
    cal_M(graph, 0)
    graph = amazon_graph_random_drop_v2(True, 1000, 0.2, 0, graph_org, True)
    cal_M(graph, 1)
    graph = amazon_graph_random_drop_v2(True, 1000, 0.4, 0, graph_org, True)
    cal_M(graph, 2)
    graph = amazon_graph_random_drop_v2(True, 1000, 0.6, 0, graph_org, True)
    cal_M(graph, 3)
    graph = amazon_graph_random_drop_v2(True, 1000, 0.8, 0, graph_org, True)
    cal_M(graph, 4)
    graph = amazon_graph_random_drop_v2(True, 1000, 1, 0, graph_org, True)
    cal_M(graph, 5)


def amazon_graph_frequency_exp3():
    graph_org = amazon_graph_random_drop(True, 1000, 0, 0, 0, None, True)

    graph = graph_org
    cal_M(graph, 0)
    graph = amazon_graph_random_drop_v2(True, 1000, 0, 0.5, graph_org, True)
    cal_M(graph, 1)
    graph = amazon_graph_random_drop_v2(True, 1000, 0.5, 0, graph_org, True)
    cal_M(graph, 2)
    graph = amazon_graph(True, 1000, 30000, 0.2, 0.8, 0.5, True, graph_org)
    cal_M(graph, 3)


def amazon_graph_frequency_exp4():
    # graph = amazon_graph(True, 1000, 30000, 0.2, 0.5, 0.5, True)
    # cal_M(graph, 0)
    # graph = amazon_graph(True, 2000, 60000, 0.2, 0.5, 0.5, True)
    # cal_M(graph, 1)
    # graph = amazon_graph(True, 4000, 120000, 0.2, 0.5, 0.5, True)
    # cal_M(graph, 2)
    # graph = amazon_graph(True, 8000, 240000, 0.2, 0.5, 0.5, True)
    # cal_M(graph, 3)
    # graph = amazon_graph(True, 16000, 30000, 0.2, 0.5, 0.5, True)
    # cal_M(graph, 4)
    # graph = amazon_graph(True, 30000, 30000, 0.2, 0.5, 0.5, True)
    # cal_M(graph, 5)


    graph = amazon_graph_random_drop(True, 1000, 0, 0, 0, None, True)
    cal_M(graph, 0)
    graph = amazon_graph_random_drop(True, 2000, 0, 0, 0, None, True)
    cal_M(graph, 1)
    graph = amazon_graph_random_drop(True, 4000, 0, 0, 0, None, True)
    cal_M(graph, 2)
    graph = amazon_graph_random_drop(True, 8000, 0, 0, 0, None, True)
    cal_M(graph, 3)
    # graph = amazon_graph_random_drop(True, 1000, 0, 0, 0, None, True)
    # cal_M(graph, 4)
    # graph = amazon_graph_random_drop(True, 1000, 0, 0, 0, None, True)
    # cal_M(graph, 5)





# plotg(psg_graph, gs, 0, 0)

# random_graph_frequency_exp()
amazon_graph_frequency_exp()
# amazon_graph_frequency_exp2()
# amazon_graph_frequency_exp2_v2()
# amazon_graph_frequency_exp3()
# amazon_graph_frequency_exp4()
plt.show()

