import pathpy as pp
from collections import defaultdict
import torch
from tqdm.notebook import tqdm
import torch_geometric
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from torch_geometric.data import Data


def optimal_maximum_order(temp_net, max_order=2):
    if isinstance(temp_net, pp.classes.temporal_network.TemporalNetwork):
    #     start_time = time.time()
        paths = pp.path_extraction.paths_from_temporal_network_dag(temp_net)
    #     print(time.time()-start_time)
    else:
        paths = temp_net
#     start_time = time.time()
    multi_order_network = pp.MultiOrderModel(paths, max_order=max_order)
#     print(time.time()-start_time)
#     start_time = time.time()
    # k = multi_order_network.estimate_order(paths)
    k=3
#     print(time.time()-start_time)
#     G = multi_order_network.layers[k]
    return multi_order_network, k


def get_edge_index(fon):

    fon_edge_index = torch.zeros([fon.ecount(), 2], dtype=torch.long)

    fon_node_to_index = fon.node_to_name_map()
    # hon_node_to_index = hon.node_to_name_map()

    # fon edge index
    i = 0
    for (v, w) in fon.edges:
        fon_edge_index[i, 0] = fon_node_to_index[v]
        fon_edge_index[i, 1] = fon_node_to_index[w]
        i += 1
    # undirected fon
    # i = 0
    # for (v, w) in fon.edges:
    #     fon_edge_index[i+fon.ecount(), 1] = fon_node_to_index[v]
    #     fon_edge_index[i+fon.ecount(), 0] = fon_node_to_index[w]
    #     i += 1

    # fon edge weight
    fon_edge_weight = torch.zeros(fon.ecount())
    i = 0
    for (v, w) in fon.edges:
        fon_edge_weight[i] = fon.edges[(v,w)]['weight']
        i += 1
    # i = 0
    # for (v, w) in fon.edges:
    #     fon_edge_weight[i+fon.ecount()] = fon.edges[(v,w)]['weight']
    #     i += 1


    num_nodes = fon.ncount()

    x = torch.eye(num_nodes, num_nodes)

    clusters = {str(v): 0 if len(str(v))<2 else (1 if str(v).startswith('1') else 2) for v in range(fon.ncount())}
    reverse_index = {v: k for k, v in fon_node_to_index.items()}
    y = torch.tensor([clusters[reverse_index[i]] for i in range(fon.ncount())]).type(torch.LongTensor)

    data = Data(
        x = x,
        edge_index = fon_edge_index.T,
        edge_weight = fon_edge_weight,
        y = y
    )

    return data