from dataclasses import replace
import datetime
import os
from typing import Callable, Optional

import collections

import numpy as np
import random
import pandas as pd
import scipy as sp
from math import floor, ceil
import pathpy as pp

from sklearn.preprocessing import normalize

import warnings
warnings.filterwarnings("ignore")

import json

from torch_geometric.transforms import RandomNodeSplit

import torch
from abc import ABC

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_gz,
)


class load_data(ABC):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root: str, directed= False):
            super().__init__()
            self.root = root
            self.directed = directed


    def find_filenames(self, suffix='.csv'):
        filenames = os.listdir(self.root + '/raw')
        return [filename for filename in filenames if filename.endswith(suffix)][0]

    def raw_file_names(self) -> str:
        file_name = self.root + '/raw/' + self.find_filenames('.ngram')
        return file_name

    def node_to_index(self):
        with open(self.root + '/raw/' + self.find_filenames('.gt'), 'r') as fp:
            data_ = json.load(fp)
        node_to_index = dict(zip(list(data_), range(len(data_))))
        node_to_index = {str(k):v for k,v in node_to_index.items()}
        return node_to_index

    def index_to_node(self):
        return {k:v for v,k in self.node_to_index().items()}

    def gt_to_index(self):
        with open(self.root + '/raw/' + self.find_filenames('.gt'), 'r') as fp:
            data_ = json.load(fp)
        gts = sorted(list(set(data_.values())))
        return {k:v for k,v in zip(gts, range(len(gts)))}

    def get_ground_truth(self) -> str:
        with open(self.root + '/raw/' + self.find_filenames('.gt'), 'r') as fp:
            gt_dict = json.load(fp)

        node_to_index = self.node_to_index()
        gts = self.gt_to_index()
        gt_dict_indices = {}
        for k,v in gt_dict.items():
            try:
                gt_dict_indices[node_to_index[k]] = gts[v]
            except:
                print('Node {} is not in the network, we just have the label.'.format(k))
        ordered_dict = collections.OrderedDict(sorted(gt_dict_indices.items()))
        ground_truth = torch.tensor([i[1] for i in ordered_dict.items()]).long()
        return ground_truth

    def process(self):

        # map nodes to indices
        node_to_index = self.node_to_index()

        # load the dataframe names
        file_name = self.raw_file_names()
        paths = pp.Paths.read_file(file_name)
        net = pp.Network.from_paths(paths)
        
        # node labels
        classes = self.get_ground_truth()

        if self.directed == False:
            edge_index = torch.zeros(2,2*net.ecount(),dtype=torch.long)
            i = 0
            for (v,w) in net.edges:
                edge_index[0, i] = node_to_index[v]
                edge_index[1, i] = node_to_index[w]
                i += 1
            i = 0
            for (v,w) in net.edges:
                edge_index[1, i+net.ecount()] = node_to_index[v]
                edge_index[0, i+net.ecount()] = node_to_index[w]
                i += 1
        else:
            edge_index = torch.zeros(2,net.ecount(),dtype=torch.long)
            i = 0
            for (v,w) in net.edges:
                edge_index[0, i] = node_to_index[v]
                edge_index[1, i] = node_to_index[w]
                i += 1

        # edge weights
        if self.directed == False:
            weight1 = torch.zeros(2 * net.ecount())
            i = 0
            for (v,w) in net.edges:
                weight1[i] = net.edges[(v,w)]['weight']
                i += 1
            i = 0
            for (v,w) in net.edges:
                weight1[i+net.ecount()] = net.edges[(v,w)]['weight']
                i += 1

        else:
            weight1 = torch.zeros(net.ecount())
            i = 0
            for (v,w) in net.edges:
                weight1[i] = net.edges[(v,w)]['weight']
                i += 1


        ## second order
        multi_order_network = pp.MultiOrderModel(paths, max_order=2)
        hon = multi_order_network.layers[2]
        hon_edge_index = torch.zeros(2, hon.ecount(), dtype=torch.long)
        bipartite_index = torch.zeros(2 ,hon.ncount(), dtype=torch.long)

        # hon edge index
        hon_node_to_index = hon.node_to_name_map()
        i = 0
        for (v, w) in hon.edges:
            hon_edge_index[0, i] = hon_node_to_index[v]
            hon_edge_index[1, i] = hon_node_to_index[w]
            i += 1

        # bipartite edge index 
        i = 0
        for v in hon.nodes:
            bipartite_index[0, i] = hon_node_to_index[v]
            fo = hon.higher_order_node_to_path(v)
            bipartite_index[1, i] = node_to_index[fo[-1]]
            i += 1

        # hon edge weight
        hon_edge_weight = torch.zeros(hon.ecount())
        i = 0
        for (v, w) in hon.edges:
            hon_edge_weight[i] = hon.edges[(v,w)]['weight'][0] + hon.edges[(v,w)]['weight'][1]
            i += 1


        num_nodes = net.ncount()
        num_nodes_3 = hon.ncount()

        x = torch.eye(num_nodes, num_nodes)
        x1 = torch.eye(num_nodes_3, num_nodes_3)
        # Data object
        data = Data(
            x = x,
            edge_index = edge_index,
            edge_weight = weight1,
            x1 = x1,
            edge_index_1 = hon_edge_index,
            edge_weight_1 = hon_edge_weight,
            num_nodes_1 = num_nodes_3,
            y = classes,
            bipartite_index = bipartite_index
        )

        return data
