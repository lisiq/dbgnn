import os
import scipy as sp
import scipy.sparse 
import numpy as np
from collections import defaultdict

from network import  Network_light
from higher_order_network  import HigherOrderNetwork_light
from variable_order_chawla import VariableOrderNetwork_Xu_light

from chawla.ExtractVariableOrderRulesFreq import *
from chawla.NetworkRewiring import *




folder_path =  "data" 




def reindex_csr_matrix(A_old,ix_to_node_old,node_to_index_new):
    """  
        Reindexes one matrix entries according to another matrix indexes
    """
    assert set(ix_to_node_old.values()).issubset(set(node_to_index_new.keys())), " nodes in ix_to_node_old are not a subset of those in node_to_index_new"
    assert isinstance(A_old, scipy.sparse.csr_matrix), "A_old is not a scipy sparse csr matrix"
    ix_v_old = A_old.nonzero()[0]
    ix_w_old = A_old.nonzero()[1]
    values_old = A_old.data
    ixs_v = []
    ixs_w = []
    values = []
    for edge_index in range(len(ix_v_old)):
        # translate into indexes of topological matrix A_c
        ixs_v.append(
            node_to_index_new[
                ix_to_node_old[
                    ix_v_old[
                        edge_index
                        ]]])
        ixs_w.append(
            node_to_index_new[
                ix_to_node_old[
                    ix_w_old[
                        edge_index
                        ]]])
        values.append(
            values_old[
                edge_index
                ])
    A_new = scipy.sparse.csr_matrix((values,(ixs_v,ixs_w)), shape = (len(node_to_index_new),len(node_to_index_new)))
    return A_new



def load_HONEM_neighborhood_matrix(
    filename,
    frequency = True,
    fixed_node_to_index = None,
    order_limit = None,
    separator = ","):
    """ 
        Loads an ngram as a sparse adjacency matrix. 
        order: int
            indicates the length of the sequences used as nodes. k implies sequences of length k+1
    """
    node_to_index = {}
    last_index = 0
    file_path = os.path.join(folder_path, filename)
    # values = []
    values_dict = defaultdict(float)
    with open(file_path, "r") as f:
        for line in f:
            line = line.replace("\n","")
            line = line.split(separator)
            if frequency:
                value = float(line.pop())
            else:
                value = 1
            for i in range(len(line)-1):
                for j in range(i+1,len(line)):
                    if order_limit is not None and j-i > order_limit:
                        break
                    v_id = tuple(line[i:i+1])
                    w_id = tuple(line[j:j+1])
                    if v_id not in node_to_index:
                        node_to_index[v_id] = last_index 
                        last_index += 1
                    if w_id not in node_to_index:
                        node_to_index[w_id] = last_index
                        last_index += 1
                    k = j - i
                    values_dict[(v_id, w_id)] += value * np.exp(-k)

    row_indexes = []
    col_indexes = [] 
    values = []
    for (row_id,col_id),value in values_dict.items():
        row_indexes.append(node_to_index[row_id])
        col_indexes.append(node_to_index[col_id])
        values.append(value)

    index_to_node = {v:k for k,v in node_to_index.items()} 
    csr_mat =  scipy.sparse.csr_matrix((values,(row_indexes,col_indexes)), shape = (len(node_to_index),len(node_to_index)))
    if fixed_node_to_index is not None: 
        assert len(index_to_node) == len(fixed_node_to_index), "externally imposed node_to_index has a different number of elemets from the ones read from file"
        csr_mat = reindex_csr_matrix(
            csr_mat,
            ix_to_node_old = index_to_node,
            node_to_index_new = fixed_node_to_index
            )
        index_to_node = {v:k for k,v in fixed_node_to_index.items()}
    return csr_mat,index_to_node #node_to_index,



###############################################################


def path_loader(dt, frequency = True,  separator = ","):
    """
    Loads paths into list of lists object 
    """
    # NON OPTIMAL BECAUSE OF HOW IT HANDLES REPEATED INTERACTIONS. USE THE NE
    observed_paths = []
    with open(os.path.join(folder_path,dt),"r") as f:
        for row in f:
            row_list = row.replace("\n","").split(separator)
            #row_list = list(map(lambda x : "n_"+x if x[0].isdigit() else x, row_list))
            if frequency: 
                list_nodes = row_list[:-1]   #all but the weight
                weight = row_list[-1]
                assert int(float(weight)) == float(weight), "Weights have to be counts, not frequencies"
                observed_paths.extend([list_nodes]*int(float(weight)))
            else:
                # row_list = list(map(lambda x : "n_"+x if x[0].isdigit() else x, row_list))
                observed_paths.append(row_list)
    return observed_paths

def load_Xu_Network(file_path):
    RawTrajectories = list(enumerate(path_loader(file_path)))
    MaxOrder = 99
    # MinSupport = 1
    Rules = ExtractRules(RawTrajectories, MaxOrder)  # history to future dictionary
    return BuildNetwork(Rules)


def Xu2016_to_sparse_representation(Xu_Network,unweighted = False, directed = True):
    """
    Converts XuNetwork (from Xu2016 - Representing higher-order dependencies in networks), 
    which is a dict from origins to destinations (with counts) into an adjacency matrix + index to node
    (variable order)
    """
    node_to_index = {}
    last_index = 0
    values_dict = defaultdict(float)

    for origin,destinations_dict in Xu_Network.items():
        if origin not in node_to_index:
            node_to_index[origin] = last_index 
            last_index += 1
        for destination, weight in destinations_dict.items():
            if destination not in node_to_index:
                node_to_index[destination] = last_index 
                last_index += 1
            values_dict[(origin, destination)] += weight

    row_indexes = []
    col_indexes = [] 
    values = []
    for (row_id,col_id),value in values_dict.items():
        row_indexes.append(node_to_index[row_id])
        col_indexes.append(node_to_index[col_id])
        values.append(value)

    index_to_node = {v:k for k,v in node_to_index.items()} 
    csr_mat =  scipy.sparse.csr_matrix((values,(row_indexes,col_indexes)), shape = (len(node_to_index),len(node_to_index)))
    if not directed:
        csr_mat = (csr_mat + csr_mat.T)/2
    if unweighted:
        csr_mat = (csr_mat > 0)*1
    return VariableOrderNetwork_Xu_light(csr_mat, {v:k for k,v in index_to_node.items()}, directed = directed)

def load_XuChawla_hon_sparse(filename, directed = True):
    # file_path = os.path.join(folder_path, filename)
    return Xu2016_to_sparse_representation(load_Xu_Network(filename), directed = directed)



################################################################


def load_model_matrix(filename,order,frequency = True, separator = ","):
    """ 
        Loads an ngram as a sparse adjacency matrix. 
        order: int
            indicates the length of the sequences used as nodes. k implies sequences of length k+1
    """
    node_to_index = {}
    last_index = 0

    # values = []
    values_dict = defaultdict(float)
    with open(filename, "r") as f:
        for line in f:
            line = line.replace("\n","")
            line = line.split(separator)
            if frequency:
                value = float(line.pop())
            else:
                value = 1
            for i in range(len(line)-order): 
                v_id = tuple(line[i:i+order])
                w_id = tuple(line[i+1:i+1+order])
                if v_id not in node_to_index:
                    node_to_index[v_id] = last_index 
                    last_index += 1
                if w_id not in node_to_index:
                    node_to_index[w_id] = last_index
                    last_index += 1
                values_dict[(v_id, w_id)] += value

    row_indexes = []
    col_indexes = [] 
    values = []
    for (row_id,col_id),value in values_dict.items():
        row_indexes.append(node_to_index[row_id])
        col_indexes.append(node_to_index[col_id])
        values.append(value)

    index_to_node = {v:k for k,v in node_to_index.items()} 
    csr_mat =  scipy.sparse.csr_matrix((values,(row_indexes,col_indexes)), shape = (len(node_to_index),len(node_to_index)))
    return csr_mat, index_to_node #node_to_index,

def load_data(filename, frequency = True):
    file_path = os.path.join(folder_path, filename)
    A, ix_to_node  = load_model_matrix(   # create option tuple True / False (TODOs)
        file_path,order = 1, frequency=frequency)
    node_to_ix = {v:k for k,v in ix_to_node.items()}

    return A, node_to_ix


def load_hon(dt,order, unweighted = False, directed = True): 
    network =  load_network(dt, unweighted=unweighted, directed = directed) 
    # hon
    sparse_ho_adj_mat, index_to_node_ho = load_model_matrix(
        os.path.join(
            folder_path,
            dt), order = order)
    if not directed:
        sparse_ho_adj_mat = (sparse_ho_adj_mat+sparse_ho_adj_mat.T)/2
    if unweighted:
        sparse_ho_adj_mat = (sparse_ho_adj_mat>0)*1

    node_to_index_ho = {v:k for k,v in index_to_node_ho.items()}  
    ho_net = HigherOrderNetwork_light(
        sparse_ho_adj_mat,
        node_to_index_ho,
        network,
        directed
        )
    return ho_net


def load_network(dt, unweighted = False, directed = True, frequency = True):
    A, node_to_ix = load_data(dt, frequency = frequency)
    if not directed:
        A = (A+A.T)/2    
    if unweighted:
        A = (A > 0)*1
    return Network_light(A, node_to_ix) 

