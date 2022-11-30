import json
import os
import numpy as np
import uuid as _uuid
import glob as _glob

from tqdm import tqdm

from utils import gt_list_from_gt_dict, evaluate_learning_classification
from utils_dataloading import load_gt_clusters
from utils_modelloading import load_network, load_hon

def save_dict(data, filename):
	"""
		saves dictionary data
	"""
	with open(filename, "w") as file:
		file.write(json.dumps(data))

def create_tasks( parameters, repeat_experiment, folder = r"res/",):
	"""
		creates files that are treated as tasks to compute

		folder has to be in form r"res/"
		
	"""
	parameters["done"] = False
	for _ in range(repeat_experiment):
		# write file
		filename_input =  folder + str(_uuid.uuid4()) + ".results"
		save_dict(parameters, filename_input)


def find_unfinished(folder):
	"""
		finds all tasks that are not finished.
		The tasks are assumed to be created with
		create_tasks.

		folder has form r"res/"
	"""
	unfinished = []
	for filename in _glob.glob(folder+"*.results"):
		with open(filename, "r") as file:
			results = json.load(file)
			if not results["done"]:
				unfinished.append(filename)
	return unfinished


##########################################


# STILL MISSING: DGE, t-node-embed

def experiment_node_classification(filename):
    with open(filename, "r") as file:
        parameters = json.load(file)
        
    if parameters["method"] == "GCN":
        output_dict = node_classification_GCN(parameters)
    elif parameters["method"] == "DBGNN":
        output_dict = node_classification_DBGNN(parameters)
    elif parameters["method"] == "HONEM":
        output_dict = node_classification_HONEM(parameters)
    elif parameters["method"] == "DeepWalk":
        output_dict = node_classification_DeepWalk(parameters)
    elif parameters["method"] == "Node2Vec":
        output_dict = node_classification_Node2Vec(parameters)
    elif parameters["method"] == "EVO":
        output_dict = node_classification_EVO(parameters)
    else:
        assert False, f'{parameters["method"]} Not implemented'
    
    output_dict["done"] = True
    save_dict(output_dict, filename)



from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.data import Data
import torch

from gcn import GCN
def node_classification_GCN(parameters):
    # CREATE DATA OBJECT
    gt = load_gt_clusters()[parameters["dataset"]]
    network = load_network(
        parameters["dataset"],
        unweighted = not parameters["weighted"],
        directed = parameters["directed"]
        ) 
    #


    ground_truth = gt_list_from_gt_dict(gt, network).long()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if parameters["weighted"]:
        edge_weight = torch.tensor(network.adjacency_matrix.data).float()
    else:
        edge_weight = torch.ones(len(network.adjacency_matrix.data)).float()
    data = Data(
        edge_index= torch.tensor(network.adjacency_matrix.nonzero()).long(),
        edge_weight = edge_weight,
        x = torch.eye(network.n_nodes),
        y = torch.tensor(ground_truth).long()
        ).to(device)
    # SPLIT
    fraction_test = parameters["fraction_test"]
    num_test = int(network.n_nodes*fraction_test) 
    RandomNodeSplit(num_val=0, num_test=num_test)(data)

    gcn = GCN(
        num_features=network.n_nodes,
        num_classes=len(set(data.y)),
        hidden_dims=parameters["hidden_dims"],
        p_dropout=parameters["p_dropout"]
        ).to(device)

    # TRAIN MODEL
    n_epochs = parameters["n_epochs"]
    lr = parameters["learning_rate"]
    weight_decay = parameters["weight_decay"]
    optimizer = torch.optim.Adam(gcn.parameters(),  lr=lr, weight_decay=weight_decay)
    loss_function = torch.nn.CrossEntropyLoss()
    for _ in tqdm(range(n_epochs)):
        output = gcn(data) 
        loss = loss_function(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # EVALUATE MODEL
    eval_test, eval_train = evaluate_learning_classification(gcn, data)
    output_dict = {**parameters,**eval_test, **eval_train}
    return output_dict



from dbgnn import HO_GCN
def node_classification_DBGNN(parameters):
    # CREATE DATA OBJECT
    gt = load_gt_clusters()[parameters["dataset"]]
    hon = load_hon(
        parameters["dataset"],
        parameters["opt_order"],
        unweighted=not parameters["weighted"],
        directed=parameters["directed"]
        )

    ground_truth = gt_list_from_gt_dict(gt, hon.network).long()
    edge_index_hon_to_fon = [(hon.node_to_index[ho_node],ix_fo_node) for ho_node,ix_fo_node in  hon.ho_node_to_fo_index.items()]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if parameters["weighted"]:
        edge_weight =  torch.tensor(hon.adjacency_matrix.data).float()
        edge_weight_fo = torch.tensor(hon.network.adjacency_matrix.data).float()
    else:
        edge_weight = torch.ones(len(hon.adjacency_matrix.data)).float()
        edge_weight_fo = torch.ones(len(hon.network.adjacency_matrix.data)).float()    
    data = Data(
        edge_index= torch.tensor(hon.adjacency_matrix.nonzero()).long(),
        edge_index_fo= torch.tensor(hon.network.adjacency_matrix.nonzero()).long(),
        edge_index_hon_to_fon= torch.tensor(edge_index_hon_to_fon).long().T,
        edge_weight =edge_weight,
        edge_weight_fo = edge_weight_fo, 
        x_ho = torch.eye(hon.n_nodes),
        x_fo = torch.eye(hon.network.n_nodes),
        y = torch.tensor(ground_truth).long(),
        num_nodes = hon.network.n_nodes,
        num_ho_nodes = hon.n_nodes,
        ).to(device)
    # SPLIT
    fraction_test =  parameters["fraction_test"]
    num_test = int(hon.network.n_nodes*fraction_test) 
    RandomNodeSplit(num_val=0, num_test=num_test)(data)

    # TRAIN MODEL
    n_epochs = parameters["n_epochs"]
    lr = parameters["learning_rate"]
    weight_decay = parameters["weight_decay"]
    hogcn = HO_GCN(
        num_features=[hon.n_nodes, hon.network.n_nodes ],
        num_classes=len(set(gt.values())),
        hidden_dims=parameters["hidden_dims"],
        p_dropout=parameters["p_dropout"]
        ).to(device)
    optimizer = torch.optim.Adam(hogcn.parameters(),  lr=lr, weight_decay=weight_decay)
    loss_function = torch.nn.CrossEntropyLoss()
    for _ in tqdm(range(n_epochs)):
        output = hogcn(data, device) #self.n2v_embedding)
        loss = loss_function(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # EVALUATE MODEL
    eval_test, eval_train = evaluate_learning_classification(hogcn, data, device)
    output_dict = {**parameters,**eval_test,**eval_train}
    return output_dict


from embeddings.SVD_embedding import SVD_embedding
from utils_modelloading import load_HONEM_neighborhood_matrix
def node_classification_HONEM(parameters):

    HONEM_neighborhood, __HONEM_index_to_nodes = load_HONEM_neighborhood_matrix(parameters["dataset"])
    network = load_network(
        dt = parameters["dataset"],
        unweighted = parameters["weighted"],
        directed = parameters["directed"]
        )
    emb_obj = SVD_embedding(
        min(parameters["d"],network.adjacency_matrix.shape[0]),
            network,
            HONEM_neighborhood
            ) # HONEM_index_to_nodes
    emb_obj.compute_truncatedSVD_embedding()

    # emb_obj.compute_embedding()
    output_dict = classification(
        emb_obj,
        load_gt_clusters(), 
        parameters
        )
    return output_dict



from embeddings.node2vec import Node_to_Vec
def node_classification_Node2Vec(parameters):
    # CREATING DATA (EMB OBJECT)
    network = load_network(
        parameters["dataset"],
        unweighted = not parameters["weighted"],
        directed = parameters["directed"]
        ) 

    emb_obj = Node_to_Vec(
        d = parameters["d"],
        network = network,
        p = parameters["p"], 
        q = parameters["q"] 
        )
    emb_obj.compute_embedding()

    output_dict = classification(
        emb_obj,
        load_gt_clusters(),
        parameters
        )
    return output_dict


from embeddings.deep_walk import DeepWalk
def node_classification_DeepWalk(parameters):
    # CREATING DATA (EMB OBJECT)
    network = load_network(
        parameters["dataset"],
        unweighted = not parameters["weighted"],
        directed = parameters["directed"]
        ) 
    emb_obj = DeepWalk(
        d = parameters["d"],
        network = network
        )
    emb_obj.compute_embedding()

    output_dict = classification(
        emb_obj,
        load_gt_clusters(),
        parameters,
        )
    return output_dict


from utils_modelloading import load_XuChawla_hon_sparse
from embeddings.evo import EVO
def node_classification_EVO(parameters):
    # CREATING DATA (EMB OBJECT)
    hon_XUChawla = load_XuChawla_hon_sparse(  # TODOs: what about weighted
        parameters["dataset"],
        directed = parameters["directed"]
        )
    n2v_4_evo = Node_to_Vec(
        d = parameters["d"],
        network = hon_XUChawla,
        p = parameters["p"], 
        q = parameters["q"] 
    )
    n2v_4_evo.compute_embedding()
    emb_obj = EVO(n2v_4_evo)
    if parameters["f_aggregation"] == "avg":
        emb_obj.compute_embedding(np.mean) 
    elif parameters["f_aggregation"] == "max":
        emb_obj.compute_embedding(np.max)
    elif parameters["f_aggregation"] == "sum":
        emb_obj.compute_embedding(np.sum)
    else:
        assert False, f"unknown aggregation {parameters['f_aggregation']} for EVO; aggregations in input are 'sum' and 'max' "

    output_dict = classification(
        emb_obj,
        load_gt_clusters(),
        parameters,
        )
    return output_dict


def node_classification_tnodeembed():
    assert False, "not implemented"

def node_classification_DGE():
    assert False, "not implemented"







from utils_dataloading import clean_objects_gt, load_target_y_tube
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from skmultilearn.model_selection import iterative_train_test_split

# from sklearn.metrics import roc_auc_score, accuracy_score
from utils import calculate_metrics

def classification(
    emb_obj,
    gt_modules_dict, # dict and not just gt_modules cause it gonna be taken and modiefied often. Dirty handling of reference in python
    parameters,
    ):
    def create_clf(
        emb_obj,
        embedding_matrix,
        dt,
        gt_modules_dict,
        train_size
        ):
            """
                Creates sklearn classification object
            """
            if dt == "tube_paths_train.ngram":
                # http://scikit.ml/stratification.html -- link to stratification for multiclass problem
                # https://datascience.stackexchange.com/questions/45174/how-to-use-sklearn-train-test-split-to-stratify-data-for-multi-label-classificat
                target_y = load_target_y_tube(
                    emb_obj.node_to_index,
                    drop_Waterloo=True)
                xtrain, ytrain, xtest, ytest = iterative_train_test_split(
                    embedding_matrix,
                    target_y,
                    test_size = 1 - train_size,
                    # shuffle=True,
                    # stratify = target_y
                    )
                clf = MultiOutputClassifier(
                    LogisticRegression(
                        multi_class='ovr',
                        max_iter = 100000)) 
            elif dt == "sociopatterns_primary_80.edgelist":
                gt_modules = gt_modules_dict[dt]
                target_y = [gt_modules[emb_obj.index_to_node[ix]] for ix in range(len(emb_obj.nodes))]
                gt_modules, embedding_matrix, target_y,_ = clean_objects_gt(
                    gt_modules,
                    embedding_matrix,
                    target_y,
                    emb_obj.index_to_node,
                    ["M","F"]
                    )

                xtrain, xtest, ytrain, ytest = train_test_split(
                    embedding_matrix,
                    target_y,
                    train_size=train_size,
                    shuffle=True,
                    stratify = target_y 
                    )
                # clf = MultiOutputClassifier(LogisticRegression(multi_class='ovr')) 
                clf = LogisticRegression(solver='sag',max_iter = 100000)  
            else:
                gt_modules = gt_modules_dict[dt]
                target_y = [gt_modules[emb_obj.index_to_node[ix]] for ix in range(len(emb_obj.nodes))]
                xtrain, xtest, ytrain, ytest = train_test_split(
                    embedding_matrix,
                    target_y,
                    train_size=train_size,
                    shuffle=True,
                    stratify = target_y
                    )
                clf = LogisticRegression(solver='sag',max_iter = 100000) # solver = sag’, ‘saga’ ,'liblinear'
            return clf, xtrain, ytrain, xtest, ytest

    embedding_matrix = emb_obj.get_embedding_matrix()
    train_size = 1 - parameters["fraction_test"]
    # try:
    clf, xtrain, y_true_train, xtest, y_true_test= create_clf(
        emb_obj,
        embedding_matrix,
        parameters["dataset"],
        gt_modules_dict,
        train_size
        )
    clf.fit(xtrain, y_true_train)
    # except ValueError:
    #     print("split lead to some classes not being represented")
    #     continue
    y_pred_test = clf.predict(xtest)
    y_pred_train = clf.predict(xtrain)

    eval_test = calculate_metrics(y_true_test, y_pred_test)
    eval_test = {k+"_test":v for k,v in eval_test.items()}
    eval_train = calculate_metrics(y_true_train, y_pred_train)
    eval_train= {k+"_train":v for k,v in eval_train.items()}
    output_dict = {**parameters,**eval_test,**eval_train}
    return output_dict

