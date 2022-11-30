import os
from timeit import repeat
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from manage_experiments import *
from multiprocessing import Pool


parameters = {
    "hidden_dims":[None,None,None],
    "dataset":None,
    "method": "GCN",
    "fraction_test":.3,
    "n_epochs":5000,
    "learning_rate":0.001,
    'weighted':True,
    'directed':True,
    'weight_decay':1e-3,
    'opt_order':2,
    }

datasets = [
    "temporal_clusters.ngram",
    "sms.ngram",
    "highschool2011_delta4_ts900_full.ngram",
    "highschool2012_delta4_ts900_full.ngram",
    "workplace2016_delta4_ts900_full.ngram",
    "workplace2018_delta4_ts900_full.ngram",
    "hospital_delta4_ts900_full.ngram",
    ]

d = 16
h1s = [4,8,16,32]
h2s = [4,8,16,32]

# opt_order_dict = {}
repeat_experiment = 10
n_parallel = 2
fold = "res_GCN/"


if __name__ == '__main__':
    for dataset in datasets:
        parameters["dataset"] = dataset
        for v1 in h1s:
            for v2 in h2s:
                parameters["hidden_dims"][0] = v1
                parameters["hidden_dims"][1] = v2
                parameters["hidden_dims"][2] = d
                create_tasks(
                    parameters,
                    repeat_experiment = repeat_experiment,
                    folder = fold
                )
    list_unfinished = find_unfinished(fold)
    with Pool(n_parallel) as p:
        list(p.map(experiment_node_classification, list_unfinished))