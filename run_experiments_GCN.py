
import os
from timeit import repeat
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from manage_experiments import *
from multiprocessing import Pool


parameters = {
    "hidden_dims":[32,64,16],
    "dataset":None,
    "method": "GCN",
    "fraction_test":.3,
    "n_epochs":5000,
    "learning_rate":0.001,
    'weighted':True,
    'directed':True,
    'weight_decay':1e-3,
    'p_dropout':0.4
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


repeat_experiment = 10
n_parallel = 5
fold = "res_GCN/"


if __name__ == '__main__':
    for dataset in datasets:
        parameters["dataset"] = dataset
        create_tasks(
            parameters,
            repeat_experiment = repeat_experiment,
            folder=fold
        )
    list_unfinished = find_unfinished(fold)
    with Pool(n_parallel) as p:
        list(p.map(experiment_node_classification, list_unfinished))


# select keyworkds in creation of run