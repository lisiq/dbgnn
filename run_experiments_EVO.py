
import os
from timeit import repeat
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from manage_experiments import *
from multiprocessing import Pool

from itertools import repeat

parameters = {
    "d":16,
    "dataset":None,
    "method": "EVO",
    "fraction_test":.3,
    'weighted':True,
    'directed':True,
    'p':None,
    'q':None,
    "f_aggregation": "avg", # avg - max - sum
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
ps = [1,4]
qs = [1,4]
fold = "res_EVO/"


if __name__ == '__main__':
    for dataset in datasets:
        parameters["dataset"] = dataset
        for p in ps:
            parameters["p"] = p
            for q in qs:
                parameters["q"] = q
                if p == q:
                    continue
                create_tasks(
                    parameters,
                    repeat_experiment = repeat_experiment,
                    folder=fold
                )
    list_unfinished = find_unfinished(fold)
    with Pool(n_parallel) as p:
        list(p.map(experiment_node_classification, list_unfinished))

