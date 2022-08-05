from src.learning.generate.generate_synthetic_graph_params import BIPARTITE_DEFAULT_RANGE, BIPARTITE_DEFAULT_PARAMS
from src.learning.generate.generate_synthetic_graph import GraphParams, generateSyntheticBipartiteGraph, get_graph_metric
# Modeling
from src.learning.models.api.box_api import box
from src.learning.models.api.matrix_factorization_api import matrix_factorization
from src.learning.models.api.vector_api import vector
from src.learning.models.api.hyperplane_api import hyperplane
from src.learning.models.api.transE_api import transE
from src.learning.models.api.ordered_embedding_api import ordered_embedding
from src.learning.models.api.hyperbolic_cone_api import hyperbolic_cone
#util
import json
import os
import pandas as pd
from pathlib import Path

def save_experiment_data_paths(data_paths, experiment_name):
    with open(os.path.join("experiments", experiment_name), "w") as file:
        for data_path in data_paths:
            file.write(data_path+"\n")
    file.close()

def load_experiment_data_paths(experiment_name):
    data_paths  = []
    with open(os.path.join("experiments", experiment_name), "r") as file:
        lines = file.readlines()
        data_paths = [line.strip() for line in lines]
    return data_paths


def run_experiment(data_paths, n_params):
    for data_path in data_paths:
        run_experiment_on_one_graph(data_path, n_params)

def run_experiment_on_one_graph(data_path, n_params):
    assert(n_params%2==0)
    assert(os.path.exists(data_path))

    transE(data_path, dim=n_params)
    hyperplane(data_path, dim=n_params-1)
    vector(data_path, dim=n_params)
    matrix_factorization(data_path, dim=n_params)
    box(data_path, dim=n_params//2, box_type = "gumbel_box", config= {"epochs":1000})
    box(data_path, dim=n_params//2, box_type = "tbox", config= {"epochs":1000})


def generate_graphs_for_experiment(test_param, save_dir=None):
    data_paths = []
    for i in BIPARTITE_DEFAULT_RANGE[test_param]:
        params = GraphParams({**BIPARTITE_DEFAULT_PARAMS, test_param: i})
        g, adj_matrix, interaction_matrix, data_path = generateSyntheticBipartiteGraph(params, save_graph = True, save_dir = save_dir)
        data_paths.append(data_path)
    return data_paths

def generate_graphs_for_experiment(test_param_1, test_param_2, save_dir=None):


def list_experiment_dirs(rootdir):
    ret = []
    for path in Path(rootdir).iterdir():
        if path.is_dir():
            if "results" in os.listdir(path):
                ret.append(path)
            else:
                ret += list_experiment_dirs(path)
    return ret


def collect_experiment_results(data_dir):
    """collect result from all experiments in the data_dir directory"""
    experiment_results_df = []
    exp_roots = list_experiment_dirs(data_dir)
    for data_path in exp_roots:
        graph_metric =  get_graph_metric(data_path)
        for n_param in os.listdir(os.path.join(data_path, "results")):
            if not os.path.isdir(os.path.join(data_path, "results", n_param)): continue
            model_best_F1 = get_models_best_F1(os.path.join(data_path, "results", n_param))
            n_param = int(n_param.split("=")[1])
            experiment_results_df.append({**graph_metric, **model_best_F1, "n_param":n_param })
    return pd.DataFrame(experiment_results_df)



def get_models_best_F1(result_dir):
    model_best_F1 = {}
    for model in os.listdir(result_dir):
        if not os.path.isdir(os.path.join(result_dir, model)): continue
        for trial in os.listdir(os.path.join(result_dir, model)):
            if os.path.isdir(os.path.join(result_dir, model, trial)):
                model_best_F1[model] = 0
                for file in os.listdir(os.path.join(result_dir, model, trial)):
                    if file.endswith(".json"):
                        metrics = json.load(open(os.path.join(result_dir, model, trial, file), "r"))
                        model_best_F1[model] = max(model_best_F1[model], metrics["F1"])
    return model_best_F1