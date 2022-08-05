'''api from https://nimfa.biolab.si/'''

import nimfa
import os
from scipy.sparse import save_npz
from src.learning.training.evaluate import compare_interaction_matrices
import json
from src.learning.generate.graph_utils import get_target_graph, NpEncoder

def matrix_factorization(data_path, dim=4):

    target_g = get_target_graph(data_path)

    sparse_interaction_matrix = target_g.get_interaction_matrix()

    #Non-negative matrix factorization
    nmf = nimfa.Nmf(sparse_interaction_matrix,  rank=dim, max_iter=1000)
    nmf_fit = nmf()
    predicted_interaction_matrix = nmf.fitted()

    folder_name = f"nmf_dim={dim}"
    output_dir = os.path.join(data_path, "results",  f"n_params={dim}","matrix_factorization", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    save_npz(os.path.join(output_dir, "interaction_matrix"), predicted_interaction_matrix)

    metrics = compare_interaction_matrices(data_path, predicted_interaction_matrix)

    metrics_path = os.path.join(output_dir, "metrics.json")
    json.dump(metrics, open(metrics_path, "w"), cls=NpEncoder)

    return metrics, predicted_interaction_matrix
