'''api from https://nimfa.biolab.si/'''

import nimfa
import numpy as np
import os
from scipy.sparse import load_npz, save_npz
from src.learning.generate.graph_utils import get_target_graph

# from graph_modeling.training.metrics import calculate_optimal_metrics

def matrix_factorization(data_path, dim=4):

    target_g = get_target_graph(data_path)

    sparse_interaction_matrix = target_g.get_interaction_matrix()

    #Non-negative matrix factorization
    nmf = nimfa.Nmf(sparse_interaction_matrix,  rank=dim, max_iter=1000)
    nmf_fit = nmf()
    predicted_interaction_matrix = nmf.fitted()

    folder_name = f"dim={dim}"
    output_dir = os.path.join(data_path, "results", "matrix_factorization", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    save_npz(os.path.join(output_dir, "prediction"), predicted_interaction_matrix)

    # metrics = calculate_optimal_metrics(np.array(sparse_matrix.todense()).flatten().squeeze(),
    #                                np.array(predicted_matrix.todense()).flatten().squeeze())

    return predicted_interaction_matrix
