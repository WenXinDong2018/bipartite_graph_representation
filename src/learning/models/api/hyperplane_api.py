from src.learning.models.loss import HYPERPLANE
from src.learning.training.train import TrainingConfig, train
from src.learning.models.model import BIPARTITE_DEFAULT_TRAINING_CONFIG
from src.learning.generate.graph_utils import get_target_graph, NpEncoder
import json
import os
from scipy.sparse import load_npz, save_npz


def hyperplane(data_path, dim = 3):
    train_g = get_target_graph(data_path)
    config = TrainingConfig(BIPARTITE_DEFAULT_TRAINING_CONFIG)
    config.geometry = HYPERPLANE()
    config.train_g = train_g
    config.dim = dim
    folder_name = config.to_string()
    output_dir = os.path.join(data_path, "results", f"n_params={dim+1}", config.geometry.to_string(), folder_name)
    metrics_path = os.path.join(output_dir, "metrics.json")

    #if already exist simply return existing metrics
    if os.path.exists(output_dir):
        prediction_coo = load_npz(os.path.join(output_dir, "interaction_matrix.npz"))
        metrics = json.load(open(metrics_path, "r"))
        return metrics, prediction_coo

    #train
    model, metrics, prediction_coo = train(config)

    #save
    os.makedirs(output_dir, exist_ok=True)
    save_npz(os.path.join(output_dir, "interaction_matrix"), prediction_coo)
    json.dump(metrics, open(metrics_path, "w"), cls=NpEncoder)
    return metrics, prediction_coo
