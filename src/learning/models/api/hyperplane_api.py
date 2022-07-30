from src.learning.models.loss import HYPERPLANE
from src.learning.training.train import TrainingConfig, train
from src.learning.models.model import BIPARTITE_DEFAULT_TRAINING_CONFIG
from src.learning.generate.graph_utils import get_target_graph

def hyperplane(data_path, dim = 3):
    train_g = get_target_graph(data_path)
    config = TrainingConfig(BIPARTITE_DEFAULT_TRAINING_CONFIG)
    config.geometry = HYPERPLANE()
    config.train_g = train_g
    config.dim = dim
    model, metrics, prediction_coo = train(config)
    return prediction_coo