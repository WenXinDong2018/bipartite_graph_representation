from src.learning.models.loss.hyperbolic_cone import HYPERBOLIC_CONE
from src.learning.training.train import TrainingConfig, train
from src.learning.models.model import BIPARTITE_DEFAULT_TRAINING_CONFIG
from src.learning.generate.graph_utils import get_target_graph

def hyperbolic_cone(data_path, dim = 4):
    train_g = get_target_graph(data_path)
    config = TrainingConfig(BIPARTITE_DEFAULT_TRAINING_CONFIG)
    config.geometry = HYPERBOLIC_CONE()
    config.train_g = train_g
    config.lr = 0.01
    config.k = 10
    config.dim = dim
    model, metrics, prediction_coo = train(config)
    return prediction_coo