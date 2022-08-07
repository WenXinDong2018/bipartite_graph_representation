import torch
import datetime
import random
from loguru import logger
from src.learning.training.dataset import get_neg_sampler, GraphDataset
from src.learning.training.trainer import trainer
from src.learning.training.evaluate import evaluate
from torch.utils.data import DataLoader
from src.learning.models.model import BipartiteModel
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainingConfig:
    def __init__(self, default_config):
        for key, value in default_config.items():
            setattr(self, key, value)

    def get_dict(self):
        return {attr: value for attr, value in self.__dict__.items()}
    def get_wandb_config(self):
        config = self.get_dict()
        config["train_g"] = config["train_g"].params
        if config["val_g"]:
            config["val_g"] = config["val_g"].params
        if config["eval_g"]:
            config["eval_g"] = config["eval_g"].params
        config["geometry"] = config["geometry"].to_string()
        return config
    def to_string(self):
        string_dict = self.get_wandb_config()
        s = "_".join([f"{key}-{value}" for key, value in string_dict.items()
        if "_g" not in key and "/" not in key])
        return s

def setup_training(config):

    logger.info("setting up training")
    train_g = config.train_g
    val_g = config.val_g
    train_dataset = GraphDataset(train_g, config.k, get_neg_sampler(config.neg_sampler))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
    val_dataset, val_loader = None, None
    if val_g:
        val_dataset = GraphDataset(val_g, config.k, get_neg_sampler(config.neg_sampler))
        val_loader = DataLoader(val_dataset, shuffle=True, batch_size=config.batch_size)
    model = BipartiteModel(config)
    model.to(device)
    logger.info("finished setting up training")
    return train_loader, val_loader, model


def train(config):

    logger.info(f"training on {device}")
    logger.info(f"batch size = {config.batch_size}")
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    config.exp_id = str(datetime.datetime.now()).replace(":", "_")

    if not config.save_dir:
        config.save_dir = os.path.join("models", config.exp_id)

    train_loader, val_loader, model = setup_training(config)
    trainer(model, train_loader, val_loader, config)
    if not config.eval_g:
        config.eval_g = config.train_g
    metrics, predictions_coo = evaluate(model, config.eval_g)

    logger.info("Training complete!")
    return model, metrics, predictions_coo
