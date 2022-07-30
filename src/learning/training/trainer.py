from src.learning.training.evaluate import get_metrics
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch
import wandb
import matplotlib.pyplot as plt
import sklearn
from loguru import logger
from collections import defaultdict


def get_optimizer(config, model):
    return torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        eps=1e-6,
        weight_decay=config.weight_decay,
    )


def get_lr_scheduler(config, optim):
    lr_scheduler = config.lr_scheduler
    if lr_scheduler == "exp":
        lr_scheduler = ExponentialLR(optim, gamma=config.lr_gamma)
    elif lr_scheduler == "step":
        lr_scheduler = StepLR(
            optim, step_size=config.lr_step_size, gamma=config.lr_gamma
        )
    else:
        raise ValueError("lr scheduler not recognized")
    return lr_scheduler


class Logger:
    def __init__(self, print_every=1):
        self.epoch_history = []  # epoch: metrics
        self.metric_history = defaultdict(list)
        self.logs = {}
        self.print_every = print_every

    def log(self, log_dict, prefix="", epoch=None):
        for key, value in log_dict.items():
            key = prefix + "_" + key
            if key not in self.logs or self.logs[key] == None:
                self.logs[key] = []
            self.logs[key].append(value)

    def best(self, key=None):
        metrics = {key: max(value) for key, value in self.metric_history.items()}
        if key == None:
            return metrics
        return metrics[key]

    def get_metrics(self, epoch):
        return self.epoch_history[epoch]

    def commit(self):
        metrics = {}
        for key in self.logs:
            if self.logs[key]:
                metrics[key] = sum(self.logs[key]) / len(self.logs[key])
                self.metric_history[key].append(metrics[key])
                self.logs[key] = None
        self.epoch_history.append(metrics)
        if len(self.epoch_history) % self.print_every == 0:
            logger.info(f"epoch = {len(self.epoch_history)}")
            for key in metrics:
                logger.info(f"{key} = {metrics[key]}")

    def get_history(self, metric):
        if metric not in self.metric_history:
            return None
        return self.metric_history[metric]


def inference(model, dataloader):
    model.eval()
    losses = []
    edge_probs = []
    ground_truths = []
    batch_sizes = []
    for batch in dataloader:
        loss, edge_prob, ground_truth = model(batch)
        losses.append(loss.detach().cpu().numpy())
        batch_sizes.append(len(batch))
        edge_probs.append(edge_prob)
        ground_truths.append(ground_truth)
    loss = sum(losses) / sum(batch_sizes)

    edge_prob = torch.concat(edge_probs).detach().cpu()
    ground_truth = torch.concat(ground_truths)
    return {
        **get_metrics(edge_prob, ground_truth),
        "loss": loss,
    }


def debug(model, train_g):
    print("debug...")
    u_embs, _ = model.get_embs(list(train_g.u2id.keys()))
    v_embs, _ = model.get_embs(list(train_g.v2id.keys()))
    pos_u, pos_v = train_g.u2id[train_g.edges_u[0]], train_g.v2id[train_g.edges_v[0]]
    pos_emb, neg_emb = u_embs[pos_u], v_embs[pos_v]
    dot_product = pos_emb.T.dot(neg_emb)
    print(dot_product)


def trainer(model, train_loader, val_loader, config):

    if config.wandb_id:
        wandb.init(
            project="bipartite_graph_representation",
            entity="wxd",
            id=config.wandb_id + config.exp_id,
            config=config.get_wandb_config(),
        )
        wandb.watch(model)

    optim = get_optimizer(config, model)
    lr_scheduler = get_lr_scheduler(config, optim)
    logger = Logger(config.print_every)

    # initial performance
    metrics = inference(model, train_loader)
    logger.log(metrics, prefix="train")

    if val_loader:
        metrics = inference(model, val_loader)
        logger.log(metrics, prefix="val")

    logger.commit()

    if config.wandb_id:
        wandb.log({**logger.get_metrics(epoch=0), "epoch": 0})

    # stop if val loss does not improve in the last "max_did_not_improve" epochs
    did_not_improve = 0
    best_val_accuracy = 0

    for e in range(1, config.epochs + 1):
        # ----------------------- train ----------------------- #
        model.train()
        edge_probs = []
        ground_truths = []
        for batch in train_loader:
            optim.zero_grad()
            batch_loss, edge_prob, ground_truth = model(batch)
            batch_loss.backward()
            edge_probs.append(edge_prob)
            ground_truths.append(ground_truth)
            logger.log({"loss": batch_loss.detach().cpu().numpy()}, prefix="train")
            if batch_loss.detach().cpu().isnan():
                break
            optim.step()
        lr_scheduler.step()

        training_metrics = get_metrics(
            torch.concat(edge_probs).detach().cpu(), torch.concat(ground_truths)
        )
        logger.log(training_metrics, prefix="train")
        debug(model, config.train_g)
        # ---------------------- validate -------------------- #
        if val_loader:
            validation_metrics = inference(model, val_loader)
            logger.log(metrics, prefix="val")
        else:
            validation_metrics = training_metrics

        if validation_metrics["accuracy"] < best_val_accuracy:
            did_not_improve += 1
            if did_not_improve > config.max_did_not_improve:
                print(f"early stopping at epoch {e}...")
                break
        else:
            best_val_accuracy = validation_metrics["accuracy"]
            did_not_improve = 0

        logger.commit()

        if config.wandb_id:
            wandb.log(
                {
                    **logger.get_metrics(e),
                    "epoch": e,
                }
            )

    if config.wandb_id:
        wandb.run.summary["val_accuracy"] = logger.best("val_accuracy")
        wandb.run.summary["train_accuracy"] = logger.best("train_accuracy")
        wandb.finish()

    # ----------- Done with training ----------------------- #
    if config.visualize:
        print("train_loss", logger.get_history("train_loss"))
        print("val_loss", logger.get_history("val_loss"))

        plt.plot(logger.get_history("train_loss"), label="train")
        if logger.get_history("val_loss"):
            plt.plot(logger.get_history("val_loss"), label="val")
        plt.title("loss")
        plt.legend()
        plt.show()

        plt.plot(logger.get_history("train_accuracy"), label="train")
        if logger.get_history("val_accuracy"):
            plt.plot(logger.get_history("val_accuracy"), label="val")
        plt.title("accuracy")
        plt.legend()
        plt.show()

    return logger.best()
