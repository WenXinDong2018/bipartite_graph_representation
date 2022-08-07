import torch
import torch.nn as nn
from src.learning.generate.generate_synthetic_graph import generateSyntheticBipartiteGraph, GraphParams
from src.learning.models.loss.vector import VECTOR
from src.learning.models.loss.loss import calc_loss

BIPARTITE_DEFAULT_TRAINING_CONFIG = {

    "epochs": 100,
    "lr": 1e-1,
    "lr_scheduler": "exp",
    "lr_step_size": 1,
    "lr_gamma": 0.9,      #multiply lr by gamma every step_size epochs
    "weight_decay": 0,  #L2 regularization
    "max_did_not_improve": 5,
    "print_every": 1,
    "train_g":generateSyntheticBipartiteGraph(GraphParams())[0],
    "val_g": None,
    "eval_g":None,
    "neg_sampler": "corrupt_tail",
    "geometry": VECTOR(),
    "dim": 4,
    "k": 128,
    "save_dir": None,
    "seed":0,
    "wandb_id":None,
    "batch_size": 32,
    "visualize": True
}

class BipartiteModel(nn.Module):
    def __init__(self, config):

        super().__init__()

        self.train_g = config.train_g
        self.val_g = config.val_g

        self.n_nodes = config.train_g.get_n_nodes()
        self.embs = nn.Parameter()
        self.offset_embs = nn.Parameter()
        self.dim = config.dim

        self.random_init_max = 1
        self.random_init_min = -1

        self.geometry = config.geometry
        self.setup()

    def setup(self):
        h = self.random_init_max
        l = self.random_init_min
        self.embs = torch.nn.Parameter(
            data=torch.rand((self.n_nodes, self.dim)) * (h - l) + l, requires_grad=True
        )
        self.offset_embs = torch.nn.Parameter(
            data=torch.rand((self.n_nodes, 1)) * (h - l) + l, requires_grad=True
        )

    def get_embs(self, nodes):
        """
        usage: get_embs(["a1", "b2"])
        """
        node_ids = self.train_g.get_nodes_id(nodes)
        min_embs = self.embs[node_ids]
        offset_embs = self.offset_embs[node_ids]
        return min_embs, offset_embs

    def forward(self, batch):

        pos_edge_u = batch["pos"][0]
        pos_edge_v = batch["pos"][1]
        neg_edge_u = torch.concat(batch["neg"][0])
        neg_edge_v = torch.concat(batch["neg"][1])

        pos_u_embs, pos_u_offset_embs = (
            self.embs[pos_edge_u],
            self.offset_embs[pos_edge_u].squeeze(),
        )
        pos_v_embs, pos_v_offset_embs = (
            self.embs[pos_edge_v],
            self.offset_embs[pos_edge_v].squeeze(),
        )
        neg_u_embs, neg_u_offset_embs = (
            self.embs[neg_edge_u],
            self.offset_embs[neg_edge_u].squeeze(),
        )
        neg_v_embs, neg_v_offset_embs = (
            self.embs[neg_edge_v],
            self.offset_embs[neg_edge_v].squeeze(),
        )


        loss, edge_prob = calc_loss(
            self.geometry,
            pos_u_embs,
            pos_u_offset_embs,
            pos_v_embs,
            pos_v_offset_embs,
            neg_u_embs,
            neg_u_offset_embs,
            neg_v_embs,
            neg_v_offset_embs,
        )


        ground_truth = torch.concat([torch.ones(len(pos_u_embs)), torch.zeros(len(neg_u_embs))])
        return loss, edge_prob, ground_truth
