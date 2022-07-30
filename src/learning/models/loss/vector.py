
from src.learning.models.loss.loss import Geometry
import torch
from torch.nn import functional as F

class VECTOR(Geometry):
    def __init__(self):
        self.is_vector = True
        # self.loss_type = Loss.BCE
        # self.agg_func = AggragationFunctions.MEAN
        self.negative_weight = 0.5
    def get_dim(self, n_parameters):
        return n_parameters

    def to_string(self):
        return "VECTOR"

    def get_loss(
        self,
        pos_u_embs,
        pos_u_offset_embs,
        pos_v_embs,
        pos_v_offset_embs,
        neg_u_embs,
        neg_u_offset_embs,
        neg_v_embs,
        neg_v_offset_embs,
    ):

        log_prob_pos = torch.log(
            torch.sigmoid(torch.sum(pos_u_embs * pos_v_embs, axis=1))
        )
        log_prob_neg = torch.log(
            torch.sigmoid(torch.sum(neg_u_embs * neg_v_embs, axis=1))
        )

        pos_loss = -log_prob_pos
        neg_loss = -torch.log(
            1 - torch.sigmoid(torch.sum(neg_u_embs * neg_v_embs, axis=1))
        )
        logit_prob_neg = log_prob_neg + neg_loss
        weights = F.softmax(logit_prob_neg, dim=-1)
        weighted_average_neg_loss = (weights * neg_loss).sum(dim=-1)

        loss =  (
            1 - self.negative_weight
        ) * pos_loss + self.negative_weight * weighted_average_neg_loss

        loss = torch.mean(loss)
        return loss

    def get_edge_prob(
        self,
        pos_u_embs,
        pos_u_offset_embs,
        pos_v_embs,
        pos_v_offset_embs,
        neg_u_embs,
        neg_u_offset_embs,
        neg_v_embs,
        neg_v_offset_embs,
    ):

        pos_edge_prob = torch.sigmoid(torch.sum(pos_u_embs * pos_v_embs, axis=1))
        neg_edge_prob = torch.sigmoid(torch.sum(neg_u_embs * neg_v_embs, axis=1))
        edge_prob = torch.concat([pos_edge_prob, neg_edge_prob])
        assert torch.min(edge_prob) >= 0
        return edge_prob
