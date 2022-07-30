from src.learning.models.loss.loss import Geometry
import torch
from torch.nn import functional as F


class HYPERPLANE(Geometry):
    def __init__(self):
        self.is_vector = False
        # self.loss_type = Loss.DOT
        # self.agg_func = AggragationFunctions.HINGE
        self.negative_weight = 0.5
    def get_dim(self, n_parameters):
        return n_parameters - 1

    def to_string(self):
        return "HYPERPLANE"


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
        '''max(0, 1-y(wx+b))'''
        pos_dist = torch.sum(pos_u_embs * pos_v_embs, axis=1) + pos_u_offset_embs
        neg_dist = torch.sum(neg_u_embs * neg_v_embs, axis=1) + neg_u_offset_embs
        pos_loss = torch.max(torch.tensor(0), 1-pos_dist)
        neg_loss = torch.max(torch.tensor(0), 1+neg_dist)

        weights = F.softmax(neg_dist, dim=-1)
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
        pos_dist = torch.sum(pos_u_embs * pos_v_embs, axis=1) + pos_u_offset_embs
        neg_dist = torch.sum(neg_u_embs * neg_v_embs, axis=1) + neg_u_offset_embs
        pos_edge_prob = (pos_dist > 0)+0.0
        neg_edge_prob = (neg_dist > 0) + 0.0
        edge_prob = torch.concat([pos_edge_prob, neg_edge_prob])
        assert torch.min(edge_prob) >= 0
        return edge_prob
