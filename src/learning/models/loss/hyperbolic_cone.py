from src.learning.models.loss.loss import Geometry
import torch
from torch.nn import functional as F


class HYPERBOLIC_CONE(Geometry):
    def __init__(self):
        self.is_vector = False
        self.negative_weight = 0.5
        self.K = 0.1
        self.margin = 0.01
    def get_dim(self, n_parameters):
        return n_parameters - 1

    def to_string(self):
        return "HYPERBOLIC CONE"


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
        '''
            hyperbolic entailment cone paper: http://proceedings.mlr.press/v80/ganea18a/ganea18a.pdf
            loss: max(0, E(u, v) - E(u',v') + 1)
            where
            E(u,v) = max(0, f(u - g(u, v)))
            f(u) = arcsin(K*(1-||u||^2)/||u||)
            g(x,y) = arccos((<x,y>(1+||x||^2)-||x||^2(1+||y||^2))/(||x||||x-y|| sqrt(1+||x||^2||y||^2-2<x,y>)))

        '''
        vectors_u = torch.concat([pos_u_embs, neg_u_embs])
        vectors_v = torch.concat([pos_v_embs, neg_v_embs])

        norm_u = torch.linalg.norm(vectors_u, axis=1)
        norms_v = torch.linalg.norm(vectors_v, axis=1)
        euclidean_dists = torch.linalg.norm(vectors_u - vectors_v, axis=1)
        dot_prod = (vectors_u * vectors_v).sum(axis=1)
        cos_angle_child = (dot_prod * (1 + norm_u ** 2) - norm_u ** 2 * (1 + norms_v ** 2)) /\
                            (norm_u * euclidean_dists * torch.sqrt(1 + norms_v ** 2 * norm_u ** 2 - 2 * dot_prod))
        angles_psi_parent = torch.arcsin(self.K * (1 - norm_u**2) / norm_u) # vector
        # To avoid numerical errors
        EPS = 1e-7
        clipped_cos_angle_child = torch.max(cos_angle_child, torch.tensor(-1 + EPS))
        clipped_cos_angle_child = torch.min(clipped_cos_angle_child, torch.tensor(1 - EPS))
        angles_child = torch.arccos(clipped_cos_angle_child)  # 1 + neg_size

        energy_vec = torch.max(torch.tensor(0), angles_child - angles_psi_parent)
        positive_term = energy_vec[:len(pos_u_embs)]
        negative_terms = energy_vec[len(pos_u_embs):]
        pos_loss = positive_term
        neg_loss = torch.max(torch.tensor(0), self.margin - negative_terms)
        loss = torch.concat([pos_loss, neg_loss])
        return torch.mean(loss)


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
        vectors_u = torch.concat([pos_u_embs, neg_u_embs])
        vectors_v = torch.concat([pos_v_embs, neg_v_embs])


        norm_u = torch.linalg.norm(vectors_u, axis=1)
        norms_v = torch.linalg.norm(vectors_v, axis=1)
        euclidean_dists = torch.linalg.norm(vectors_u - vectors_v, axis=1)
        dot_prod = (vectors_u * vectors_v).sum(axis=1)

        cos_angle_child = (dot_prod * (1 + norm_u ** 2) - norm_u ** 2 * (1 + norms_v ** 2)) /\
                            (norm_u * euclidean_dists * torch.sqrt(1 + norms_v ** 2 * norm_u ** 2 - 2 * dot_prod))

        angles_psi_parent = torch.arcsin(self.K * (1 - norm_u**2) / norm_u) # vector

        # To avoid numerical errors
        EPS = 1e-7
        clipped_cos_angle_child = torch.max(cos_angle_child, torch.tensor(-1 + EPS))
        clipped_cos_angle_child = torch.min(clipped_cos_angle_child, torch.tensor(1 - EPS))
        angles_child = torch.arccos(clipped_cos_angle_child)  # 1 + neg_size

        energy_vec = torch.max(torch.tensor(0), angles_child - angles_psi_parent)
        return -energy_vec
