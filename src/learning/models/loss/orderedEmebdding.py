
import torch
from torch.nn import functional as F

class ORDERED_EMBEDDING():
    def __init__(self):
        self.is_vector = True
    def get_dim(self, n_parameters):
        return n_parameters

    def to_string(self):
        return "ORDERED EMBEDDING"

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
            Ordered Embedding paper: https://arxiv.org/pdf/1511.06361.pdf
            original loss: E(u, v) + max(0, 1- E(u',v')) where
            E = ||max(0, v-u)||_2^2

            adapted loss: max(0, E(u, v) - E(u',v') + 1)
            this allows for negative sampling ratio to be higher than 1
        '''

        pos_E = torch.pow(torch.linalg.norm(torch.max(torch.tensor(0), pos_v_embs - pos_u_embs), axis=1),2)
        neg_E = torch.pow(torch.linalg.norm(torch.max(torch.tensor(0), neg_v_embs - neg_u_embs), axis=1),2)

        N = len(pos_E)
        NK = len(neg_E)
        pos_E = pos_E.reshape(N, 1)
        neg_E = neg_E.reshape(N, -1)
        loss  = torch.max(torch.tensor(0), pos_E - neg_E + 1).flatten()
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
        #returns the logits, not probabilities.
        pos_edge_prob = -torch.pow(torch.linalg.norm(torch.max(torch.tensor(0), pos_v_embs - pos_u_embs), axis=1),2)
        neg_edge_prob = -torch.pow(torch.linalg.norm(torch.max(torch.tensor(0), neg_v_embs - neg_u_embs), axis=1),2)

        edge_prob = torch.concat([pos_edge_prob, neg_edge_prob])
        return edge_prob
