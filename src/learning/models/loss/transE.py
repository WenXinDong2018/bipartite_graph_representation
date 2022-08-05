
import torch
from torch.nn import functional as F

class TRANS_E():
    def __init__(self):
        self.is_vector = True
    def get_dim(self, n_parameters):
        return n_parameters

    def to_string(self):
        return "TRANS_E"

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
            TransE paper: https://papers.nips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf
            loss: max(0, pos_dist - neg_dist + 1) where
            dist = ||h-t||_2
            This is the same as rotateE since we are dealing with monorelational data
        '''

        pos_dist = torch.linalg.norm(pos_u_embs - pos_v_embs, axis=1)
        neg_dist = torch.linalg.norm(neg_u_embs - neg_v_embs, axis=1)
        N = len(pos_dist)
        NK = len(neg_dist)
        pos_dist = pos_dist.reshape(N, 1)
        neg_dist = neg_dist.reshape(N, -1)
        loss  = torch.max(torch.tensor(0), pos_dist - neg_dist + 1).flatten()
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
        pos_edge_prob = -torch.linalg.norm(pos_u_embs - pos_v_embs, axis=1)
        neg_edge_prob = -torch.linalg.norm(neg_u_embs - neg_v_embs, axis=1)

        edge_prob = torch.concat([pos_edge_prob, neg_edge_prob])
        return edge_prob
