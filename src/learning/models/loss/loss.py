import torch
from enum import IntEnum
from torch.nn import functional as F


class AggragationFunctions(IntEnum):
    MEAN = 0
    MARGIN = 1
    PAIRWISE = 2
    PAIRWISE_RANKING = 3
    CONTRASTIVE = 4
    HINGE = 5


class Loss(IntEnum):
    COSINE = 0
    DOT = 1
    L2 = 2
    L1 = 3
    BCE = 4


def get_distance(u, v, loss_type, pos_pair=True):
    """get distance between vectors u and v"""
    if Loss.COSINE == loss_type:
        dist = -torch.sum(u * v, axis=1) / (
            torch.linalg.norm(u, axis=1) * torch.linalg.norm(v, axis=1)
        )
    elif Loss.DOT == loss_type:
        dist = -torch.sum(u * v, axis=1)
    elif Loss.L2 == loss_type:
        dist = torch.linalg.norm(u - v, axis=1)
    elif Loss.L1 == loss_type:
        dist = torch.sum(torch.abs(u - v), axis=1)
    elif Loss.BCE == loss_type:
        if pos_pair:
            dist = -torch.log(torch.sigmoid(torch.sum(u * v, axis=1)))
        else:
            dist = -torch.log(1 - torch.sigmoid(torch.sum(u * v, axis=1)))
    else:
        raise ValueError("loss_type not recognized", loss_type)

    return dist


def average_loss(pos_loss, neg_loss):
    loss = torch.concat([pos_loss, neg_loss])
    return torch.mean(loss)


def pairwise_loss(pos_dist, neg_dist, margin):
    """pos_loss: shape (N, 1) neg_los: shape (NK, 1)"""
    """when using pairwise loss, make sure use corruptTailNegativeSampler"""
    N = len(pos_dist)
    NK = len(neg_dist)
    pos_dist = pos_dist.reshape(N, 1)
    neg_dist = neg_dist.reshape(N, -1)
    """want  neg_dist - pos_dist > margin """
    pairwise_loss = get_max_margin(margin, neg_dist - pos_dist).flatten()
    assert len(pairwise_loss) == NK
    pairwise_loss = torch.mean(pairwise_loss)
    return pairwise_loss


def pairwise_ranking_loss(pos_dist, neg_dist, neg_dist_radius):
    """want positive distance = 0 and"""
    """     neg_dist < neg_dist_radius"""
    neg_loss = get_max_margin(neg_dist_radius, neg_dist)
    return average_loss(pos_dist, neg_loss)


def hinge_loss(pos_dist, neg_dist):
    """want pos_dist < -1"""
    pos_loss = get_max_margin(pos_dist, -1)
    """want neg_dist > 1"""
    neg_loss = get_max_margin(1, neg_dist)
    return average_loss(pos_loss, neg_loss)


def get_max_margin(A, B):
    """helper function: want (A+eps)<=B"""
    eps = 1e-5
    return torch.max(torch.tensor(0), (A + eps) - B)


def margin_loss(pos_dist, neg_dist, pos_dist_radius, neg_dist_radius):
    """want pos_dist < pos_dist_radius"""
    pos_loss = get_max_margin(pos_dist, pos_dist_radius)
    """want neg_dist > neg_dist_radius"""
    neg_loss = get_max_margin(neg_dist_radius, neg_dist)
    return average_loss(pos_loss, neg_loss)


def contrastive_loss(pos_dist, neg_dist):
    pos_sim = -pos_dist
    neg_sim = -neg_dist
    N = len(pos_sim)
    NK = len(neg_sim)
    pos_sim = pos_sim.reshape(N, 1)
    neg_sim = neg_sim.reshape(N, -1)
    loss = -pos_sim + torch.logsumexp(neg_sim, axis=1)
    loss = loss.flatten()
    assert len(loss) == NK
    return torch.mean(loss)


def aggregate_loss(pos_dist, neg_dist, agg_fun):

    if agg_fun == AggragationFunctions.MEAN:
        return average_loss(pos_dist, -neg_dist)
    elif agg_fun == AggragationFunctions.HINGE:
        return hinge_loss(pos_dist, neg_dist)
    elif agg_fun == AggragationFunctions.MARGIN:
        return margin_loss(pos_dist, neg_dist, 1, 1)
    elif agg_fun == AggragationFunctions.PAIRWISE:
        margin = 1
        return pairwise_loss(pos_dist, neg_dist, margin)
    elif agg_fun == AggragationFunctions.PAIRWISE_RANKING:
        neg_dist_radius = 1
        return pairwise_ranking_loss(pos_dist, neg_dist, neg_dist_radius)
    elif agg_fun == AggragationFunctions.CONTRASTIVE:
        return contrastive_loss(pos_dist, neg_dist)
    else:
        raise ValueError("unrecognized agg function", agg_fun)


def aggregate_geometric_loss(
    pos_dist, neg_dist, pos_u_offset_embs, neg_u_offset_embs, agg_fun, loss_type
):

    if agg_fun == AggragationFunctions.HINGE:
        return hinge_loss(pos_dist + pos_u_offset_embs, neg_dist + neg_u_offset_embs)
    elif agg_fun == AggragationFunctions.MARGIN:
        return margin_loss(
            pos_dist, neg_dist, pos_u_offset_embs - 1, neg_u_offset_embs + 1
        )
    elif agg_fun == AggragationFunctions.MEAN:
        return margin_loss(pos_dist, neg_dist, pos_u_offset_embs, neg_u_offset_embs)
    elif agg_fun == AggragationFunctions.PAIRWISE_RANKING:
        return pairwise_ranking_loss(pos_dist, neg_dist, neg_u_offset_embs)
    else:
        raise ValueError(
            "unrecognized aggregation function or does not apply to geometries", agg_fun
        )


class Geometry:
    def __init__(self):
        pass

    def to_string(self):
        return ""

    def get_dim(self):
        return 0




class L2_SPHERE(Geometry):
    def __init__(self):
        self.is_vector = False
        self.loss_type = Loss.L2
        self.agg_func = AggragationFunctions.PAIRWISE_RANKING

    def get_dim(self, n_parameters):
        return n_parameters - 1

    def to_string(self):
        return "L2_SPHERE"


class L1_SPHERE(Geometry):
    def __init__(self):
        self.is_vector = False
        self.loss_type = Loss.L1
        self.agg_func = AggragationFunctions.PAIRWISE_RANKING

    def get_dim(self, n_parameters):
        return n_parameters - 1

    def to_string(self):
        return "L1_SPHERE"


class CONE(Geometry):
    def __init__(self):
        self.is_vector = False
        self.loss_type = Loss.COSINE
        self.agg_func = AggragationFunctions.PAIRWISE_RANKING

    def get_dim(self, n_parameters):
        return n_parameters - 1

    def to_string(self):
        return "CONE"



def calc_loss(
    geometry,
    pos_u_embs,
    pos_u_offset_embs,
    pos_v_embs,
    pos_v_offset_embs,
    neg_u_embs,
    neg_u_offset_embs,
    neg_v_embs,
    neg_v_offset_embs,
):
    return geometry.get_loss(
        pos_u_embs,
        pos_u_offset_embs,
        pos_v_embs,
        pos_v_offset_embs,
        neg_u_embs,
        neg_u_offset_embs,
        neg_v_embs,
        neg_v_offset_embs,
    ), geometry.get_edge_prob(
        pos_u_embs,
        pos_u_offset_embs,
        pos_v_embs,
        pos_v_offset_embs,
        neg_u_embs,
        neg_u_offset_embs,
        neg_v_embs,
        neg_v_offset_embs,
    )

    # pos_dist = get_distance(pos_u_embs, pos_v_embs, loss_type, pos_pair=True)
    # neg_dist = get_distance(neg_u_embs, neg_v_embs, loss_type, pos_pair=False)
    # print("pos_dist", pos_dist)
    # print("neg_dist", neg_dist)
    # if is_vector:  # vector loss
    #     return aggregate_loss(pos_dist, neg_dist, agg_func), get_edge_prob(
    #         pos_dist, neg_dist, loss_type
    #     )
    # else:  # geometric loss
    #     return aggregate_geometric_loss(
    #         pos_dist,
    #         neg_dist,
    #         pos_u_offset_embs,
    #         neg_u_offset_embs,
    #         agg_func,
    #         loss_type,
    #     ), get_edge_prob_geometric(
    #         pos_dist, neg_dist, pos_u_offset_embs, neg_u_offset_embs, loss_type
    #     )


def get_edge_prob_geometric(
    pos_dist, neg_dist, pos_u_offset_embs, neg_u_offset_embs, loss_type
):

    if Loss.COSINE == loss_type:
        pos_edge_prob = -pos_dist >= -pos_u_offset_embs
        neg_edge_prob = -neg_dist >= -neg_u_offset_embs
    elif Loss.DOT == loss_type:
        # hyperplane
        pos_edge_prob = (-pos_dist + pos_u_offset_embs) >= 0
        neg_edge_prob = (-neg_dist + neg_u_offset_embs) >= 0
    elif Loss.L2 == loss_type:
        pos_edge_prob = pos_dist <= pos_u_offset_embs
        neg_edge_prob = neg_dist <= neg_u_offset_embs
    elif Loss.L1 == loss_type:
        pos_edge_prob = pos_dist <= pos_u_offset_embs
        neg_edge_prob = neg_dist <= neg_u_offset_embs
    else:
        raise ValueError("loss_type not recognized", loss_type)
    edge_prob = torch.concat([pos_edge_prob, neg_edge_prob])
    assert torch.min(edge_prob) >= 0
    return edge_prob


def get_edge_prob(pos_dist, neg_dist, loss_type):

    if Loss.COSINE == loss_type:
        pos_edge_prob = -pos_dist
        neg_edge_prob = -neg_dist
    elif Loss.DOT == loss_type:
        pos_edge_prob = -pos_dist >= 0
        neg_edge_prob = -neg_dist >= 0
    elif Loss.L2 == loss_type:
        pos_edge_prob = pos_dist <= 1
        neg_edge_prob = neg_dist <= 1
    elif Loss.L1 == loss_type:
        pos_edge_prob = pos_dist <= 1
        neg_edge_prob = neg_dist <= 1
    elif Loss.BCE == loss_type:
        pos_edge_prob = torch.exp(-pos_dist)
        neg_edge_prob = 1 - torch.exp(-pos_dist)
    else:
        raise ValueError("loss_type not recognized", loss_type)

    edge_prob = torch.concat([pos_edge_prob, neg_edge_prob])
    assert torch.min(edge_prob) >= 0
    return edge_prob
