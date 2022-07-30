import numpy as np
from torch.utils.data import Dataset


def get_neg_sampler(sampler_name):
    return {
        "corrupt_tail": CorruptTailNegativeSampler,
        "random": RandomNegativeSampler,
    }[sampler_name]


class CorruptTailNegativeSampler:
    """Given a postive edge (u, v), sample k negative edges (u, *)"""

    def __init__(self, graph, k):
        self.k = k
        self.graph = graph

    def sample(self, pos_edge):
        u, v = pos_edge
        v_neg = self.graph.get_node_neg_neighbours(u)
        v_neg = list(set(v_neg) - set([u]))
        try:
            random_idxs = np.random.choice(len(v_neg), (self.k,))
            n_u = [u for i in random_idxs]
            nk_v = [v_neg[i] for i in random_idxs]
            n_u_type = [self.graph.node2type[uu] for uu in n_u]
            nk_v_type = [self.graph.node2type[vv] for vv in nk_v]
            return n_u, nk_v, n_u_type, nk_v_type
        except Exception as e:

            return [], [], [], []


class RandomNegativeSampler:
    """Given a postive edge (u, v), sample k negative edges (*, *)"""

    def __init__(self, graph, k):
        self.k = k
        self.graph = graph

    def sample(self, pos_edge=None):
        n_u, nk_v = self.graph.sample_random_negative_edges(self.k)
        n_u_type = [self.graph.node2type[uu] for uu in n_u]
        nk_v_type = [self.graph.node2type[vv] for vv in nk_v]

        return n_u, nk_v, n_u_type, nk_v_type


class GraphDataset(Dataset):
    def __init__(self, graph, k, negative_sampler):
        self.graph = graph
        self.k = k  # neg:pos ratio
        self.n_neg_edges = len(self.graph.neg_edges_u)
        self.negative_sampler = negative_sampler(self.graph, k)

    def __len__(self):
        return len(self.graph.edges_u)

    def __getitem__(self, idx):

        # pos item
        u = self.graph.edges_u[idx]  # user (type: string)
        v = self.graph.edges_v[idx]  # item (type: string)

        u_id = self.graph.node2idx[u]  # user (type: int)
        v_id = self.graph.node2idx[v] # item (type: string)

        u_type = self.graph.node2type[u]
        v_type = self.graph.node2type[v]

        u_type_id = self.graph.nodetype2idx[u_type]
        v_type_id = self.graph.nodetype2idx[v_type]

        # k corrupted neg edges
        n_u, nk_v, n_u_type, nk_v_type = self.negative_sampler.sample((u, v))

        n_u_ids = [self.graph.node2idx[x] for x in n_u]  # user (type: int)
        nk_v_ids = [self.graph.node2idx[x] for x in nk_v]  # item (type: string)

        n_u_type_ids = [self.graph.nodetype2idx[x] for x in n_u_type]
        nk_v_type_ids = [self.graph.nodetype2idx[x] for x in nk_v_type]

        return {
            "pos": (u_id, v_id, u_type_id, v_type_id),
            "neg": (n_u_ids, nk_v_ids, n_u_type_ids, nk_v_type_ids),
        }


