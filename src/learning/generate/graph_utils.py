import numpy as np
from scipy.sparse import coo_matrix
import pickle
import os
from scipy.sparse import load_npz


def get_target_adj_matrix(data_path):
    for file in os.listdir(data_path):
        if file.endswith("adj.npz"):
            return load_npz(os.path.join(data_path, file))


def get_target_interaction_matrix(data_path):
    target_g = get_target_graph(data_path)
    return target_g.get_interaction_matrix()


def get_target_graph(data_path):
    for file in os.listdir(data_path):
        if file.endswith(".pickle"):
            return pickle.load(open(os.path.join(data_path, file), "rb"))


def convert_adj_to_interaction(adj_matrix, data_path):
    """convert the adjacency matrix of a graph to the corresponding interaction matrix"""
    """only works with bipartite graphs"""

    target_g = get_target_graph(data_path)
    assert target_g.is_bipartite == True

    unique_u = list(set(target_g.edges_u))
    unique_v = list(set(target_g.edges_v))
    n = len(unique_u)
    m = len(unique_v)

    u2id = {u: i for i, u in enumerate(sorted(unique_u))}
    v2id = {v: i for i, v in enumerate(sorted(unique_v))}

    u_ids = []
    v_ids = []

    for u_id, v_id in zip(adj_matrix.nonzero()[0], adj_matrix.nonzero()[1]):

        u = target_g.idx2node[u_id]
        v = target_g.idx2node[v_id]
        # if this is a valid edge (from B nodes to A nodes), add to the interaction matrix
        if u in u2id and v in v2id:
            u_ids.append(u2id[u])
            v_ids.append(v2id[v])

    data = np.ones(len(u_ids))
    interaction_matrix = coo_matrix((data, (u_ids, v_ids)), shape=(n, m), dtype=bool)
    return interaction_matrix
