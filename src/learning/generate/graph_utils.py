import numpy as np
from scipy.sparse import coo_matrix
import pickle
import os
from scipy.sparse import load_npz, save_npz
import json
import time

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

def save_graph(g, save_dir="temp"):

    folder_name = os.path.join(
        "saved_graphs", save_dir
    )
    if save_dir == "temp":
        seed = str(time.time_ns())
        save_dir = os.path.join("temp", seed)
    os.makedirs(folder_name)
    g_path = os.path.join(folder_name, "g.pickle")
    adj_matrix_path = os.path.join(folder_name, "adj.npz")

    sparse_adj_matrix = g.get_adj_matrix()
    sparse_interaction_matrix = g.get_interaction_matrix()

    pickle.dump(g, open(g_path, "wb"))
    save_npz(adj_matrix_path, sparse_adj_matrix)
    return g, sparse_adj_matrix, sparse_interaction_matrix, folder_name

def get_n_vertices_B(g):
    unique_u = list(set(g.edges_u))
    n = len(unique_u)
    return n
    
def get_n_vertices_A(g):
    unique_v = list(set(g.edges_v))
    m = len(unique_v)
    return m

def get_exp_deg_B(g):
    unique_u = list(set(g.edges_u))
    exp_deg_B = np.mean([len(g.get_node_neighbours(u)) for u in unique_u])
    return exp_deg_B

def get_exp_deg_B_range(g):
    unique_u = list(set(g.edges_u))
    max_deg_b = np.max([len(g.get_node_neighbours(u)) for u in unique_u])
    min_deg_b = np.min([len(g.get_node_neighbours(u)) for u in unique_u])
    return max_deg_b - min_deg_b
# def get_n_clusters(g):

# def get_cluster_slope(g):
# def get_pq_ratio(g):


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

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
