import numpy as np
from src.learning.generate.generate_synthetic_graph_params import (
    BIPARTITE_DEFAULT_PARAMS,
)
from src.learning.generate.graph import BipartiteGraph
from src.learning.generate.graph_utils import NpEncoder
import os
import time
from scipy.sparse import save_npz
import json
import pickle


def generateSyntheticBipartiteGraph(params, save_graph=True, save_dir=None):

    """generate a BipartiteGraph object and a corresponding sparse adjacency matrix"""
    """naming one set of nodes "A" nodes (items) and the other set of nodes "B" nodes (users)"""
    """the directed bipartite graph consists of edges from B nodes to A nodes"""
    """
        usage:
            params = GraphParams() #default bipartite setting
            params.n_vertices_A = 100 #override default setting
            generateSyntheticBipartiteGraph(params)

    """

    """step 1: retrieve graph parameters"""
    n_vertices_A = params.n_vertices_A
    n_vertices_B = params.n_vertices_B

    exp_deg_B = params.exp_deg_B
    exp_deg_B_range = params.exp_deg_B_range

    n_clusters = params.n_clusters
    cluster_slope = params.cluster_slope
    pq_ratio = params.pq_ratio

    """step 2: cluster the B nodes"""
    smallest_cluster = n_vertices_A / (
        np.sum(np.arange(1, n_clusters) * cluster_slope) + 1
    )
    cluster_sizes = [smallest_cluster] + list(
        cluster_slope * smallest_cluster * np.arange(1, n_clusters)
    )
    cluster_sizes = np.array(cluster_sizes, dtype=int)
    cluster_sizes[-1] += n_vertices_A - np.sum(cluster_sizes)

    clusters = []
    currIdx = 0
    for i in cluster_sizes:
        clusters.append(np.arange(currIdx, currIdx + i))
        currIdx += i

    """step 3: assign cluster to each B node"""
    cluster_assignment = sorted(np.random.randint(n_clusters, size=(n_vertices_B)))

    """step 4: assign node deg to each B node"""
    B_deg = np.random.randint(
        max(0, exp_deg_B - exp_deg_B_range),
        min(exp_deg_B + exp_deg_B_range, n_vertices_A),
        size=(n_vertices_B),
    )

    """step 5: assign in-out deg"""
    B_in_cluster = np.array(B_deg * pq_ratio, dtype=int)
    B_out_cluster = B_deg - B_in_cluster

    """step 6: create edges"""
    g = BipartiteGraph()
    g.params = params.getDict()
    for b in range(n_vertices_B):
        cluster = clusters[cluster_assignment[b]]
        in_edges = np.random.choice(
            cluster, size=(min(len(cluster), B_in_cluster[b])), replace=False
        )
        for a in in_edges:
            g.add_edge(f"b{str(b).zfill(5)}", f"a{str(a).zfill(5)}", "b", "a")
        out_edges = np.random.choice(
            np.arange(n_vertices_A), size=(B_out_cluster[b]), replace=False
        )
        for a in out_edges:
            g.add_edge(f"b{str(b).zfill(5)}", f"a{str(a).zfill(5)}", "b", "a")

    """for each node, get it's non-neighbours"""
    # g.get_negative_neighbours()

    """get adj matrix"""
    sparse_adj_matrix = g.get_adj_matrix()
    sparse_interaction_matrix = g.get_interaction_matrix()

    if save_graph:
        if not save_dir:
            save_dir = "bipartite_graphs"

        seed = str(time.time_ns())
        folder_name = os.path.join(
            "synthetic_graphs", save_dir, params.toString(), seed
        )
        os.makedirs(folder_name)
        g_path = os.path.join(folder_name, seed + ".pickle")
        adj_matrix_path = os.path.join(folder_name, seed + "adj.npz")
        # interaction_matrix_path = os.path.join(folder_name, seed, seed+"interaction.npz")
        meta_path = os.path.join(folder_name, seed + ".json")

        pickle.dump(g, open(g_path, "wb"))
        json.dump(params.getDict(), open(meta_path, "w"), cls=NpEncoder)
        save_npz(adj_matrix_path, sparse_adj_matrix)
        # save_npz(interaction_matrix_path, sparse_interaction_matrix)

        return g, sparse_adj_matrix, sparse_interaction_matrix, folder_name
    return g, sparse_adj_matrix, sparse_interaction_matrix, None


class GraphParams:
    def __init__(self, params=BIPARTITE_DEFAULT_PARAMS):
        self.params = params
        for k, v in params.items():
            setattr(self, k, v)

    def getDict(self):
        return self.params

    def toString(self):
        s = "-".join([f"{key}={value}" for key, value in self.params.items()])
        return s


def get_graph_metric(data_path):
    for file in os.listdir(data_path):
        if file.endswith(".json"):
            graph_metrics = json.load(open(os.path.join(data_path, file), "r"))
            return graph_metrics
