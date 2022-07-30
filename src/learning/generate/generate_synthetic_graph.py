import numpy as np
from src.learning.generate.generate_synthetic_graph_params import BIPARTITE_DEFAULT_PARAMS
from src.learning.generate.graph import BipartiteGraph

import os
import time
from scipy.sparse import save_npz
import json
import pickle

def generateSyntheticBipartiteGraph(params, save_graph = True):

    '''generate a BipartiteGraph object and a corresponding sparse adjacency matrix'''
    '''naming one set of nodes "A" nodes (items) and the other set of nodes "B" nodes (users)'''
    '''the directed bipartite graph consists of edges from B nodes to A nodes'''
    '''
        usage:
            params = GraphParams() #default bipartite setting
            params.n_vertices_A = 100 #override default setting
            generateSyntheticBipartiteGraph(params)

    '''

    '''step 1: retrieve graph parameters'''
    n_vertices_A = params.n_vertices_A
    n_vertices_B = params.n_vertices_B

    exp_deg_B = params.exp_deg_B
    exp_deg_B_range = params.exp_deg_B_range

    n_clusters = params.n_clusters
    cluster_slope = params.cluster_slope
    pq_ratio = params.pq_ratio

    '''step 2: cluster the B nodes'''
    smallest_cluster = n_vertices_A / (np.sum(np.arange(1, n_clusters) * cluster_slope)+1)
    cluster_sizes = [smallest_cluster]  + list(cluster_slope* smallest_cluster*np.arange(1, n_clusters))
    cluster_sizes = np.array(cluster_sizes, dtype = int)
    cluster_sizes[-1] += n_vertices_A - np.sum(cluster_sizes)

    clusters = []
    currIdx = 0
    for i in cluster_sizes:
        clusters.append(np.arange(currIdx, currIdx+i))
        currIdx += i

    '''step 3: assign cluster to each B node'''
    cluster_assignment = sorted(np.random.randint(n_clusters, size = (n_vertices_B)))

    '''step 4: assign node deg to each B node'''
    B_deg = np.random.randint(max(0, exp_deg_B - exp_deg_B_range), min(exp_deg_B + exp_deg_B_range, n_vertices_A), size = (n_vertices_B))

    '''step 5: assign in-out deg'''
    B_in_cluster = np.array(B_deg* pq_ratio,dtype = int)
    B_out_cluster = B_deg - B_in_cluster

    '''step 6: create edges'''
    g = BipartiteGraph()
    g.params = params.getDict()
    for b in range(n_vertices_B):
        cluster = clusters[cluster_assignment[b]]
        in_edges = np.random.choice(cluster, size = (min(len(cluster), B_in_cluster[b])), replace = False)
        for a in in_edges:
            g.add_edge(f"b{str(b).zfill(5)}",f"a{str(a).zfill(5)}", "b", "a")
        out_edges = np.random.choice(np.arange(n_vertices_A), size = (B_out_cluster[b]), replace = False)
        for a in out_edges:
            g.add_edge(f"b{str(b).zfill(5)}",f"a{str(a).zfill(5)}", "b", "a")

    '''for each node, get it's non-neighbours'''
    # g.get_negative_neighbours()

    '''get adj matrix'''
    sparse_adj_matrix = g.get_adj_matrix()
    sparse_interaction_matrix = g.get_interaction_matrix()

    if save_graph:
        seed = str(time.time_ns())
        folder_name = os.path.join("synthetic_graphs", "bipartite_graphs", f"n_vertices_A={n_vertices_A}-n_vertices_B={n_vertices_B}-exp_deg_B={exp_deg_B}-exp_deg_B_range={exp_deg_B_range}-n_clusters={n_clusters}-cluster_slope={cluster_slope}-pq_ratio={pq_ratio}", seed)
        os.makedirs(folder_name)
        g_path = os.path.join(folder_name, seed+".pickle")
        adj_matrix_path = os.path.join(folder_name, seed+"adj.npz")
        # interaction_matrix_path = os.path.join(folder_name, seed, seed+"interaction.npz")
        meta_path = os.path.join(folder_name, seed+".json")

        pickle.dump(g, open(g_path, "wb"))
        json.dump(params.getDict(), open(meta_path, "w"), cls=NpEncoder)
        save_npz(adj_matrix_path, sparse_adj_matrix)
        # save_npz(interaction_matrix_path, sparse_interaction_matrix)

        return g, sparse_adj_matrix,sparse_interaction_matrix, folder_name
    return g, sparse_adj_matrix, sparse_interaction_matrix, None

class GraphParams():
    def __init__(self, params=BIPARTITE_DEFAULT_PARAMS):
        self.params = params
        for k, v in params.items():
            setattr(self, k, v)
    def getDict(self):
        return self.params

# def getParams(graph):
#     '''TODO'''
#     params = GraphParams()
#     params.graph_matrix = graph.getMatrix()
#     params.n_vertices_A = graph.get_nodes()["A"]
#     params.n_vertices_B = graph.get_nodes()["B"]
#     params.exp_deg_A = np.mean(graph_matrix.sum(axis=0))
#     params.exp_deg_B = np.mean(graph_matrix.sum(axis=1))
#     params.n_edges = np.sum(graph_matrix)
#     params.n_domains_A = len(np.unique(graph_matrix.rows()))
#     params.n_domains_B = len(np.unique(graph_matrix.cols()))

#     return params

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
