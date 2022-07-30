import numpy as np

# BIPARTITE_DEFAULT_PARAMS = {
#     "n_vertices_A": 100,
#     "n_vertices_B": 100,
#     "exp_deg_B": 10,
#     "exp_deg_B_range": 1,
#     "n_clusters": 5,
#     "cluster_slope": 1,
#     "pq_ratio": 0.8,
# }

BIPARTITE_DEFAULT_PARAMS = {
    "n_vertices_A": 10,
    "n_vertices_B": 10,
    "exp_deg_B": 1,
    "exp_deg_B_range": 1,
    "n_clusters": 1,
    "cluster_slope": 1,
    "pq_ratio": 0.8,
}

BIPARTITE_DEFAULT_RANGE = {
    "n_vertices_A": np.arange(100, 2000, 400),
    "n_vertices_B": np.arange(100, 2000, 400),
    "exp_deg_B": np.arange(5, 100, 10),
    "exp_deg_B_range": np.arange(1, 20, 2),
    "n_clusters": np.arange(1, 100),
    "cluster_slope": np.arange(1, 10, 1),
    "pq_ratio": np.arange(0.5, 1, 0.1),
}
