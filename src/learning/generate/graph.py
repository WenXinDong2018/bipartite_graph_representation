import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from scipy.sparse import coo_matrix


class Graph:
    """class for representing directed graphs"""

    """every node need to have unique name, a type, and optionally, a description"""

    def __init__(self):

        self.node2type = {}
        self.edges_u = []  # directed edges u->v
        self.edges_v = []
        self.neg_edges_u = []
        self.neg_edges_v = []
        self.gotten_neg_edges = False
        self.node_desc = {}  # node description
        self.adj_list = defaultdict(set)
        self.type2node = defaultdict(set)
        self.edgetype2edges = defaultdict(list)
        self.node2idx = {}  # assign a number to each node name
        self.nodetype2idx = {} # assign a number to each node type
        self.idx2node = {}
        self.idx2nodetype = {}
        self.params = {} #parameters in synthetic generation

    def get_n_nodes(self):
        return len(self.node2type)

    def get_node_types(self):
        return list(self.type2node.keys())

    def get_edges(self):
        return self.edges_u, self.edges_v

    def get_edge_types(self):
        return self.edgetype2edges.keys()

    def get_edges_by_type(self):
        return self.edgetype2edges

    def get_nodes(self):
        return list(self.node2type.keys())

    def get_nodes_by_type(self):
        return self.type2node

    def get_node_neg_neighbours(self, node):
        """assuming this is a bipartite graph with 2 types of node"""
        if not hasattr(self, "neg_neighbours"):
            self.get_negative_neighbours()
        return self.neg_neighbours[node]

    def get_node_neighbours(self, node):
        return self.adj_list[node]
    def get_nodes_id(self, nodes):
        return [self.node2idx[node] for node in nodes]
    def add_node_desc(self, u, desc):
        self.node_desc[u] = desc
    def get_node_desc(self, node):
        return self.node_desc[node]
    def add_edge(self, u, v, u_type, v_type):

        self.edges_u.append(u)
        self.edges_v.append(v)

        """assert that either nodes are new or the type is consistent"""
        assert u not in self.node2type or self.node2type[u] == u_type
        assert v not in self.node2type or self.node2type[v] == v_type

        self.node2type[u] = u_type
        self.node2type[v] = v_type

        if u not in self.node2idx:
            self.node2idx[u] = len(self.node2idx)
            self.idx2node[self.node2idx[u]] = u
        if v not in self.node2idx:
            self.node2idx[v] = len(self.node2idx)
            self.idx2node[self.node2idx[v]] = v
        if u_type not in self.nodetype2idx:
            self.nodetype2idx[u_type] = len(self.nodetype2idx)
            self.idx2nodetype[self.nodetype2idx[u_type]] = u_type
        if v_type not in self.nodetype2idx:
            self.nodetype2idx[v_type] = len(self.nodetype2idx)
            self.idx2nodetype[self.nodetype2idx[v_type]] = v_type

        self.type2node[u_type].add(u)
        self.type2node[v_type].add(v)

        self.adj_list[u].add(v)
        # self.adj_list[v].add(u)

        self.edgetype2edges[(u_type, v_type)].append((u, v))

    def add_edges(self, edges):
        for u, v, u_type, v_type in edges:
            self.add_edge(u, v, u_type, v_type)

    def add_negative_edges(self, edges):
        for u, v in edges:
            self.neg_edges_u.append(u)
            self.neg_edges_v.append(v)

    def shuffle(self):
        """shuffle edge list"""
        temp = list(zip(self.edges_u, self.edges_v))
        random.shuffle(temp)
        self.edges_u, self.edges_v = zip(*temp)
        """shuffle node list"""
        l = list(self.node2type.items())
        random.shuffle(l)
        self.node2type = dict(l)

    def inductive_split(self, test_u_types, ratios):
        # todo: check code correctness
        """imagine adding new nodes to the training graph during test time

        test_u_types: which type of "u" nodes to introduce during test time e.g. ["topic"]

        ratios: (0.8, 0.1, 0.1) train, val, test
        """
        self.shuffle()

        train_graph, val_graph, test_graph = Graph(), Graph(), Graph()

        u_only_train = [
            node
            for node in set(self.edges_u)
            if self.node2type[node] not in test_u_types
        ]
        u_split = [
            node for node in set(self.edges_u) if self.node2type[node] in test_u_types
        ]

        random.shuffle(u_split)
        n_train, n_val, n_test = np.array(np.array(ratios) * len(u_split), dtype=int)
        u_train = u_only_train + u_split[:n_train]
        u_val = u_split[n_train : n_train + n_val]
        u_test = u_split[n_train + n_val :]
        edges_train = [
            (u, v, self.node2type[u], self.node2type[v])
            for u, v in zip(self.edges_u, self.edges_v)
            if u in u_train
        ]
        edges_val = [
            (u, v, self.node2type[u], self.node2type[v])
            for u, v in zip(self.edges_u, self.edges_v)
            if u in u_val
        ]
        edges_test = [
            (u, v, self.node2type[u], self.node2type[v])
            for u, v in zip(self.edges_u, self.edges_v)
            if u in u_test
        ]

        train_graph.add_edges(edges_train)
        train_graph.get_negative_edges()
        val_graph.add_edges(edges_val + edges_train)
        val_graph.get_negative_edges()
        val_graph.delete_edges()
        val_graph.add_edges(edges_val)

        test_graph.add_edges(edges_test + edges_val + edges_train)
        test_graph.get_negative_edges()
        test_graph.delete_edges()
        test_graph.add_edges(edges_test)

        return train_graph, val_graph, test_graph

    def transductive_split(self, keep_edge_types, ratios):
        # todo: check code correctness
        """imagine adding edges onto training graph during testing
        keep_edge_types: [("professor", "class"), ("department","class")]
        ratios: (0.8, 0.1, 0.1) train, val, test

        """
        train_graph, val_graph, test_graph = Graph(), Graph(), Graph()
        train_graph.node2type = self.node2type
        val_graph.node2type = self.node2type
        test_graph.node2type = self.node2type

        """split negative edges"""
        self.get_negative_edges()
        negative_edges = [(u, v) for u, v in zip(self.neg_edges_u, self.neg_edges_v)]
        print(f"got {len(negative_edges)} negative edges")
        negative_edges_only_train, negative_edges_split = [], []
        for e in negative_edges:
            if (self.node2type[e[0]], self.node2type[e[1]]) in keep_edge_types:
                negative_edges_only_train.append(e)
            else:
                negative_edges_split.append(e)
        print(f"got {len(negative_edges_split)} negative edges for splitting")
        n_train, n_val, n_test = np.array(
            np.array(ratios) * len(negative_edges_split), dtype=int
        )
        train_split_idx, val_test_split_idx = train_test_split(
            range(len(negative_edges_split)), train_size=n_train
        )

        val_split_idx, test_split_idx = train_test_split(
            val_test_split_idx, train_size=n_val
        )

        train_graph.add_negative_edges(negative_edges_only_train)
        train_graph.add_negative_edges(
            [negative_edges_split[i] for i in train_split_idx]
        )
        val_graph.add_negative_edges([negative_edges_split[i] for i in val_split_idx])
        test_graph.add_negative_edges([negative_edges_split[i] for i in test_split_idx])

        """positive edges"""

        edges_only_train = [
            (u, v, self.node2type[u], self.node2type[v])
            for u, v in zip(self.edges_u, self.edges_v)
            if (self.node2type[u], self.node2type[v]) in keep_edge_types
        ]
        edges_split = [
            (u, v, self.node2type[u], self.node2type[v])
            for u, v in zip(self.edges_u, self.edges_v)
            if (self.node2type[u], self.node2type[v]) not in keep_edge_types
        ]
        #         random.shuffle(edges_split)
        n_train, n_val, n_test = np.array(
            np.array(ratios) * len(edges_split), dtype=int
        )

        train_graph.add_edges(edges_only_train)
        train_graph.add_edges(edges_split[:n_train])
        val_graph.add_edges(edges_split[n_train : n_train + n_val])
        test_graph.add_edges(edges_split[n_train + n_val :])

        return train_graph, val_graph, test_graph

    def get_negative_neighbours(self):

        self.neg_neighbours = {}
        all_nodes = set(self.node2idx)
        for node in self.node2idx:
            self.neg_neighbours[node] = list(all_nodes - set(self.adj_list[node]))

    def sample_random_negative_edges(self, k):

        adj_matrix = self.get_adj_matrix().tocsr()
        n, m = adj_matrix.shape

        total_n_neg = n * m - len(self.edges_u)

        k = min(k, total_n_neg)
        neg_edges_u, neg_edges_v = [], []
        while True:
            random_i = np.random.choice(n, k)
            random_j = np.random.choice(m, k)
            for i, j in zip(random_i, random_j):
                if adj_matrix[i, j] == False or adj_matrix[i, j] == 0:
                    neg_edges_u.append(self.idx2node[i])
                    neg_edges_v.append(self.idx2node[j])
                    if len(neg_edges_u) >= k:
                        return neg_edges_u, neg_edges_v

    def get_negative_edges(self, neg_ratio=1):
        """for small graphs use all entires in the neg adj matrix.
        for large graphs use sampling"""

        if self.gotten_neg_edges:
            return self.neg_edges_u, self.neg_edges_v
        self.gotten_neg_edges = True

        adj = self.get_adj_matrix().tocsr()
        N = adj.shape[0]

        if N * N < 1e8:
            for i in range(N):
                for j in range(N):
                    if adj[i, j] == False or adj[i, j] == 0:
                        self.neg_edges_u.append(self.idx2node[i])
                        self.neg_edges_v.append(self.idx2node[j])
        else:
            total_n_neg = N * N - len(self.edges_u)
            n_random = min(100 * len(self.edges_u), total_n_neg)
            while True:
                random_i = np.random.choice(N, n_random)
                random_j = np.random.choice(N, n_random)
                for i, j in zip(random_i, random_j):
                    if adj[i, j] == False or adj[i, j] == 0:
                        self.neg_edges_u.append(self.idx2node[i])
                        self.neg_edges_v.append(self.idx2node[j])
                        if len(self.neg_edges_u) >= min(
                            neg_ratio * len(self.edges_u), total_n_neg
                        ):
                            return self.neg_edges_u, self.neg_edges_v


    def get_adj_matrix(self):
        """sparse adj coo_matrix of size #nodes x #nodes"""
        if hasattr(self, "adj_matrix"):
            return self.adj_matrix
        N = len(self.node2idx)
        u_ids = [self.node2idx[u] for u in self.edges_u]
        v_ids = [self.node2idx[v] for v in self.edges_v]
        data = np.ones(len(u_ids))
        adj = coo_matrix((data, (u_ids, v_ids)), shape=(N, N), dtype=bool)
        self.adj_matrix = adj
        return adj


class BipartiteGraph(Graph):

    """class for representing directed bipartite graphs for collaborative filtering"""

    """edges are one direction: from users nodes  to item nodes """
    """edges_u only consists of user nodes, edges_v only consists of item nodes"""

    def __init__(self) -> None:
        super().__init__()
        self.is_bipartite = True
        self.u2id = {}
        self.v2id = {}

    def get_negative_neighbours(self):

        u_nodes = set(self.edges_u)
        v_nodes = set(self.edges_v)

        self.neg_neighbours = {}
        for a in u_nodes:
            self.neg_neighbours[a] = list(v_nodes - set(self.adj_list[a]))

        return self.neg_neighbours
    def sample_random_negative_edges(self, k):

        u_nodes = list(set(self.edges_u))
        v_nodes = list(set(self.edges_v))

        n = len(u_nodes)
        m = len(v_nodes)

        total_n_neg = n * m - len(self.edges_u)

        k = max(0, min(k, total_n_neg))
        neg_edges_u, neg_edges_v = [], []

        if k == 0:
            print("there are no negative edges")
            return neg_edges_u, neg_edges_v


        while True:
            random_i = np.random.choice(n, k)
            random_j = np.random.choice(m, k)
            for i, j in zip(random_i, random_j):
                if v_nodes[j] not in self.adj_list[u_nodes[i]]:
                    neg_edges_u.append(u_nodes[i])
                    neg_edges_v.append(v_nodes[j])
                    if len(neg_edges_u) >= k:
                        return neg_edges_u, neg_edges_v

    def get_negative_edges(self):


        if self.gotten_neg_edges:
            return self.neg_edges_u, self.neg_edges_v

        u_nodes = set(self.edges_u)
        for u in u_nodes:
            neg_v = self.get_node_neg_neighbours(u)
            self.neg_edges_u.extend([u for _ in neg_v])
            self.neg_edges_v.extend([v for v in neg_v])

        self.gotten_neg_edges = True
        return self.neg_edges_u, self.neg_edges_v

    def get_interaction_matrix(self):
        """interaction matrix is NOT adj matrix"""
        """size: n x m where n is number of users and m is number of items"""
        if hasattr(self, "interaction_matrix"):
            return self.interaction_matrix

        unique_u = list(set(self.edges_u))  # B nodes (users)
        unique_v = list(set(self.edges_v))  # A nodes (items)
        n = len(unique_u)
        m = len(unique_v)

        self.u2id = {u: i for i, u in enumerate(sorted(unique_u))}
        self.v2id = {v: i for i, v in enumerate(sorted(unique_v))}
        u_ids = [self.u2id[u] for u in self.edges_u]
        v_ids = [self.v2id[v] for v in self.edges_v]
        data = np.ones(len(u_ids))
        matrix = coo_matrix((data, (u_ids, v_ids)), shape=(n, m), dtype=bool)
        self.interaction_matrix = matrix
        return matrix

    def user_based_transductive_split(self,ratios, keep_u_types=[]):

        """
        imagine a user liked 100 items. use 80 items
        for training, 10 for validation, and 10 for testing
        """
        train_graph, val_graph, test_graph = (
            BipartiteGraph(),
            BipartiteGraph(),
            BipartiteGraph(),
        )
        train_graph.node2type = self.node2type
        val_graph.node2type = self.node2type
        test_graph.node2type = self.node2type

        """split negative edges"""
        self.get_negative_edges()
        negative_edges = [(u, v) for u, v in zip(self.neg_edges_u, self.neg_edges_v)]
        print(f"got {len(negative_edges)} negative edges")

        negative_edges_only_train, negative_edges_split = [], []
        for e in negative_edges:
            if self.node2type[e[0]] in keep_u_types:
                negative_edges_only_train.append(e)
            else:
                negative_edges_split.append(e)
        print(f"got {len(negative_edges_split)} negative edges for splitting")

        n_train, n_val, n_test = np.array(
            np.array(ratios) * len(negative_edges_split), dtype=int
        )
        train_split_idx, val_test_split_idx = train_test_split(
            range(len(negative_edges_split)), train_size=n_train
        )

        val_split_idx, test_split_idx = train_test_split(
            val_test_split_idx, train_size=n_val
        )

        train_graph.add_negative_edges(negative_edges_only_train)
        train_graph.add_negative_edges(
            [negative_edges_split[i] for i in train_split_idx]
        )
        val_graph.add_negative_edges([negative_edges_split[i] for i in val_split_idx])
        test_graph.add_negative_edges([negative_edges_split[i] for i in test_split_idx])

        """positive edges"""
        u_nodes = list(set(self.edges_u))
        for node in u_nodes:

            liked_items = list(self.adj_list[node])
            random.shuffle(liked_items)

            if self.node2type[node] in keep_u_types:

                train_graph.add_edges(
                    [
                        (node, v, self.node2type[node], self.node2type[v])
                        for v in liked_items
                    ]
                )

            else:

                n_train, n_val, n_test = np.array(
                    np.array(ratios) * len(liked_items), dtype=int
                )
                train_items = liked_items[:n_train]
                val_items = liked_items[n_train : n_train + n_val]
                test_items = liked_items[n_train + n_val :]
                train_graph.add_edges(
                    [
                        (node, v, self.node2type[node], self.node2type[v])
                        for v in train_items
                    ]
                )
                val_graph.add_edges(
                    [
                        (node, v, self.node2type[node], self.node2type[v])
                        for v in val_items
                    ]
                )
                test_graph.add_edges(
                    [
                        (node, v, self.node2type[node], self.node2type[v])
                        for v in test_items
                    ]
                )

        return train_graph, val_graph, test_graph
