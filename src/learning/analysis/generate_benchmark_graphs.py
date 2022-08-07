from src.learning.generate.graph import BipartiteGraph
from src.learning.generate.graph_utils import (
    get_target_graph,
    save_graph,
    get_target_adj_matrix,
    get_target_interaction_matrix,
)
import pickle


def generate_benchmark_graph(file_path, dataset_name):
    g = BipartiteGraph()
    with open(file_path, "r") as f:
        line = f.readline().strip()
        while line:
            line = line.split(" ")
            userid = line[0]
            for itemid in line[1:]:
                if "item" + itemid not in g.get_node_neighbours("user" + userid):
                    g.add_edge("user" + userid, "item" + itemid, "user", "item")
            line = f.readline().strip()
    g, sparse_adj_matrix, sparse_interaction_matrix, folder_name = save_graph(
        g, dataset_name
    )
    return g, sparse_adj_matrix, sparse_interaction_matrix, folder_name


def generate_course_benchmark_graph():
    (topic_to_class_u, topic_to_class_v) = pickle.load(
        open("cache/topic_to_class_all.pickle", "rb")
    )
    g = BipartiteGraph()
    """topic <-> class"""
    for topic, course in zip(topic_to_class_u, topic_to_class_v):
        g.add_edge("topic-"+topic, "course-"+course, "topic", "class")
    g, sparse_adj_matrix, sparse_interaction_matrix, folder_name = save_graph(
        g, "course_dataset"
    )
    return g, sparse_adj_matrix, sparse_interaction_matrix, folder_name

def load_benchmark(file_path):
    g = get_target_graph(file_path)
    sparse_adj_matrix = get_target_adj_matrix(file_path)
    sparse_interaction_matrix = get_target_interaction_matrix(file_path)

    return g, sparse_adj_matrix, sparse_interaction_matrix, file_path
