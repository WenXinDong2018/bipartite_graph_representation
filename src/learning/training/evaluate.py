from src.learning.training.trainer import get_metrics
from sklearn.metrics import ndcg_score
import torch
from src.learning.generate.graph import BipartiteGraph
from scipy.sparse import coo_matrix
import numpy as np
import sklearn
from loguru import logger


def evaluate(model, eval_graph):
    """Evaluate model after completing training

    Args:
        model (BipartiteModel): model
        eval_graph (BipartiteGraph): target graph

    Returns:
        (dict, sparse coo): metrics, reconstructed interaction matrix
    """
    # evaluate on all edges (positive and negative)
    pos_edges_u, pos_edges_v = eval_graph.get_edges()
    neg_edges_u, neg_edges_v = eval_graph.get_negative_edges()
    ground_truth = torch.concat([torch.ones(len(pos_edges_u)), torch.zeros(len(neg_edges_u))])
    edges_u = pos_edges_u + neg_edges_u
    edges_v = pos_edges_v + neg_edges_v

    batch_size = 128
    edge_pred_prob = []
    for s in range(0, len(edges_u), batch_size):
        batch = {"pos":[[],[]], "neg": [[], []]}
        batch["pos"][0] = torch.tensor([eval_graph.node2idx[u] for u in edges_u[s:s+batch_size]])
        batch["pos"][1] = torch.tensor([eval_graph.node2idx[v] for v in edges_v[s:s+batch_size]])
        batch["neg"][0] = [batch["pos"][0]] #hack. need to have neg samples for loss calculate
        batch["neg"][1] = [batch["pos"][1]]
        _, edge_prob, _ = model(batch)
        edge_pred_prob.append(edge_prob[:len(batch["pos"][0])])

    edge_prob = torch.concat(edge_pred_prob).detach().cpu()
    metrics = get_metrics(edge_prob, ground_truth)
    prediction_coo = get_prediction_coo(eval_graph, edges_u, edges_v, edge_prob, ground_truth)
    return metrics, prediction_coo


def get_metrics(edge_prob, ground_truth):

    threshold = calculate_optimal_threshold(ground_truth, edge_prob)
    pred = edge_prob > threshold
    accuracy = sklearn.metrics.accuracy_score(ground_truth, pred)
    precision = sklearn.metrics.precision_score(ground_truth, pred)
    recall = sklearn.metrics.recall_score(ground_truth, pred)
    F1 = sklearn.metrics.f1_score(ground_truth, pred)
    auc = sklearn.metrics.roc_auc_score(ground_truth, pred)

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": F1,
        "AUC": auc,
    }


def calculate_optimal_threshold(targets, scores):
    """Calculate the optimal threshold for F1 score

    Args:
        targets (tensor(int)): binary labels
        scores (tensor(float)): probabilities between [0, 1]

    Returns:
        float: optimal threshold for F1 score
    """
    targets = targets.numpy()
    scores = scores.numpy()
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        targets, scores, drop_intermediate=False
    )
    num_pos = targets.sum()
    num_neg = (1 - targets).sum()
    f1 = 2 * tpr / (1 + tpr + fpr * num_neg / num_pos)
    threshold = thresholds[np.argmax(f1)]

    return threshold


def get_prediction_coo(eval_g, edges_u, edges_v, edge_prob, ground_truth):
    """Generate sparse coo matrix representing the reconstructed graph

    Args:
        eval_g (BipartiteGraph): the target graph that the model was trained to reconstruct
        edges_u (list(string)): all edges in the graph u->v, the u nodes
        edges_v (list(string)): all edges in the graph u->v, the v nodes
        edge_prob (tensor(float)): the predicted presence probability of each edge
        ground_truth (tensor(int)): 0 or 1 for whether the edge is present in the eval graph

    Returns:
        sparse coo: the reconstructed interaction matrix
    """
    optimal_threshold = calculate_optimal_threshold(ground_truth, edge_prob)
    hard_pred = edge_prob > optimal_threshold
    logger.info("total #edges:", len(hard_pred))
    logger.info("predicted #pos edges:", sum(hard_pred))
    logger.info("hard_pred", hard_pred[0])

    edge_ids = torch.where(hard_pred)[0]
    pred_pos_u = [eval_g.u2id[edges_u[i]] for i in edge_ids]
    pred_neg_u = [eval_g.v2id[edges_v[i]] for i in edge_ids]
    data = np.ones(sum(hard_pred))
    n,m = len(eval_g.u2id), len(eval_g.v2id)
    matrix = coo_matrix((data, (pred_pos_u, pred_neg_u)), shape=(n, m), dtype=bool)

    return matrix
