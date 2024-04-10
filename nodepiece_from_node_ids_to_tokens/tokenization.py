"""A module with the main functions performing the NddePiece tokenization.
"""

import networkx as nx
import torch as th
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils
from typing import List, Tuple, Dict
import models
from tqdm.auto import tqdm
import numpy as np


def degree_anchor_select(g: nx.Graph, n_anchors: int|float = 0.1) -> Tuple[List[int], Dict[int, int]]:
    """Anchor selection method, based on the node degree. It is based on the simplest heuristic,
    where the nodes with the highest degree are selected as anchors - as they will most likely be connected
    to the most nodes in the graph.

    Parameters
    ----------
    g : nx.Graph
        Networkx graph.
    n_anchors : int | float, optional
        Number of anchors to select, by default 0.1.
        If int - the number of anchors to select.
        If float - the fraction of the nodes to select as anchors.

    Returns
    -------
    Tuple[List[int], Dict[int, int]]
        1. List of anchor nodes.
        2. Dictionary mapping anchor node to its id. Anchor ids are in the range [0, n_anchors).
    """
    if type(n_anchors) == float:
        n_anchors = int(g.number_of_nodes() * n_anchors)

    degrees = sorted(g.degree, key=lambda x: x[1], reverse=True)
    anchor_2_id = {}
    anchors = []
    for i, (node, _) in enumerate(degrees[:n_anchors]):
        anchors.append(node)
        anchor_2_id[node] = i

    return anchors, anchor_2_id


def build_distance_to_k_nearest_anchors(
        G: nx.Graph,
        anchors: List[int],
        anchor2id: dict,
        k_closest_anchors: int = 15,
        use_closest: bool = True) -> Tuple[np.ndarray, np.ndarray, int]:
    """For each node in the graph, calculate the distance to the k closest anchors.

    Parameters
    ----------
    G : nx.Graph
        Netowrkx graph.
    anchors : List[int]
        List of anchor nodes.
    anchor2id : dict
        Anchor to id mapping.
    k_closest_anchors : int, optional
        Number of k closest anchors to pick per node, by default 15
    use_closest : bool, optional
        Should closest anchors be used, or all? By default True

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int]
        Tuple of:
        1. Node to anchor distance matrix. Shape: (num_nodes, num_anchors).
        2. Node to anchor id matrix. Shape: (num_nodes, num_anchors).
        3. Maximum distance in the graph. Will be used for distance encoding/embedding.
    """
    node_distances = {i: [] for i in range(G.number_of_nodes())}
    for a in tqdm(anchors):
        for node, dist in nx.shortest_path_length(G, source=a).items():
            node_distances[node].append((a, dist))

    node2anchor_dist = np.zeros((G.number_of_nodes(), len(anchors)))
    node2anchor_idx = np.zeros((G.number_of_nodes(), len(anchors)))
    unreachable_anchor_token = len(anchors)
    node2anchor_idx.fill(unreachable_anchor_token)

    max_dist = 0

    for node, distances in tqdm(node_distances.items()):
        indices_of_anchors = sorted(distances, key=lambda x: x[1])[:k_closest_anchors] if use_closest else node_distances[node]
        for i, (anchor, dist) in enumerate(indices_of_anchors):
            anchor_id = anchor2id[anchor]
            node2anchor_dist[node, anchor_id] = dist

            node2anchor_idx[node, i] = anchor_id
            if dist > max_dist:
                max_dist = dist
    unreachable_anchor_indices = node2anchor_idx == unreachable_anchor_token
    node2anchor_dist[unreachable_anchor_indices] = max_dist + 1
    return node2anchor_dist, node2anchor_idx, max_dist


def sample_rels(pyg_g: pyg_data.Data, max_rels: int = 50) -> th.Tensor:
    """Samples m outgoing relations for each node. If the node has less than m relations, it pads the output with a special token.

    Parameters
    ----------
    pyg_g : pyg_data.Data
        PyTorch Geometric graph.
    max_rels : int, optional
        Maximal number of relations to use, by default 50.

    Returns
    -------
    th.Tensor
        Matrix of relations for each node. Shape: (num_nodes, max_rels).
        Each row corresponds to specific node, each column to a relation (id).
    """
    rels_matrix = []
    missing_rel_token = pyg_g.edge_type.max() + 1
    for node in tqdm(range(pyg_g.num_nodes)):
        node_edges = pyg_g.edge_index[0] == node
        node_edge_types = pyg_g.edge_type[node_edges].unique()
        num_edge_types = len(node_edge_types)

        if num_edge_types < max_rels:
            pad = th.ones(max_rels - num_edge_types, dtype=th.long) * missing_rel_token
            padded_edge_types = th.cat([node_edge_types, pad])
            padded_edge_types = padded_edge_types.sort()[0]
        else:
            sampled_edge_types = th.randperm(num_edge_types)[:max_rels]
            padded_edge_types = node_edge_types[sampled_edge_types].sort()[0]
        rels_matrix.append(padded_edge_types)
    return th.stack(rels_matrix)


def tokenize_graph(
        pyg_graph: pyg_data.Data,
        n_anchors: int|float = 0.1,
        k_nearest_anchors: int = 15,
        use_closest: bool = True,
        m_relations: int = 50) -> models.NodePieceTokens:
    """Performs a simplified NodePiece tokenization of the graph.
    1. Selects anchors based on the node degree.
    2. Calculates the distance to the k closest anchors for each node.
    3. Samples m relations for each node.
    4. Returns the NodePieceTokens object containing all this data.

    Parameters
    ----------
    pyg_graph : pyg_data.Data
        PyTorch Geometric graph.
    n_anchors : int | float, optional
        Number of anchors to select, by default 0.1.
        If int - the number of anchors to select.
        If float - the fraction of the nodes to select as anchors.
    k_nearest_anchors : int, optional
        How many closest anchors to use per node, by default 15
    use_closest : bool, optional
        Should only closest anchors be used? By default True
    m_relations : int, optional
        M unique outgoing relations to sample per node, by default 50

    Returns
    -------
    models.NodePieceTokens
        An object summarizingthe tokenization process.
    """
    G = pyg_utils.to_networkx(pyg_graph)
    G = nx.to_undirected(G)
    anchors, anchors_2_id = degree_anchor_select(G, n_anchors)
    id_2_anchor = {v: k for k, v in anchors_2_id.items()}
    k_nearest_anchors = len(anchors) if not use_closest else k_nearest_anchors
    distances, anchors_per_node, max_dist = build_distance_to_k_nearest_anchors(G, anchors, anchors_2_id, k_nearest_anchors, use_closest)
    rel_hasehes = sample_rels(pyg_graph, m_relations)
    return models.NodePieceTokens(
        th.tensor(anchors_per_node).to(th.long),
        th.tensor(distances).to(th.long),
        th.tensor(rel_hasehes).to(th.long),
        len(anchors),
        pyg_graph.edge_type.max() + 1,
        max_dist,
        k_nearest_anchors,
        m_relations,
        anchors_2_id,
        id_2_anchor
    )
