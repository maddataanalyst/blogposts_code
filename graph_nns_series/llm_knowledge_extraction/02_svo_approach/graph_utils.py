import torch as th
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class TripletMapping:
    elem2idx: dict
    rel2idx: dict

    idx2elem: dict
    idx2rel: dict

    subject_ids: List[int]
    rel_ids: List[int]
    object_ids: List[int]
    
    @property
    def subjects_and_objects(self):
        return list(self.elem2idx.keys())

    @property
    def relations(self):
        return list(self.rel2idx.keys())
                    
                

def triplets2idx(ents_triplets) -> TripletMapping:
    elem2idx = defaultdict(int)
    rel2idx = defaultdict(int)

    obj_id = -1
    rel_id = -1

    subject_ids = []
    rel_ids = []
    object_ids = []


    for h,r,t in ents_triplets:
        if h not in elem2idx:
            obj_id += 1
            elem2idx[h] = obj_id
        if r not in rel2idx:
            rel_id += 1
            rel2idx[r] = rel_id
        if t not in elem2idx:
            obj_id += 1
            elem2idx[t] = obj_id
        
        subject_ids.append(elem2idx[h])
        rel_ids.append(rel2idx[r])
        object_ids.append(elem2idx[t])

    idx2elem = {v:k for k,v in elem2idx.items()}
    idx2rel = {v:k for k,v in rel2idx.items()}

    return TripletMapping(elem2idx, rel2idx, idx2elem, idx2rel, subject_ids, rel_ids, object_ids)


def build_triplets_tensors(triplet_mapping: TripletMapping):
    subject_ids = th.tensor(triplet_mapping.subject_ids, dtype=th.long)
    rel_ids = th.tensor(triplet_mapping.rel_ids, dtype=th.long)
    object_ids = th.tensor(triplet_mapping.object_ids, dtype=th.long)

    return subject_ids, rel_ids, object_ids


def create_graph_viz(triplets):  
    G = nx.DiGraph()

    edges = {(s, o): r for s,r,o in triplets}
    G.add_edges_from(edges.keys())
    pos =  nx.nx_pydot.graphviz_layout(G)
    plt.figure(figsize=(10, 10))
    nx.draw(
        G, pos, edge_color='black', width=1, linewidths=1,
        node_size=500, node_color='pink', alpha=0.9,
        labels={node: node for node in G.nodes()}
    )
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edges,
        font_color='red'
    )
    plt.axis('off')
    plt.show()


def analyze_sentence_soundness(model, triplet_mapping: TripletMapping, subj: str=None, rel: str=None, object:str=None, topk: int = 5):
    """
    Analyzes the soundness of a given sentence by computing the scores of its subject, relation, and object triplets.
    User can provide an incomplete triplet (e.g. skip object or subject, etc.) - the function will compute the scores
    and find best-matching elements for the missing triplet element.

    Args:
        model (nn.Module): The PyTorch model used for scoring the triplets.
        triplet_mapping (TripletMapping): The mapping of triplets to their corresponding indices.
        subj (str, optional): The subject of the sentence. Defaults to None.
        rel (str, optional): The relation of the sentence. Defaults to None.
        object (str, optional): The object of the sentence. Defaults to None.
        topk (int, optional): The number of top-scoring triplets to return. Defaults to 5.

    Raises:
        ValueError: If both subj and object are None, or if both subj and rel, or object and rel are None.

    Returns:
        pd.DataFrame: A DataFrame containing the top-k scoring triplets and their scores.
    """
    
    # Check that at least one of the triplet elements is provided    
    if (subj is None) and (rel is None) and (object is None):
        raise ValueError("Subj+rel or object+rel must be provided")

    # If only one triplet element is provided, find the best-matching elements for the other two
    if (subj is not None) and (rel is not None):
        scores = {}
        with th.no_grad():
            for obj in triplet_mapping.subjects_and_objects:
                scores[obj] = score_triplet(subj, rel, obj, triplet_mapping, model)
    elif (object is not None) and (rel is not None):
        scores = {}
        with th.no_grad():
            for subj in triplet_mapping.subjects_and_objects:
                scores[subj] = score_triplet(subj, rel, object, triplet_mapping, model)
    elif (subj is not None) and (object is not None):
        scores = {}
        with th.no_grad():
            for rel in triplet_mapping.relations:
                scores[rel] = score_triplet(subj, rel, object, triplet_mapping, model)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
    sorted_scores = pd.DataFrame.from_records(sorted_scores, columns=['entity', 'score'])
    return sorted_scores


def score_triplet(s: str, r: str, o: str, triplet_mapping: TripletMapping, model) -> float:
    """
    Compute the score of a given triplet using a given model.
    
    Args:
    - s (str): the subject of the triplet
    - r (str): the relation of the triplet
    - o (str): the object of the triplet
    - triplet_mapping (TripletMapping): a mapping from triplets to their corresponding IDs
    - model: the model used to compute the score
    
    Returns:
    - score (float): the score of the triplet
    """
    sid = triplet_mapping.elem2idx[s]
    rid = triplet_mapping.rel2idx[r]
    oid = triplet_mapping.elem2idx[o]
    score = model.forward(th.tensor([sid]), th.tensor([rid]), th.tensor([oid])).item()
    return score