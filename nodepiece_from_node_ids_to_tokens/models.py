"""A module that contains embedding models for knowledge graphs."""

from typing import Tuple
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torcheval.metrics.functional as tmf
import numpy as np
import math
import pytorch_lightning as pl
from tqdm.auto import tqdm
from enum import Enum
from pydantic import BaseModel
from dataclasses import dataclass


@dataclass
class NodePieceTokens:

    anchor_hashes: th.Tensor
    anchor_distances: th.Tensor
    rel_hashes: th.Tensor

    n_anchors: int
    n_rels: int
    max_distance: int
    k_nearest_anchors: int
    m_relations: int
    anchor2id: dict
    id2anchor: dict


class NodePieceKGModel(nn.Module):
    """Generic class for a knowledge graph embedding model, compatible with the NodePiece tokenization"""

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        num_anchors: int,
        top_m_relations: int,
        max_distance: int,
        embedding_dim: int,
        hidden_sizes: Tuple[int, ...],
        margin: float = 1.0,
        p_norm: float = 1.0,
        sparse: bool = False,
        drop_prob: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.num_anchors = num_anchors
        self.max_distance = max_distance
        self.embedding_dim = embedding_dim
        self.hidden_sizes = hidden_sizes
        self.p_norm = p_norm
        self.margin = margin
        self.top_m_relations = top_m_relations
        self.embedding_dim = embedding_dim
        self.re_embed_dim = embedding_dim // 2
        self.drop_prob = drop_prob
        self.sparse = sparse
        self.device = device

    def embed_node(
        self,
        node: th.Tensor,
        closest_anchors: th.Tensor,
        anchor_distances: th.Tensor,
        rel_hash: th.Tensor,
    ) -> th.Tensor:
        """Emeds node (head or tail- it doesn't matter) using the closest anchors, anchor distances and relation hash.

        Parameters
        ----------
        node : th.Tensor
            Id of node of interest. Dim: N x 1
        closest_anchors : th.Tensor
            Ids of the closest anchors to the node. Dim: N x k
        anchor_distances : th.Tensor
            Distances from the node to the closest anchors. Dim: N x k
        rel_hash : th.Tensor
            Ids of the relation type. Dim: N x 1

        Returns
        -------
        th.Tensor
            Node embedding that can be treated as feature vector. Dim N x embedding_dim

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError("Implement me")

    @th.no_grad()
    def random_sample(
        self,
        head_index: th.Tensor,
        rel_type: th.Tensor,
        tail_index: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        r"""This implementation is based on the KGEModel from Pytorch Geometric (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.kge.KGEModel.html).
        It was slightly modified to work with the NodePiece tokenization.

        Randomly samples negative triplets by either replacing the head or
        the tail (but not both).

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        """
        # Random sample either `head_index` or `tail_index` (but not both):
        num_negatives = head_index.numel() // 2
        num_nodes = max(head_index.max(), tail_index.max())
        rnd_index = th.randint(num_nodes, head_index.size(), device=head_index.device)

        head_index = head_index.clone()
        head_index[:num_negatives] = rnd_index[:num_negatives]
        tail_index = tail_index.clone()
        tail_index[num_negatives:] = rnd_index[num_negatives:]

        return head_index, rel_type, tail_index

    def forward(
        self,
        head_index: th.Tensor,
        rel_type: th.Tensor,
        tail_index: th.Tensor,
        closest_anchors: th.Tensor,
        anchor_distances: th.Tensor,
        rel_hashes: th.Tensor,
    ) -> th.Tensor:
        """Main function for scoring the (head, relation, tail) triplets.

        Parameters
        ----------
        head_index : th.Tensor
            Ids of the head nodes. Dim: N x 1
        rel_type : th.Tensor
            Ids of the relation types. Dim: N x 1
        tail_index : th.Tensor
            Ids of the tail nodes. Dim: N x 1
        closest_anchors : th.Tensor
            Ids of the closest anchors to the head nodes. Dim: N x k
        anchor_distances : th.Tensor
            Distances from the head nodes to the closest anchors. Dim: N x k
        rel_hashes : th.Tensor
            Ids of the relation types. Dim: N x 1

        Returns
        -------
        th.Tensor
            Scores for the (head, relation, tail) triplets. Dim: N x 1

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError("Implement me")

    def loss(
        self,
        head_index: th.Tensor,
        rel_type: th.Tensor,
        tail_index: th.Tensor,
        closest_anchors: th.Tensor,
        anchor_distances: th.Tensor,
        rel_hashes: th.Tensor,
    ) -> th.Tensor:
        """Loss function for the model.

        Parameters
        ----------
        head_index : th.Tensor
            Ids of the head nodes. Dim: N x 1
        rel_type : th.Tensor
            Ids of the relation types. Dim: N x 1
        tail_index : th.Tensor
            Ids of the tail nodes. Dim: N x 1
        closest_anchors : th.Tensor
            Ids of the closest anchors to the head nodes. Dim: N x k
        anchor_distances : th.Tensor
            Distances from the head nodes to the closest anchors. Dim: N x k
        rel_hashes : th.Tensor
            Ids of the relation types. Dim: N x 1


        Returns
        -------
        th.Tensor
            Loss value

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError(("Implement me"))

    @th.no_grad()
    def test(
        self,
        head_index: th.Tensor,
        rel_type: th.Tensor,
        tail_index: th.Tensor,
        batch_size: int,
        closest_anchors: th.Tensor,
        anchor_distances: th.Tensor,
        rel_hashes: th.Tensor,
        k: int = 10,
        log: bool = True,
    ) -> Tuple[float, float]:
        """Calculates test scores for the model. Implementation based on the KGEModel from Pytorch Geometric (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.kge.KGEModel.html)
        Slightly modified to work with the NodePiece tokenization.

        Parameters
        ----------
                head_index : th.Tensor
            Ids of the head nodes. Dim: N x 1
        rel_type : th.Tensor
            Ids of the relation types. Dim: N x 1
        tail_index : th.Tensor
            Ids of the tail nodes. Dim: N x 1
        closest_anchors : th.Tensor
            Ids of the closest anchors to the head nodes. Dim: N x k
        anchor_distances : th.Tensor
            Distances from the head nodes to the closest anchors. Dim: N x k
        rel_hashes : th.Tensor
            Ids of the relation types. Dim: N x 1
        k : int, optional
            Top K elements to calculte hits@k, by default 10
        log : bool, optional
            Should results be logged using tqdm, by default True

        Returns
        -------
        Tuple[float, float]
            Mean rank and hits@k
        """
        hits = []
        ranks = []
        arange = range(head_index.numel())
        arange = tqdm(arange) if log else arange
        all_tail_indices = th.arange(self.num_nodes, device=tail_index.device)
        for i in arange:
            h, r, t = head_index[i], rel_type[i], tail_index[i]
            scores_i = []
            for ts in all_tail_indices.split(batch_size):
                hs = h.expand_as(ts)
                rs = r.expand_as(ts)

                score = self(hs, rs, ts, closest_anchors, anchor_distances, rel_hashes)
                scores_i.append(score)
            score_i = th.cat(scores_i)
            hits_at_10 = tmf.hit_rate(score_i.unsqueeze(0), t.unsqueeze(0), k=10)
            reciprocal_rank = tmf.reciprocal_rank(score_i.unsqueeze(0), t.unsqueeze(0))

            hits.append(hits_at_10.item())
            ranks.append(reciprocal_rank.item())
        hits_at_k = np.mean(hits)
        mean_rank = np.mean(ranks)

        return mean_rank, hits_at_k


class NodePieceRotatE(NodePieceKGModel):
    """Simplified RoratE implementaiton, following the original RotatE paper (https://arxiv.org/abs/1902.10197)."""

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        num_anchors: int,
        top_m_relations: int,
        max_distance: int,
        embedding_dim: int,
        hidden_sizes: Tuple[int, ...],
        margin: float = 1.0,
        p_norm: float = 1.0,
        sparse: bool = False,
        drop_prob: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__(
            num_nodes=num_nodes,
            num_relations=num_relations,
            num_anchors=num_anchors,
            top_m_relations=top_m_relations,
            max_distance=max_distance,
            embedding_dim=embedding_dim,
            hidden_sizes=hidden_sizes,
            margin=margin,
            p_norm=p_norm,
            sparse=sparse,
            drop_prob=drop_prob,
            device=device,
        )
        self.re_embed_dim = embedding_dim // 2

        self.anchor_embed = nn.Embedding(
            self.num_anchors + 1, embedding_dim, sparse=sparse
        )
        self.anchor_distances_embed = nn.Embedding(
            self.max_distance + 1, embedding_dim, sparse=sparse
        )

        layers = []

        last_in = (self.num_anchors + self.top_m_relations) * self.embedding_dim
        layers.append(pyg_nn.BatchNorm(last_in))
        for h_sz in hidden_sizes:
            layers.append(nn.Linear(last_in, h_sz))
            layers.append(nn.Dropout(self.drop_prob))
            layers.append(nn.LeakyReLU())
            last_in = h_sz
        layers.append(nn.Linear(last_in, self.embedding_dim))
        self.lin_layer = nn.Sequential(*layers)

        self.rel_emb = nn.Embedding(num_relations + 1, embedding_dim, sparse=sparse)

        self.reset_parameters()

    def embed_node(self, node: th.Tensor, closest_anchors, anchor_distances, rel_hash):

        anchor_embed = self.anchor_embed(closest_anchors[node])
        anchor_distances_embed = self.anchor_distances_embed(anchor_distances[node])
        rel_embed = self.rel_emb(rel_hash[node])
        combined_anchor_embed = anchor_embed + anchor_distances_embed

        # N x (anchors + rel) x hidden channels
        stacked_embed = th.cat([combined_anchor_embed, rel_embed], dim=1)
        N, anchors_plus_rel, hidden_channels = stacked_embed.shape

        flattened_embed = stacked_embed.view(N, anchors_plus_rel * hidden_channels)
        lin_out = self.lin_layer(flattened_embed)

        return lin_out

    def reset_parameters(self):
        th.nn.init.xavier_uniform_(self.anchor_embed.weight)
        th.nn.init.xavier_uniform_(self.anchor_distances_embed.weight)
        phases = (
            2
            * np.pi
            * th.rand(self.num_relations, self.re_embed_dim, device=self.device)
        )
        relations = th.stack([th.cos(phases), th.sin(phases)], dim=-1)
        self.rel_emb.weight.data = relations.view(
            self.num_relations, self.embedding_dim
        )

    def forward(
        self,
        head_index: th.Tensor,
        rel_type: th.Tensor,
        tail_index: th.Tensor,
        closest_anchors: th.Tensor,
        anchor_distances: th.Tensor,
        rel_hashes: th.Tensor,
    ) -> th.Tensor:

        # N x (hidden channels // 2) x (2 - re, and im)
        h = self.embed_node(
            head_index, closest_anchors, anchor_distances, rel_hashes
        ).view(-1, self.re_embed_dim, 2)
        t = self.embed_node(
            tail_index, closest_anchors, anchor_distances, rel_hashes
        ).view(-1, self.re_embed_dim, 2)
        r = self.rel_emb(rel_type).view(-1, self.re_embed_dim, 2)

        h_re = h[:, :, 0]
        h_im = h[:, :, 1]

        t_re = t[:, :, 0]
        t_im = t[:, :, 1]

        r_re = r[:, :, 0]
        r_im = r[:, :, 1]

        # Hadamard product in complex space
        re_score = (r_re * h_re - r_im * h_im) - t_re
        im_score = (r_re * h_im + r_im * h_re) - t_im
        complex_score = th.stack([re_score, im_score], dim=2)
        scores = self.margin - th.linalg.vector_norm(complex_score, dim=(1, 2))

        return scores

    def loss(
        self,
        head_index: th.Tensor,
        rel_type: th.Tensor,
        tail_index: th.Tensor,
        closest_anchors: th.Tensor,
        anchor_distances: th.Tensor,
        rel_hashes: th.Tensor,
    ) -> th.Tensor:

        pos_score = self(
            head_index,
            rel_type,
            tail_index,
            closest_anchors,
            anchor_distances,
            rel_hashes,
        )
        rand_h, rand_r, rand_t = self.random_sample(head_index, rel_type, tail_index)
        neg_score = self(
            rand_h, rand_r, rand_t, closest_anchors, anchor_distances, rel_hashes
        )

        scores = th.cat([pos_score, neg_score], dim=0)

        pos_target = th.ones_like(pos_score)
        neg_target = th.zeros_like(neg_score)
        target = th.cat([pos_target, neg_target])

        return F.binary_cross_entropy_with_logits(scores, target)


class NodePieceTransE(NodePieceKGModel):
    """Implmentation of TransE for the NodePiece tokenization. It borrows the implementation parts from Pytorch Geometric KGModels."""

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        num_anchors: int,
        max_distance: int,
        top_m_relations: int,
        embedding_dim: int,
        hidden_sizes: Tuple[int, ...],
        drop_prob: float = 0.1,
        margin: float = 1.0,
        p_norm: float = 1.0,
        sparse: bool = False,
        device: str = "cuda",
    ):
        super().__init__(
            num_nodes=num_nodes,
            num_relations=num_relations,
            num_anchors=num_anchors,
            top_m_relations=top_m_relations,
            max_distance=max_distance,
            embedding_dim=embedding_dim,
            hidden_sizes=hidden_sizes,
            p_norm=p_norm,
            sparse=sparse,
            margin=margin,
            drop_prob=drop_prob,
            device=device,
        )
        self.anchor_embed = nn.Embedding(
            self.num_anchors + 1, embedding_dim, sparse=sparse
        )
        self.anchor_distances_embed = nn.Embedding(
            self.max_distance + 1, embedding_dim, sparse=sparse
        )

        layers = []
        last_in = (self.num_anchors + self.top_m_relations) * self.embedding_dim
        layers.append(pyg_nn.BatchNorm(last_in))
        for h_sz in hidden_sizes:
            layers.append(nn.Linear(last_in, h_sz))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(self.drop_prob))
            last_in = h_sz
        layers.append(nn.Linear(last_in, self.embedding_dim))
        self.lin_layer = nn.Sequential(*layers)

        self.rel_emb = nn.Embedding(num_relations, embedding_dim, sparse=sparse)

        self.reset_parameters()

    def embed_node(self, node: th.Tensor, closest_anchors, anchor_distances, rel_hash):

        # Dim: (N x K) values are anchor ids --> (N x K x D)
        anchor_embed = self.anchor_embed(closest_anchors[node])

        # Dim: (N x K) values are anchor distances --> (N x K x D)
        anchor_distances_embed = self.anchor_distances_embed(anchor_distances[node])

        # Dim: (N x M) values are relation types --> (N x M x D)
        rel_embed = self.rel_emb(rel_hash[node])

        # Dim: (N x K x D)
        combined_anchor_embed = anchor_embed + anchor_distances_embed

        # N x (K + M) x D
        stacked_embed = th.cat([combined_anchor_embed, rel_embed], dim=1)
        N, anchors_plus_rel, hidden_channels = stacked_embed.shape

        # reshape: (N x (K + M) x D) --> (N x (K + M) * D)
        flattened_embed = stacked_embed.view(N, anchors_plus_rel * hidden_channels)

        # N x (K + M) * D --> N x O
        lin_out = self.lin_layer(flattened_embed)

        return lin_out

    def reset_parameters(self):
        bound = 6.0 / math.sqrt(self.embedding_dim)
        nn.init.uniform_(self.anchor_embed.weight, -bound, bound)
        nn.init.uniform_(self.anchor_distances_embed.weight, -bound, bound)
        nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        F.normalize(
            self.rel_emb.weight.data,
            p=self.p_norm,
            dim=-1,
            out=self.rel_emb.weight.data,
        )

    def forward(
        self,
        head_index: th.Tensor,
        rel_type: th.Tensor,
        tail_index: th.Tensor,
        closest_anchors: th.Tensor,
        anchor_distances: th.Tensor,
        rel_hashes: th.Tensor,
    ) -> th.Tensor:

        head = self.embed_node(
            head_index, closest_anchors, anchor_distances, rel_hashes
        )
        rel = self.rel_emb(rel_type)
        tail = self.embed_node(
            tail_index, closest_anchors, anchor_distances, rel_hashes
        )

        head = F.normalize(head, p=self.p_norm, dim=-1)
        tail = F.normalize(tail, p=self.p_norm, dim=-1)
        score = -((head + rel) - tail)
        norm_score = score.norm(p=self.p_norm, dim=-1)
        # Calculate *negative* TransE norm:
        return norm_score

    def loss(
        self,
        head_index: th.Tensor,
        rel_type: th.Tensor,
        tail_index: th.Tensor,
        closest_anchors: th.Tensor,
        anchor_distances: th.Tensor,
        rel_hashes: th.Tensor,
    ) -> th.Tensor:

        pos_score = self(
            head_index,
            rel_type,
            tail_index,
            closest_anchors,
            anchor_distances,
            rel_hashes,
        )
        rand_h, rand_r, rand_t = self.random_sample(head_index, rel_type, tail_index)
        neg_score = self(
            rand_h, rand_r, rand_t, closest_anchors, anchor_distances, rel_hashes
        )

        return F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=th.ones_like(pos_score),
            margin=self.margin,
        )


class ModelType(str, Enum):
    """Helper class for defining the model type."""

    RotatE = "rotatE"
    TransE = "tranE"


class KGModelParams(BaseModel):
    """Dataclass for holding the model parameters."""

    num_nodes: int
    num_relations: int
    num_anchors: int
    top_m_relations: int
    max_distance: int
    embedding_dim: int
    hidden_sizes: Tuple[int, ...]
    margin: float = 1.0
    p_norm: float = 1.0
    sparse: bool = False
    drop_prob: float = 0.1
    kg_model_type: ModelType = (ModelType.TransE,)
    device: str = "cuda"

    def get_params_dict(self) -> dict:
        param_d = self.model_dump()
        param_d.pop("kg_model_type")
        return param_d


class NodePiecePL(pl.LightningModule):
    """Pytorch Lightning module for training the NodePieceKGModel. It can work with both: TransE and RotatE models."""

    def __init__(
        self,
        model_params: KGModelParams,
        lr: float = 1e-3,
        log_every_n_steps: int = 1,
        train_features: NodePieceTokens = None,
        val_features: NodePieceTokens = None,
    ):
        super().__init__()
        self.model_params = model_params
        self.train_features = train_features
        self.lr = lr
        self.log_every_n_steps = log_every_n_steps
        self.model = self.build_model()
        self.train_features = train_features
        self.val_features = val_features

    def build_model(self) -> NodePieceKGModel:
        if self.model_params.kg_model_type == ModelType.TransE:
            model = NodePieceTransE(**self.model_params.get_params_dict())
        elif self.model_params.kg_model_type == ModelType.RotatE:
            model = NodePieceRotatE(**self.model_params.get_params_dict())
        else:
            raise ValueError("Unknown model type")
        return model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        head_idx, rel_type, tail_idx = batch
        batch_size = head_idx.size(0)
        loss = self.model.loss(
            head_idx,
            rel_type,
            tail_idx,
            self.train_features.anchor_hashes,
            self.train_features.anchor_distances,
            self.train_features.rel_hashes,
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        head_idx, rel_type, tail_idx = batch
        batch_size = head_idx.size(0)
        mean_rank, hits_at_k = self.model.test(
            head_idx,
            rel_type,
            tail_idx,
            batch_size,
            self.val_features.anchor_hashes,
            self.val_features.anchor_distances,
            self.val_features.rel_hashes,
        )
        self.log("val_mean_rank", mean_rank, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_hits_at_k", hits_at_k, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
