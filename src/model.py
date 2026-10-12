"""
model.py
Graph Neural Network Architectures for Node Classification on Cora/CiteSeer.

Implements:
1. 2-Layer Graph Convolutional Network (GCN) [Kipf & Welling, ICLR 2017]
   Layer formula: H^(l+1) = sigma( D^(-1/2) * A_hat * D^(-1/2) * H^(l) * W^(l) )
2. 2-Layer Graph Attention Network (GAT) [Veličković et al., ICLR 2018]
   Layer formula: H_i^(l+1) = sigma( sum_{j in N(i)} alpha_{ij} * W * H_j )

Author: Farid Masood Khan
Roadmap Project 2 — Defensible Graph Representation Learning
"""

import math
from typing import Optional, Tuple
import numpy as np


class GCNLayerNumpy:
    """
    NumPy-native reference implementation of Kipf & Welling GCN Layer.
    Useful for line-by-line interview explanation without framework black-boxes.
    """

    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        std = 1.0 / math.sqrt(out_features)
        self.W = rng.uniform(-std, std, (in_features, out_features))
        self.bias = np.zeros(out_features)

    @staticmethod
    def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
        """
        Compute symmetric normalized adjacency with added self-loops:
        A_tilde = A + I
        D_tilde_{ii} = sum_j A_tilde_{ij}
        A_norm = D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2)
        """
        N = adj.shape[0]
        adj_tilde = adj + np.eye(N)
        deg = np.sum(adj_tilde, axis=1)
        deg_inv_sqrt = np.zeros_like(deg, dtype=np.float32)
        nonzero_mask = deg > 0
        deg_inv_sqrt[nonzero_mask] = 1.0 / np.sqrt(deg[nonzero_mask])
        D_inv_sqrt = np.diag(deg_inv_sqrt)
        return D_inv_sqrt @ adj_tilde @ D_inv_sqrt

    def forward(self, H: np.ndarray, A_norm: np.ndarray, activation: str = "relu") -> np.ndarray:
        """
        Message aggregation and linear projection:
        Z = A_norm * H * W + b
        """
        support = H @ self.W
        output = A_norm @ support + self.bias
        if activation == "relu":
            return np.maximum(0, output)
        elif activation == "softmax":
            exp_out = np.exp(output - np.max(output, axis=1, keepdims=True))
            return exp_out / np.sum(exp_out, axis=1, keepdims=True)
        return output


class TwoLayerGCN:
    """
    End-to-end 2-Layer GCN for semi-supervised citation node classification.
    """

    def __init__(self, num_features: int, hidden_dim: int, num_classes: int, seed: int = 42):
        self.layer1 = GCNLayerNumpy(num_features, hidden_dim, seed=seed)
        self.layer2 = GCNLayerNumpy(hidden_dim, num_classes, seed=seed + 1)
        self.hidden_embeddings = None

    def forward(self, X: np.ndarray, A_norm: np.ndarray) -> np.ndarray:
        # Layer 1: message passing + ReLU
        self.hidden_embeddings = self.layer1.forward(X, A_norm, activation="relu")
        # Layer 2: message passing + Softmax classification
        logits = self.layer2.forward(self.hidden_embeddings, A_norm, activation="softmax")
        return logits

    def get_embeddings(self) -> Optional[np.ndarray]:
        return self.hidden_embeddings
