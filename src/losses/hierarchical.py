"""Hierarchical ontology-aware loss functions."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.heads import NULL_TARGET_SENTINEL


class HierarchicalCrossEntropyLoss(nn.Module):
    """Ontology-aware cross entropy for ICD-O-3 axes.

    Converts hard targets into soft targets based on a precomputed distance
    matrix, penalizing "near misses" less than "far misses".
    """
    def __init__(
        self,
        distance_matrix: torch.Tensor,
        smoothing: float = 0.3,
        axis: str = "",
    ):
        super().__init__()
        self.axis = axis
        self.smoothing = smoothing
        
        # [K, K] distance matrix. We convert it to a soft target distribution.
        # similarity = 1.0 - distance
        similarity = 1.0 - distance_matrix
        # zero out the diagonal so a class doesn't distribute mass to itself
        similarity.fill_diagonal_(0.0)
        
        # normalize rows to sum to 1.0
        row_sums = similarity.sum(dim=1, keepdim=True)
        # avoid division by zero
        safe_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
        redistribution = similarity / safe_sums
        
        # create the soft target matrix M of shape [K, K]
        # M[i] is the soft target distribution when the true class is i
        # M[i, j] = smoothing * redistribution[i, j] if i != j
        # M[i, i] = 1.0 - smoothing
        k = distance_matrix.size(0)
        M = redistribution * smoothing
        M.diagonal().copy_(torch.full((k,), 1.0 - smoothing, dtype=M.dtype))
        
        # Handle rows that had sum 0 (e.g. all distances were 1.0)
        # Just put all mass on the target class
        zero_rows = (row_sums.squeeze(1) == 0)
        M[zero_rows] = 0.0
        M[zero_rows, zero_rows] = 1.0
        
        # store as a buffer so it moves to device
        self.register_buffer("soft_target_matrix", M)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Compute loss.
        
        Args:
            logits: [B, K]
            targets: [B]
        """
        valid_mask = targets != NULL_TARGET_SENTINEL
        if not valid_mask.any():
            return logits.sum() * 0.0, 0
            
        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]
        
        # Get soft targets for the valid batch
        # [N_valid, K]
        soft_targets = self.soft_target_matrix[valid_targets]
        
        # Cross entropy with soft targets: -sum(target * log_softmax(logits))
        log_probs = F.log_softmax(valid_logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        
        return loss, int(valid_mask.sum())
