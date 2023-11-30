import torch
from torch import Tensor
from argparse import ArgumentParser

def roc_auc(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Area under the receiver operating characteristic curve (ROC AUC).

    Unlike scikit-learn's implementation, this function supports batched inputs of
    shape `(N, n)` where `N` is the number of datasets and `n` is the number of samples
    within each dataset. This is primarily useful for efficiently computing bootstrap
    confidence intervals.

    Args:
        y_true: Ground truth tensor of shape `(N,)` or `(N, n)`.
        y_pred: Predicted class tensor of shape `(N,)` or `(N, n)`.

    Returns:
        Tensor: If the inputs are 1D, a scalar containing the ROC AUC. If they're 2D,
            a tensor of shape (N,) containing the ROC AUC for each dataset.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred should have the same shape; "
            f"got {y_true.shape} and {y_pred.shape}"
        )
    if y_true.dim() not in (1, 2):
        raise ValueError("y_true and y_pred should be 1D or 2D tensors")

    # Sort y_pred in descending order and get indices
    indices = y_pred.argsort(descending=True, dim=-1)

    # Reorder y_true based on sorted y_pred indices
    y_true_sorted = y_true.gather(-1, indices)

    # Calculate number of positive and negative samples
    num_positives = y_true.sum(dim=-1)
    num_negatives = y_true.shape[-1] - num_positives

    # Calculate cumulative sum of true positive counts (TPs)
    tps = torch.cumsum(y_true_sorted, dim=-1)

    # Calculate cumulative sum of false positive counts (FPs)
    fps = torch.cumsum(1 - y_true_sorted, dim=-1)

    # Calculate true positive rate (TPR) and false positive rate (FPR)
    tpr = tps / num_positives.view(-1, 1)
    fpr = fps / num_negatives.view(-1, 1)

    # Calculate differences between consecutive FPR values (widths of trapezoids)
    fpr_diffs = torch.cat(
        [fpr[..., 1:] - fpr[..., :-1], torch.zeros_like(fpr[..., :1])], dim=-1
    )

    # Calculate area under the ROC curve for each dataset using trapezoidal rule
    return torch.sum(tpr * fpr_diffs, dim=-1).squeeze()
