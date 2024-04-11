import torch
from torch import Tensor

from elk_generalization.elk.roc_auc import roc_auc


def eval_random_baseline(
    X_train: Tensor,
    X_test: Tensor,
    Y_train: Tensor,
    Y_test: Tensor,
    num_samples: int = 10_000_000,
) -> dict[str, float | dict[float, float]]:
    n_train, v, _, d = X_train.shape

    # Generate random samples from a standard normal distribution
    Z = torch.randn(num_samples, d, device=X_train.device, dtype=X_train.dtype)
    # "Platt scale"
    Y_hats = torch.einsum("nvki,si->nvks", X_train, Z)
    Y_hats = (
        (Y_hats[..., 1, :] - Y_hats[..., 0, :]).mean(dim=1).reshape(-1, num_samples)
    )

    aurocs = roc_auc(Y_train.expand(num_samples, -1), Y_hats.mT)
    Z *= torch.sign(aurocs[..., None] - 0.5)  # Flip sign of Z if AUROC < 0.5

    # Actually test
    Y_hats = torch.einsum("nvki,si->nvks", X_test, Z)
    Y_hats = (
        (Y_hats[..., 1, :] - Y_hats[..., 0, :]).mean(dim=1).reshape(-1, num_samples)
    )
    aurocs = roc_auc(Y_test.expand(num_samples, -1), Y_hats.mT)

    ps = torch.arange(
        start=-16, end=-1, device=aurocs.device, dtype=aurocs.dtype
    ).exp2()
    ps = torch.cat(
        [
            ps,
            torch.tensor([0.5], device=aurocs.device, dtype=aurocs.dtype),
            1 - ps.flip(0),
        ]
    )

    quantiles = torch.quantile(aurocs, ps)
    return {
        "mean": float(aurocs.mean()),
        "quantiles": {p.item(): q.item() for p, q in zip(ps, quantiles)},
    }
