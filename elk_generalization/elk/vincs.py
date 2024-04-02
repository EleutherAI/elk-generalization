from typing import Literal

import torch
from einops import repeat
from torch import Tensor, nn, optim

from elk_generalization.elk.classifier import Classifier


class VincsReporter(Classifier):
    """Implements supervised VINC with Platt scaling."""

    def __init__(
        self,
        in_features: int,
        device: torch.device,
        dtype: torch.dtype,
        w_var: float = 0.0,
        w_inv: float = 1.0,
        w_cov: float = 1.0,
        w_supervised: float = 0.0,
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, 1, device=device, dtype=dtype)

        # Learnable Platt scaling parameter
        self.scale = nn.Parameter(torch.ones(1, device=device, dtype=dtype))

        self.w_var = w_var
        self.w_inv = w_inv
        self.w_cov = w_cov
        self.w_supervised = w_supervised
        self.var = None
        self.inv = None
        self.cov = None
        self.supervised_var = None

    def forward(
        self, x: Tensor, ens: Literal["none", "partial", "full"] = "none"
    ) -> Tensor:
        """Return the credence assigned to the hidden state `x`."""
        raw_scores = self.linear(x).squeeze(-1)
        platt_scaled_scores = (
            raw_scores.mul(self.scale).add(self.linear.bias).squeeze(-1)
        )
        if ens == "none":
            # return the raw scores. (n, v, 2)
            return platt_scaled_scores
        elif ens == "partial":
            # return the difference between the positive and negative scores. (n, v)
            return platt_scaled_scores[..., 1] - platt_scaled_scores[..., 0]
        elif ens == "full":
            # average over the variants. (n,)
            return (platt_scaled_scores[..., 1] - platt_scaled_scores[..., 0]).mean(
                dim=-1
            )
        else:
            raise ValueError(f"Unknown ensemble type: {ens}")

    def raw_forward(self, hiddens: Tensor) -> Tensor:
        return self.linear(hiddens).mul(self.scale).squeeze()

    def fit(self, x: Tensor, y: Tensor):
        """
        x: [n, v, 2, d]
        y: [n]
        """
        neg, pos = x.unbind(2)

        # One-hot indicators for each prompt template
        n, v, d = neg.shape
        assert y.shape == (n,)

        centroids = x.mean(1)  # [n, 2, d]
        neg_centroids, pos_centroids = centroids.unbind(1)  # [n, d]
        # we compute the covariance of pos and neg separately to avoid
        # picking up on the pseudolabel dimension
        self.var = (
            neg_centroids.mT.cov() + pos_centroids.mT.cov()
        ) / 2  # cov assumes [d, n]

        # for invariance, first subtract out the mean over variants for each
        # example (the centering step in covariance calculation)
        neg_centered, pos_centered = (
            neg - neg_centroids[:, None],
            pos - pos_centroids[:, None],
        )
        # then do a batch mat multiplication over examples, summing over examples
        # the result is then divided by (v - 1) (b/c bessel's correction) and averaged over examples
        self.inv = (
            torch.einsum("niv,nvj->ij", neg_centered.mT, neg_centered)
            + torch.einsum("niv,nvj->ij", pos_centered.mT, pos_centered)
        ) / (2 * n * (v - 1))

        xcov = (
            (neg_centroids - neg_centroids.mean(dim=0, keepdim=True)).mT
            @ (pos_centroids - pos_centroids.mean(dim=0, keepdim=True))
            / (n - 1)
        )
        self.cov = 0.5 * (xcov + xcov.mT)

        # Supervised variance is the variance of the 2-datapoint dataset
        # with the centroids of the two classes
        y_dup = torch.cat([y, 1 - y], dim=0).bool()
        flat_centroids = centroids.reshape(-1, d)  # [2n, d]
        self.supervised_var = torch.stack(
            [flat_centroids[y_dup].mean(0), flat_centroids[~y_dup].mean(0)], dim=1
        ).cov()

        # Top principal component of the contrast pair diffs
        objective_mat = (
            self.w_var * self.var
            - self.w_inv * self.inv
            + self.w_cov * self.cov
            + self.w_supervised * self.supervised_var
        )
        _, _, vh = torch.pca_lowrank(objective_mat, q=1, niter=10)

        # Use the TPC as the weight vector
        self.linear.weight.data = vh.T

    def resolve_sign(self, x: Tensor, y: Tensor, max_iter: int = 100):
        _, v, k, _ = x.shape
        y = repeat(to_one_hot(y, k), "n k -> n v k", v=v)

        opt = optim.LBFGS(
            [self.linear.bias, self.scale],
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_change=torch.finfo(x.dtype).eps,
            tolerance_grad=torch.finfo(x.dtype).eps,
        )

        def closure():
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                self.raw_forward(x), y.float()
            )
            loss.backward()
            return float(loss)

        opt.step(closure)


def to_one_hot(labels: Tensor, n_classes: int) -> Tensor:
    """
    Convert a tensor of class labels to a one-hot representation.

    Args:
        labels (Tensor): A tensor of class labels of shape (N,).
        n_classes (int): The total number of unique classes.

    Returns:
        Tensor: A one-hot representation tensor of shape (N, n_classes).
    """
    one_hot_labels = labels.new_zeros(*labels.shape, n_classes)
    return one_hot_labels.scatter_(-1, labels.unsqueeze(-1).long(), 1)
