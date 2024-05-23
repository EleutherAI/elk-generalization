from typing import Literal

import torch
from concept_erasure import LeaceEraser
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import Tensor, nn

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
        leace_pseudolabels: bool = False,
        leace_variants: bool = False,
        use_centroids: bool = True,
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, 1, device=device, dtype=dtype)

        # Learnable Platt scaling parameter
        self.scale = nn.Parameter(torch.ones(1, device=device, dtype=dtype))

        self.eraser = None

        self.w_var = w_var
        self.w_inv = w_inv
        self.w_cov = w_cov
        self.w_supervised = w_supervised
        self.var = None
        self.inv = None
        self.cov = None
        self.supervised_var = None
        self.leace_pseudolabels = leace_pseudolabels
        self.leace_variants = leace_variants
        self.use_centroids = use_centroids

    def forward(
        self, x: Tensor, ens: Literal["none", "partial", "full"] = "none"
    ) -> Tensor:
        """Return the credence assigned to the hidden state `x`."""
        n, v, _, d = x.shape
        if self.eraser is not None:
            x = self.eraser(x)
        platt_scaled_scores = (
            self.linear(x).mul(self.scale).add(self.linear.bias).reshape(n, v, 2)
        )
        if ens == "none":
            # return the raw scores. -> (n, v, 2)
            return platt_scaled_scores
        elif ens == "partial":
            # return the difference between the positive and negative scores. -> (n, v)
            return platt_scaled_scores[..., 1] - platt_scaled_scores[..., 0]
        elif ens == "full":
            # average over the variants. -> (n,)
            return (platt_scaled_scores[..., 1] - platt_scaled_scores[..., 0]).mean(
                dim=-1
            )
        else:
            raise ValueError(f"Unknown ensemble type: {ens}")

    def fit(self, x: Tensor, y: Tensor):
        """
        x: [n, v, 2, d]
        y: [n]
        """
        n, v, _, d = x.shape
        if self.leace_pseudolabels or self.leace_variants:
            if self.leace_pseudolabels and self.leace_variants:
                indicators = (
                    torch.eye(2 * v, device=x.device)
                    .expand(n, -1, -1)
                    .reshape(n, v, 2, 2 * v)
                )
            elif self.leace_pseudolabels:
                indicators = torch.eye(2, device=x.device).expand(
                    n, v, -1, -1
                )  # [n, v, 2, 2]
            elif self.leace_variants:
                indicators = (
                    torch.eye(v, device=x.device).expand(n, 2, -1, -1).transpose(1, 2)
                )  # [n, v, 2, v]

            self.eraser = LeaceEraser.fit(
                x=x.reshape(-1, x.shape[-1]),
                z=indicators.reshape(-1, indicators.shape[-1]),
            )
            x = self.eraser(x)

        neg, pos = x.unbind(2)

        # One-hot indicators for each prompt template
        assert y.shape == (n,)

        centroids = x.mean(1)  # [n, 2, d]
        neg_centroids, pos_centroids = centroids.unbind(1)  # [n, d]
        neg_data = neg_centroids if self.use_centroids else neg.reshape(-1, d)
        pos_data = pos_centroids if self.use_centroids else pos.reshape(-1, d)
        # we compute the covariance of pos and neg separately to avoid
        # picking up on the pseudolabel dimension

        self.var = (neg_data.mT.cov() + neg_data.mT.cov()) / 2  # cov assumes [d, n]

        if self.w_inv == 0:
            self.inv = torch.zeros(d, d, device=x.device, dtype=x.dtype)
        else:
            # for invariance, first subtract out the mean over variants for each
            # example (the centering step in covariance calculation)
            neg_centered, pos_centered = (
                neg - neg_centroids[:, None],
                pos - pos_centroids[:, None],
            )
            # then do a batch mat multiplication over examples, summing over examples
            # the result is then divided by (v - 1) (b/c bessel's correction) and averaged
            self.inv = (
                torch.einsum("niv,nvj->ij", neg_centered.mT, neg_centered)
                + torch.einsum("niv,nvj->ij", pos_centered.mT, pos_centered)
            ) / (2 * n * (v - 1))

        xcov = (
            (neg_data - neg_data.mean(dim=0, keepdim=True)).mT
            @ (pos_data - pos_data.mean(dim=0, keepdim=True))
            / (n - 1)
        )
        self.cov = 0.5 * (xcov + xcov.mT)

        # Supervised variance is the variance of the 2-datapoint dataset
        # with the centroids of the two classes
        y_dup = torch.cat([1 - y, y], dim=0).bool()
        flat_centroids = centroids.reshape(-1, d)  # [2n, d]
        self.supervised_var = torch.stack(
            [flat_centroids[y_dup].mean(0), flat_centroids[~y_dup].mean(0)], dim=1
        ).cov()

        objective_mat = (
            self.w_var * self.var
            - self.w_inv * self.inv
            + self.w_cov * self.cov
            + self.w_supervised * self.supervised_var
        )
        # assert objective_mat is hermitian
        assert torch.allclose(objective_mat, objective_mat.mT)

        try:
            L, Q = torch.linalg.eigh(objective_mat)
        except torch.linalg.LinAlgError:
            try:
                L, Q = torch.linalg.eig(objective_mat)
                L, Q = L.real, Q.real
            except torch.linalg.LinAlgError as e:
                # Check if the matrix has non-finite values
                if not objective_mat.isfinite().all():
                    raise ValueError(
                        "Fitting the reporter failed because the VINC matrix has "
                        "non-finite entries. Usually this means the hidden states "
                        "themselves had non-finite values."
                    ) from e
                else:
                    raise e

        # take the algebraically largest eigenval's eigenvector
        vh = Q[:, torch.argmax(L)]

        # Use the TPC as the weight vector
        self.linear.weight.data = vh.reshape(1, d)

    def resolve_sign(self, x: Tensor, y: Tensor, max_iter: int = 100):
        """Flip the scale term if AUROC < 0.5. Use acc if all labels are the same."""
        y = y.detach().cpu().numpy()
        preds = self(x, ens="full").detach().cpu().numpy()

        if len(set(y)) == 1:
            auroc = accuracy_score(y, preds > 0)
        else:
            auroc = roc_auc_score(y, preds)
        if float(auroc) < 0.5:
            self.scale.data = -self.scale.data
