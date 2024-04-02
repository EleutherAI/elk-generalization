import torch
from classifier import Classifier
from concept_erasure.shrinkage import optimal_linear_shrinkage
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import Tensor, nn


class LdaReporter(Classifier):
    def __init__(self, in_features: int, device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.linear = nn.Linear(in_features, 1, device=device, dtype=dtype)

        # Learnable Platt scaling parameter
        self.scale = nn.Parameter(torch.ones(1, device=device, dtype=dtype))

    def forward(self, hiddens: Tensor) -> Tensor:
        return self.linear(hiddens).mul(self.scale).squeeze()

    def fit2(self, x: Tensor, y: Tensor):
        x0, x1 = x[y == 0], x[y == 1]
        mu0, mu1 = x0.mean(dim=0), x1.mean(dim=0)

        # Between-class covariance
        B = torch.cov(
            torch.stack([mu0, mu1]).T,
            fweights=torch.tensor([len(x0), len(x1)], device=x.device),
        )

        # Within-class covariance
        cov0 = optimal_linear_shrinkage(x0.mT.cov(correction=0), len(x0))
        cov1 = optimal_linear_shrinkage(x1.mT.cov(correction=0), len(x1))
        W = (len(x0) * cov0 + len(x1) * cov1) / len(x)

        # Get the top generalized eigenpair
        _, w = torch.lobpcg(B.to(torch.float64), B=W.to(torch.float64))
        w = w / w.norm()

        self.linear.weight.data = w.mT.to(self.linear.weight.dtype)

    def fit(self, x: Tensor, y: Tensor):
        x0, x1 = x[y == 0], x[y == 1]
        mu0, mu1 = x0.mean(dim=0), x1.mean(dim=0)

        # Within-class covariance
        cov0 = x0.mT.cov()
        cov1 = x1.mT.cov()

        S = (len(x0) * cov0 + len(x1) * cov1) / len(x)
        precision = torch.linalg.pinv(S)
        w = precision @ (mu1 - mu0)

        self.linear.weight.data = w[None].to(self.linear.weight.dtype)

    @torch.no_grad()
    def resolve_sign(
        self,
        x: Tensor,
        y: Tensor,
    ):
        """Flip the scale term if AUROC < 0.5. Use acc if all labels are the same."""
        y = y.cpu().numpy()
        preds = self.forward(x).cpu().numpy()
        if len(set(y)) == 1:
            auroc = accuracy_score(y, preds > 0)
        else:
            auroc = roc_auc_score(y, preds)
        if float(auroc) < 0.5:
            self.scale.data = -self.scale.data
