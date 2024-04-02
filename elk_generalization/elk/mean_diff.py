import torch
from classifier import Classifier
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import Tensor, nn


class MeanDiffReporter(Classifier):
    def __init__(self, in_features: int, device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.linear = nn.Linear(in_features, 1, device=device, dtype=dtype)

        # Learnable Platt scaling parameter
        self.scale = nn.Parameter(torch.ones(1, device=device, dtype=dtype))

    def forward(self, hiddens: Tensor) -> Tensor:
        return self.linear(hiddens).mul(self.scale).squeeze()

    def fit(self, x: Tensor, y: Tensor):
        assert x.ndim == 2, "x must have shape [n, d]"
        diff = x[y == 1].mean(dim=0) - x[y == 0].mean(dim=0)
        diff = diff / diff.norm()

        self.linear.weight.data = diff.unsqueeze(0)

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
