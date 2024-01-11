import torch
from torch import Tensor, nn, optim
from sklearn.metrics import roc_auc_score, accuracy_score

class MeanDiffReporter(nn.Module):
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
    def resolve_sign(self, labels: Tensor, hiddens: Tensor):
        """Flip the scale term if AUROC < 0.5. Use acc if all labels are the same."""
        labels = labels.cpu().numpy()
        preds = self.forward(hiddens).cpu().numpy()
        if labels.unique().numel() == 1:
            auroc = accuracy_score(labels, preds > 0)
        else:
            auroc = roc_auc_score(labels, preds)
        if float(auroc) < 0.5:
            self.scale.data = -self.scale.data
