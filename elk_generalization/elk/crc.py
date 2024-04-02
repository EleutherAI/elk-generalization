import torch
import torch.nn.functional as F
from classifier import Classifier
from concept_erasure import LeaceEraser
from torch import Tensor, nn, optim


class CrcReporter(Classifier):
    def __init__(self, in_features: int, device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.linear = nn.Linear(in_features, 1, device=device, dtype=dtype)
        self.eraser = None

        # Learnable Platt scaling parameter
        self.scale = nn.Parameter(torch.ones(1, device=device, dtype=dtype))

    def forward(self, hiddens: Tensor) -> Tensor:
        return self.raw_forward(hiddens).diff(dim=1).squeeze()

    def raw_forward(self, hiddens: Tensor) -> Tensor:
        if self.eraser is not None:
            hiddens = self.eraser(hiddens)
        return self.linear(hiddens).mul(self.scale).squeeze()

    def fit(self, x: Tensor):
        n = len(x)

        self.eraser = LeaceEraser.fit(
            x=x.flatten(0, 1),
            z=torch.stack([x.new_zeros(n), x.new_ones(n)], dim=1).flatten(),
        )
        x = self.eraser(x)

        # Top principal component of the contrast pair diffs
        neg, pos = x.unbind(-2)
        _, _, vh = torch.pca_lowrank(pos - neg, q=1, niter=10)

        # Use the TPC as the weight vector
        self.linear.weight.data = vh.T

    def resolve_sign(self, x: Tensor, y: Tensor, max_iter: int = 100):
        """Fit the scale and bias terms to data with LBFGS.

        Args:
            labels: Binary labels of shape [batch].
            hiddens: Hidden states of shape [batch, dim].
            max_iter: Maximum number of iterations for LBFGS.
        """
        _, k, _ = x.shape
        y = F.one_hot(y.long(), k)

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
