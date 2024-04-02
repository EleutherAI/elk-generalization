"""A CCS reporter network."""

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, cast

import torch
import torch.nn as nn
from classifier import Classifier
from concept_erasure import LeaceFitter
from einops import repeat
from torch import Tensor, optim
from typing_extensions import override

from elk_generalization.elk.burns_norm import BurnsNorm
from elk_generalization.elk.ccs_losses import LOSSES, parse_loss


@dataclass
class CcsConfig:
    bias: bool = True
    """Whether to use a bias term in the linear layers."""
    init: Literal["default", "pca", "spherical", "zero"] = "default"
    """The initialization scheme to use."""
    loss: list[str] = field(default_factory=lambda: ["ccs"])
    """
    The loss function to use. list of strings, each of the form "coef*name", where coef
    is a float and name is one of the keys in `ccs.training.losses.LOSSES`.
    Example: `--loss 1.0*consistency_squared 0.5*prompt_var` corresponds to the loss
    function 1.0*consistency_squared + 0.5*prompt_var.
    """
    loss_dict: dict[str, float] = field(default_factory=dict, init=False)
    norm: Literal["leace", "burns", "meanonly"] = "meanonly"

    lr: float = 1e-2
    """The learning rate to use. Ignored when `optimizer` is `"lbfgs"`."""
    num_epochs: int = 1000
    """The number of epochs to train for."""
    num_tries: int = 10
    """The number of times to try training the reporter."""
    optimizer: Literal["adam", "lbfgs"] = "lbfgs"
    """The optimizer to use."""
    weight_decay: float = 0.01
    """The weight decay or L2 penalty to use."""

    def __post_init__(self):
        self.loss_dict = parse_loss(self.loss)

        # standardize the loss field
        self.loss = [f"{coef}*{name}" for name, coef in self.loss_dict.items()]


class CcsReporter(Classifier):
    """CCS reporter network.

    Args:
        in_features: The number of input features.
        cfg: The reporter configuration.
    """

    config: CcsConfig

    def __init__(
        self,
        cfg: CcsConfig,
        in_features: int,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        num_variants: int = 1,
    ):
        super().__init__()

        self.config = cfg
        self.in_features = in_features
        self.num_variants = num_variants

        # Learnable Platt scaling parameters
        self.bias = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
        self.scale = nn.Parameter(torch.ones(1, device=device, dtype=dtype))

        self.linear = nn.Linear(
            in_features, 1, bias=cfg.bias, device=device, dtype=dtype
        )

    @override
    def parameters(self, recurse=True):
        parameters = super(CcsReporter, self).parameters(recurse=recurse)
        for param in parameters:
            # exclude the platt scaling parameters
            # kind of a hack for now, we should find probably a cleaner way
            if param is not self.scale and param is not self.bias:
                yield param

    def maybe_unsqueeze(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            return x.unsqueeze(1)
        assert x.ndim == 4, f"Expected input of shape [n, v, 2, d], got {x.shape}"
        return x

    def reset_parameters(self):
        """Reset the parameters of the probe.

        If init is "spherical", use the spherical initialization scheme.
        If init is "default", use the default PyTorch initialization scheme for
        nn.Linear (Kaiming uniform).
        If init is "zero", initialize all parameters to zero.
        """
        if self.config.init == "spherical":
            # Mathematically equivalent to the unusual initialization scheme used in
            # the original paper. They sample a Gaussian vector of dim in_features + 1,
            # normalize to the unit sphere, then add an extra all-ones dimension to the
            # input and compute the inner product. Here, we use nn.Linear with an
            # explicit bias term, but use the same initialization.
            probe = cast(nn.Linear, self.linear)  # Pylance gets the type wrong here

            theta = torch.randn(1, probe.in_features + 1, device=probe.weight.device)
            theta /= theta.norm()
            probe.weight.data = theta[:, :-1]
            probe.bias.data = theta[:, -1]

        elif self.config.init == "default":
            self.linear.reset_parameters()

        elif self.config.init == "zero":
            for param in self.parameters():
                param.data.zero_()
        elif self.config.init != "pca":
            raise ValueError(f"Unknown init: {self.config.init}")

    def forward(
        self, x: Tensor, ens: Literal["none", "partial", "full"] = "none"
    ) -> Tensor:
        """Return the credence assigned to the hidden state `x`."""
        assert self.norm is not None, "Must call fit() before forward()"
        x = self.maybe_unsqueeze(x)
        raw_scores = self.linear(self.norm(x)).squeeze(-1)
        platt_scaled_scores = raw_scores.mul(self.scale).add(self.bias).squeeze(-1)
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

    def loss(self, logit0: Tensor, logit1: Tensor) -> Tensor:
        """Return the loss of the reporter on the contrast pair (x0, x1).

        Args:
            logit0: The raw score output of the reporter on x0.
            logit1: The raw score output of the reporter on x1.

        Returns:
            loss: The loss of the reporter on the contrast pair (x0, x1).
        """
        loss = sum(
            LOSSES[name](logit0, logit1, coef)
            for name, coef in self.config.loss_dict.items()
        )
        return loss  # type: ignore

    def fit(self, hiddens: Tensor) -> float:
        """Fit the probe to the contrast pair `hiddens`.

        Returns:
            best_loss: The best loss obtained.
        """
        self.maybe_unsqueeze(hiddens)
        x_neg, x_pos = hiddens.unbind(2)

        # One-hot indicators for each prompt template
        n, v, d = x_neg.shape
        prompt_ids = torch.eye(v, device=x_neg.device).expand(n, -1, -1)

        if self.config.norm == "burns":
            self.norm = BurnsNorm()
        elif self.config.norm == "meanonly":
            self.norm = BurnsNorm(scale=False)
        else:
            fitter = LeaceFitter(d, 2 * v, dtype=x_neg.dtype, device=x_neg.device)
            fitter.update(
                x=x_neg,
                # Independent indicator for each (template, pseudo-label) pair
                z=torch.cat([torch.zeros_like(prompt_ids), prompt_ids], dim=-1),
            )
            fitter.update(
                x=x_pos,
                # Independent indicator for each (template, pseudo-label) pair
                z=torch.cat([prompt_ids, torch.zeros_like(prompt_ids)], dim=-1),
            )
            self.norm = fitter.eraser

        x_neg, x_pos = self.norm(x_neg), self.norm(x_pos)

        # Record the best acc, loss, and params found so far
        best_loss = torch.inf
        best_state: dict[str, Tensor] = {}  # State dict of the best run

        for i in range(self.config.num_tries):
            self.reset_parameters()

            # This is sort of inefficient but whatever
            if self.config.init == "pca":
                diffs = torch.flatten(x_pos - x_neg, 0, 1)
                _, __, V = torch.pca_lowrank(diffs, q=i + 1)
                self.linear.weight.data = V[:, -1, None].T

            if self.config.optimizer == "lbfgs":
                loss = self.train_loop_lbfgs(x_neg, x_pos)
            elif self.config.optimizer == "adam":
                loss = self.train_loop_adam(x_neg, x_pos)
            else:
                raise ValueError(f"Optimizer {self.config.optimizer} is not supported")

            if loss < best_loss:
                best_loss = loss
                best_state = deepcopy(self.state_dict())

        if not math.isfinite(best_loss):
            raise RuntimeError("Got NaN/infinite loss during training")

        self.load_state_dict(best_state)

        return best_loss

    def train_loop_adam(self, x_neg: Tensor, x_pos: Tensor) -> float:
        """Adam train loop, returning the final loss. Modifies params in-place."""

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )

        loss = torch.inf
        for _ in range(self.config.num_epochs):
            optimizer.zero_grad()

            # We already normalized in fit()
            loss = self.loss(self(x_neg), self(x_pos))
            loss.backward()
            optimizer.step()

        return float(loss)

    def train_loop_lbfgs(self, x_neg: Tensor, x_pos: Tensor) -> float:
        """LBFGS train loop, returning the final loss. Modifies params in-place."""

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=self.config.num_epochs,
            tolerance_change=torch.finfo(x_pos.dtype).eps,
            tolerance_grad=torch.finfo(x_pos.dtype).eps,
        )
        # Raw unsupervised loss, WITHOUT regularization
        loss = torch.inf

        def closure():
            nonlocal loss
            optimizer.zero_grad()

            # We already normalized in fit()
            loss = self.loss(self(x_neg), self(x_pos))
            regularizer = 0.0

            # We explicitly add L2 regularization to the loss, since LBFGS
            # doesn't have a weight_decay parameter
            for param in self.parameters():
                regularizer += self.config.weight_decay * param.norm() ** 2 / 2

            regularized = loss + regularizer
            regularized.backward()

            return float(regularized)

        optimizer.step(closure)
        return float(loss)

    def resolve_sign(self, x: Tensor, y: Tensor, max_iter: int = 100):
        """Fit the scale and bias terms to data with LBFGS.

        Args:
            labels: Binary labels of shape [batch].
            hiddens: Hidden states of shape [batch, dim].
            max_iter: Maximum number of iterations for LBFGS.
        """
        x = self.maybe_unsqueeze(x)
        _, v, k, _ = x.shape
        y = repeat(to_one_hot(y, k), "n k -> n v k", v=v)

        opt = optim.LBFGS(
            [self.bias, self.scale],
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_change=torch.finfo(x.dtype).eps,
            tolerance_grad=torch.finfo(x.dtype).eps,
        )

        def closure():
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                self(x, ens="none"), y.float()
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
