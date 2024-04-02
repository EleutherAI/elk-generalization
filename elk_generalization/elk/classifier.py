from abc import ABC, abstractmethod

from torch import Tensor, nn


class Classifier(ABC, nn.Module):
    @abstractmethod
    def forward(self, hiddens: Tensor) -> Tensor:
        pass

    @abstractmethod
    def fit(self, x: Tensor, y: Tensor):
        pass

    @abstractmethod
    def resolve_sign(self, x: Tensor, y: Tensor) -> Tensor:
        pass
