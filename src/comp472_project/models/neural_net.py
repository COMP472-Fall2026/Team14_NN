"""Feed-forward neural network (PyTorch), trained by us."""

import numpy as np
import torch
from torch import nn


class NeuralNet(nn.Module):
    """Input -> [Linear -> activation -> dropout] * len(hidden_sizes) -> Linear."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_sizes: tuple[int, ...] = (256, 64),
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        # TODO: build the layers.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: return logits (no softmax; nn.CrossEntropyLoss applies it).
        raise NotImplementedError


class NeuralNetClassifier:
    """fit/predict wrapper around ``NeuralNet`` so it plugs into ``evaluate``.

    Parameters
    ----------
    hidden_sizes, activation, dropout : network architecture.
    lr, optimizer, weight_decay, batch_size, epochs : training settings.
    device : "cpu", "cuda", or None to pick automatically.
    seed : seed for weight initialisation and batch shuffling.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (256, 64),
        activation: str = "relu",
        dropout: float = 0.0,
        lr: float = 1e-3,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        batch_size: int = 128,
        epochs: int = 30,
        device: str | None = None,
        seed: int | None = None,
    ):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetClassifier":
        # TODO: build NeuralNet, train with cross-entropy loss over mini-batches.
        # Consider recording per-epoch train/val loss in self.history_ so you
        # can plot learning curves for the over/underfitting discussion.
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: argmax of the network's logits (in eval mode, without grad).
        raise NotImplementedError
