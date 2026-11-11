"""Feed-forward neural network (PyTorch), trained by us."""

import numpy as np
import torch
from torch import nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}
OPTIMIZERS = ("adam", "adamw", "sgd")
SCHEDULERS = ("none", "cosine")

# Final configuration per task, chosen on the validation split
# (see scripts/nn_experiments.py and results/neural_net.csv).
# "preprocessing" names a transform chain in nn_experiments.TRANSFORMS;
# everything else is a NeuralNetClassifier argument.
TUNED_CONFIGS = {
    # Plain MLPs memorise the image training set (100% train) and plateau
    # around 34-37% val. Square-rooting the count histograms, a wider net and
    # strong regularisation (dropout + AdamW weight decay, cosine LR decay)
    # give ~39%. k-NN does much better on this task (see the report).
    "image": {
        "preprocessing": "sqrt+standard",
        "hidden_sizes": (1024, 512),
        "dropout": 0.5,
        "lr": 1e-3,
        "optimizer": "adamw",
        "weight_decay": 0.1,
        "epochs": 60,
        "scheduler": "cosine",
    },
    # The base net (lr 1e-3, no regularisation) peaks after 1-3 epochs and
    # then overfits. A 10x smaller learning rate with dropout + weight decay
    # trains slowly and stays stable: ~68% val at the last epoch.
    "sentiment": {
        "preprocessing": "standard",
        "hidden_sizes": (512,),
        "dropout": 0.5,
        "lr": 1e-4,
        "optimizer": "adamw",
        "weight_decay": 0.1,
        "epochs": 20,
        "scheduler": "cosine",
    },
}


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
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}, got {activation!r}")

        layers: list[nn.Module] = []
        width_in = n_features
        for width_out in hidden_sizes:
            layers.append(nn.Linear(width_in, width_out))
            layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width_in = width_out
        # Output layer: one logit per class. No softmax here, because
        # nn.CrossEntropyLoss applies log-softmax itself (numerically safer).
        layers.append(nn.Linear(width_in, n_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class NeuralNetClassifier:
    """fit/predict wrapper around ``NeuralNet`` so it plugs into ``evaluate``.

    Parameters
    ----------
    hidden_sizes, activation, dropout : network architecture.
    lr, optimizer, weight_decay, batch_size, epochs : training settings.
        optimizer is "adam", "adamw" or "sgd" (SGD uses momentum 0.9).
    scheduler : "none" (constant learning rate) or "cosine" (learning rate
        decays smoothly from ``lr`` to ~0 over ``epochs``).
    device : "cpu", "cuda", or None to pick automatically.
    seed : seed for weight initialisation and batch shuffling.

    After ``fit``, ``history_`` holds one entry per epoch with the training
    loss/accuracy and, if ``X_val``/``y_val`` were given, the validation
    loss/accuracy. Use it to plot learning curves.
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
        scheduler: str = "none",
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
        self.scheduler = scheduler
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "NeuralNetClassifier":
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SCHEDULERS}, got {self.scheduler!r}")
        if self.seed is not None:
            torch.manual_seed(self.seed)  # weight initialisation

        # Labels -> positions 0..n_classes-1 (CrossEntropyLoss needs that);
        # predict() maps them back through classes_.
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        X_t = self._to_tensor(X)
        y_t = torch.as_tensor(y_idx, dtype=torch.long, device=self.device)

        val = None
        if X_val is not None and y_val is not None:
            # Validation labels must be encoded with the *training* classes.
            pos = np.searchsorted(self.classes_, np.asarray(y_val))
            pos = np.clip(pos, 0, len(self.classes_) - 1)
            if not np.all(self.classes_[pos] == np.asarray(y_val)):
                raise ValueError("y_val contains labels not seen in y")
            val = (self._to_tensor(X_val), torch.as_tensor(pos, dtype=torch.long, device=self.device))

        self.model_ = NeuralNet(
            X_t.shape[1], len(self.classes_), tuple(self.hidden_sizes), self.activation, self.dropout
        ).to(self.device)
        opt = self._make_optimizer()
        sched = (
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
            if self.scheduler == "cosine"
            else None
        )
        loss_fn = nn.CrossEntropyLoss()
        gen = torch.Generator().manual_seed(self.seed if self.seed is not None else 0)

        self.history_: list[dict] = []
        n = len(X_t)
        for epoch in range(1, self.epochs + 1):
            self.model_.train()  # enables dropout
            perm = torch.randperm(n, generator=gen).to(self.device)
            total_loss, correct = 0.0, 0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb, yb = X_t[idx], y_t[idx]
                logits = self.model_(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(idx)
                correct += (logits.argmax(1) == yb).sum().item()
            if sched is not None:
                sched.step()

            # train_loss / train_acc are averaged over the epoch while the
            # weights were still changing (and with dropout on); the *_eval
            # versions re-score the training set at the end of the epoch in
            # eval mode, which is the fair comparison with val_acc.
            row = {
                "epoch": epoch,
                "lr": opt.param_groups[0]["lr"],
                "train_loss": total_loss / n,
                "train_acc_running": correct / n,
            }
            row["train_loss_eval"], row["train_acc"] = self._score(X_t, y_t, loss_fn)
            if val is not None:
                row["val_loss"], row["val_acc"] = self._score(*val, loss_fn)
            self.history_.append(row)
        return self

    # -------------------------------------------------------------- predict
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Softmax class probabilities, shape (n_samples, n_classes)."""
        logits = self._logits(self._to_tensor(X))
        return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self._logits(self._to_tensor(X))
        return self.classes_[logits.argmax(1).cpu().numpy()]

    # -------------------------------------------------------------- helpers
    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)

    @torch.no_grad()
    def _logits(self, X_t: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
        self.model_.eval()  # disables dropout
        return torch.cat([self.model_(X_t[s : s + chunk]) for s in range(0, len(X_t), chunk)])

    def _score(self, X_t: torch.Tensor, y_t: torch.Tensor, loss_fn) -> tuple[float, float]:
        logits = self._logits(X_t)
        loss = loss_fn(logits, y_t).item()
        acc = (logits.argmax(1) == y_t).float().mean().item()
        return loss, acc

    def _make_optimizer(self) -> torch.optim.Optimizer:
        params = self.model_.parameters()
        if self.optimizer == "adam":
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        return torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
