from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.dataset.records import Record

_EPS = 1e-6


@dataclass
class TrainStats:
    n_examples: int
    steps: int
    final_loss: float


def _scale(bundle, X: np.ndarray) -> np.ndarray:
    if bundle.scaler_mean is None:
        return X
    return (X - bundle.scaler_mean) / (bundle.scaler_std + _EPS)


def local_train(bundle, records: list[Record], config: dict, seed: int) -> TrainStats:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    tc = config.get("train", {})
    lr = float(tc.get("learning_rate", 1e-3))
    epochs = int(tc.get("local_epochs", 3))
    batch_size = int(tc.get("batch_size", 32))
    max_grad_norm = float(tc.get("max_grad_norm", 1.0))

    if not records:
        return TrainStats(n_examples=0, steps=0, final_loss=float("nan"))

    X = _scale(bundle, np.stack([r.features for r in records]))
    y = np.array([r.label for r in records], dtype=np.int64)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)

    optimizer = torch.optim.AdamW(bundle.model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    bundle.model.train()
    steps = 0
    last_loss = float("nan")
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(bundle.device), yb.to(bundle.device)
            optimizer.zero_grad()
            logits = bundle.model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bundle.model.parameters(), max_grad_norm)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            steps += 1
    return TrainStats(n_examples=len(records), steps=steps, final_loss=last_loss)


def compute_metrics(y: np.ndarray, preds: np.ndarray, loss: float) -> dict[str, float]:
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, preds, average="macro", zero_division=0, labels=sorted(set(y.tolist()))
    )
    accuracy = float(np.mean(preds == y))

    return {
        "n": len(y),
        "eval_loss": loss,
        "accuracy": accuracy,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def evaluate_model(bundle, records: list[Record], config: dict) -> dict[str, float]:
    import torch

    if not records:
        return {
            "n": 0,
            "eval_loss": float("nan"),
            "accuracy": float("nan"),
            "precision_macro": float("nan"),
            "recall_macro": float("nan"),
            "f1_macro": float("nan"),
        }

    X = _scale(bundle, np.stack([r.features for r in records]))
    y = np.array([r.label for r in records], dtype=np.int64)

    bundle.model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=bundle.device)
        yb = torch.tensor(y, dtype=torch.long, device=bundle.device)
        logits = bundle.model(xb)
        loss = float(torch.nn.functional.cross_entropy(logits, yb).cpu())
        preds = logits.argmax(dim=1).cpu().numpy()

    return compute_metrics(y, preds, loss)
