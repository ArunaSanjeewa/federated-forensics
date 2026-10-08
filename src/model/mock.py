from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.dataset.records import Record

_DIM = 32


@dataclass
class MockBundle:
    device: str = "cpu"
    param_keys: list[str] = field(default_factory=lambda: ["W", "b"])
    n_classes: int = 2
    W: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    b: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    proj: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    scaler_mean: Optional[np.ndarray] = None
    scaler_std: Optional[np.ndarray] = None


def build_mock_bundle(
    config: dict,
    seed: int = 42,
    n_classes: int = 2,
    scaler_mean=None,
    scaler_std=None,
    feature_dim: int = 2381,
) -> MockBundle:
    rng = np.random.default_rng(seed)
    proj = rng.normal(0, 1.0 / np.sqrt(feature_dim), (feature_dim, _DIM)).astype(np.float32)
    return MockBundle(
        n_classes=n_classes,
        W=rng.normal(0, 0.01, (_DIM, n_classes)).astype(np.float32),
        b=np.zeros(n_classes, dtype=np.float32),
        proj=proj,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )


def get_mock_parameters(bundle: MockBundle) -> list[np.ndarray]:
    return [bundle.W.copy(), bundle.b.copy()]


def set_mock_parameters(bundle: MockBundle, params: list[np.ndarray]) -> None:
    bundle.W = np.asarray(params[0], dtype=np.float32).copy()
    bundle.b = np.asarray(params[1], dtype=np.float32).copy()


def _features(bundle: MockBundle, records: list[Record]) -> np.ndarray:
    X = np.stack([r.features for r in records]).astype(np.float32)
    if bundle.scaler_mean is not None:
        X = (X - bundle.scaler_mean) / (bundle.scaler_std + 1e-6)
    return X @ bundle.proj


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def mock_train(bundle: MockBundle, records: list[Record], config: dict, seed: int):
    if not records:
        return
    lr = float(config.get("train", {}).get("learning_rate", 1e-3)) * 50
    epochs = int(config.get("train", {}).get("local_epochs", 3))
    rng = np.random.default_rng(seed)
    X = _features(bundle, records)
    y = np.array([r.label for r in records], dtype=np.int64)
    n = len(records)
    for _ in range(max(epochs, 3)):
        idx = rng.permutation(n)
        for start in range(0, n, 16):
            batch = idx[start : start + 16]
            logits = X[batch] @ bundle.W + bundle.b
            probs = _softmax(logits)
            onehot = np.zeros_like(probs)
            onehot[np.arange(len(batch)), y[batch]] = 1.0
            grad = probs - onehot
            gW = X[batch].T @ grad / len(batch)
            gb = grad.mean(axis=0)
            bundle.W -= lr * gW
            bundle.b -= lr * gb


def mock_evaluate(bundle: MockBundle, records: list[Record], config: dict) -> dict[str, float]:
    if not records:
        return {
            "n": 0, "eval_loss": float("nan"), "accuracy": float("nan"),
            "precision_macro": float("nan"), "recall_macro": float("nan"), "f1_macro": float("nan"),
        }
    X = _features(bundle, records)
    y = np.array([r.label for r in records], dtype=np.int64)
    logits = X @ bundle.W + bundle.b
    probs = _softmax(logits)
    eps = 1e-9
    loss = float(-np.mean(np.log(probs[np.arange(len(y)), y] + eps)))
    preds = probs.argmax(axis=1)

    classes = np.unique(np.concatenate([y, preds]))
    precisions, recalls, f1s = [], [], []
    for c in classes:
        tp = int(np.sum((preds == c) & (y == c)))
        fp = int(np.sum((preds == c) & (y != c)))
        fn = int(np.sum((preds != c) & (y == c)))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return {
        "n": len(records),
        "eval_loss": loss,
        "accuracy": float(np.mean(preds == y)),
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
    }
