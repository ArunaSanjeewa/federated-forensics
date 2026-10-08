from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

FEATURE_DIM = 2381


def select_device(prefer: str = "auto") -> str:
    import torch

    if prefer != "auto":
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class ModelBundle:
    model: Any
    device: str
    param_keys: list[str] = field(default_factory=list)
    n_classes: int = 0
    scaler_mean: Optional[np.ndarray] = None
    scaler_std: Optional[np.ndarray] = None


def _build_mlp(input_dim: int, hidden_dims: list[int], n_classes: int, dropout: float):
    import torch.nn as nn

    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    return nn.Sequential(*layers)


def build_model_bundle(
    config: dict,
    seed: int = 42,
    n_classes: int = 0,
    scaler_mean: Optional[np.ndarray] = None,
    scaler_std: Optional[np.ndarray] = None,
) -> ModelBundle:
    import torch

    torch.manual_seed(seed)
    mc = config.get("model", {})
    device = select_device(mc.get("device", "auto"))
    hidden_dims = list(mc.get("hidden_dims", [256, 64]))
    dropout = float(mc.get("dropout", 0.1))

    model = _build_mlp(FEATURE_DIM, hidden_dims, n_classes, dropout)
    model.to(device)

    param_keys = [n for n, _ in model.named_parameters()]
    return ModelBundle(
        model=model,
        device=device,
        param_keys=param_keys,
        n_classes=n_classes,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )


def get_adapter_parameters(bundle: ModelBundle) -> list[np.ndarray]:
    state = dict(bundle.model.named_parameters())
    return [state[k].detach().to("cpu").numpy().astype(np.float32) for k in bundle.param_keys]


def set_adapter_parameters(bundle: ModelBundle, params: list[np.ndarray]) -> None:
    import torch

    if len(params) != len(bundle.param_keys):
        raise ValueError(f"expected {len(bundle.param_keys)} tensors, got {len(params)}")
    state = dict(bundle.model.named_parameters())
    with torch.no_grad():
        for key, value in zip(bundle.param_keys, params):
            tensor = torch.as_tensor(np.asarray(value), dtype=state[key].dtype, device=state[key].device)
            if tuple(tensor.shape) != tuple(state[key].shape):
                raise ValueError(f"shape mismatch for {key}: {tuple(tensor.shape)} vs {tuple(state[key].shape)}")
            state[key].copy_(tensor)


def count_trainable(bundle: ModelBundle) -> int:
    return int(sum(p.numel() for p in bundle.model.parameters() if p.requires_grad))
