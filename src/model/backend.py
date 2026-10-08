from __future__ import annotations

from typing import Any, Optional

import numpy as np

from src.dataset.records import Record


def _backend(config: dict) -> str:
    return str(config.get("model", {}).get("backend", "hf")).lower()


def build_bundle(
    config: dict,
    seed: int,
    n_classes: int = 0,
    scaler_mean: Optional[np.ndarray] = None,
    scaler_std: Optional[np.ndarray] = None,
) -> Any:
    if _backend(config) == "mock":
        from src.model.mock import build_mock_bundle

        return build_mock_bundle(
            config, seed, n_classes=n_classes, scaler_mean=scaler_mean, scaler_std=scaler_std
        )
    from src.model.model import build_model_bundle

    return build_model_bundle(
        config, seed=seed, n_classes=n_classes, scaler_mean=scaler_mean, scaler_std=scaler_std
    )


def get_parameters(config: dict, bundle) -> list[np.ndarray]:
    if _backend(config) == "mock":
        from src.model.mock import get_mock_parameters

        return get_mock_parameters(bundle)
    from src.model.model import get_adapter_parameters

    return get_adapter_parameters(bundle)


def set_parameters(config: dict, bundle, params: list[np.ndarray]) -> None:
    if _backend(config) == "mock":
        from src.model.mock import set_mock_parameters

        set_mock_parameters(bundle, params)
        return
    from src.model.model import set_adapter_parameters

    set_adapter_parameters(bundle, params)


def train(config: dict, bundle, records: list[Record], seed: int):
    if _backend(config) == "mock":
        from src.model.mock import mock_train

        return mock_train(bundle, records, config, seed)
    from src.model.training import local_train

    return local_train(bundle, records, config, seed)


def evaluate(config: dict, bundle, records: list[Record]) -> dict[str, float]:
    if _backend(config) == "mock":
        from src.model.mock import mock_evaluate

        return mock_evaluate(bundle, records, config)
    from src.model.training import evaluate_model

    return evaluate_model(bundle, records, config)


def count_trainable(config: dict, bundle) -> int:
    if _backend(config) == "mock":
        return int(bundle.W.size + bundle.b.size)
    from src.model.model import count_trainable as _ct

    return _ct(bundle)
