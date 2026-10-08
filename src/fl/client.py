from __future__ import annotations

from typing import Any, Optional

import numpy as np

from src.dataset.records import Record
from src.model import backend
from src.policy.scope import PolicyScope, parse_scope


class AgencyClient:

    def __init__(
        self,
        *,
        client_id: str,
        agency_id: str,
        region_id: str,
        policy_scope: PolicyScope | str,
        bundle,
        train_records: list[Record],
        config: dict,
        seed: int,
    ) -> None:
        self.client_id = client_id
        self.agency_id = agency_id
        self.region_id = region_id
        self.policy_scope = parse_scope(policy_scope)
        self.bundle = bundle
        self.train_records = train_records
        self.config = config
        self.seed = seed
        self._round = 0

    def get_parameters(self, config) -> list[np.ndarray]:
        return backend.get_parameters(self.config, self.bundle)

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        backend.set_parameters(self.config, self.bundle, parameters)

    def fit(self, parameters, config) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        self.set_parameters(parameters)
        self._round = int(config.get("round_id", self._round + 1))
        stats = backend.train(
            self.config, self.bundle, self.train_records, self.seed + self._round
        )
        metrics = {
            "client_id": self.client_id,
            "agency_id": self.agency_id,
            "region_id": self.region_id,
            "round_id": self._round,
            "policy_scope": self.policy_scope.value,
            "final_loss": getattr(stats, "final_loss", float("nan")),
            "steps": getattr(stats, "steps", 0),
        }
        n = max(getattr(stats, "n_examples", len(self.train_records)), 1)
        return self.get_parameters({}), n, metrics

    def evaluate(self, parameters, config) -> tuple[float, int, dict[str, Any]]:
        self.set_parameters(parameters)
        return 0.0, max(len(self.train_records), 1), {}

    def to_client(self):
        import flwr as fl

        outer = self

        class _NP(fl.client.NumPyClient):
            def get_parameters(self, config):
                return outer.get_parameters(config)

            def fit(self, parameters, config):
                return outer.fit(parameters, config)

            def evaluate(self, parameters, config):
                return outer.evaluate(parameters, config)

        return _NP().to_client()
