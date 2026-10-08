from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np

from src.policy.scope import PolicyScope, parse_scope


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_update_id(agency_id: str, policy_scope: PolicyScope, round_id: int) -> str:
    scope_tag = "global" if parse_scope(policy_scope) is PolicyScope.GLOBAL else "regional"
    return f"{agency_id}_{scope_tag}_r{round_id}"


@dataclass
class Update:
    parameters: list[np.ndarray]
    agency_id: str
    region_id: str
    round_id: int
    policy_scope: PolicyScope
    num_examples: int
    update_id: str = ""
    destination: str = "REGIONAL"
    parent_version: Optional[str] = None
    timestamp: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        self.policy_scope = parse_scope(self.policy_scope)
        self.parameters = [np.asarray(p) for p in self.parameters]
        if not self.update_id:
            self.update_id = make_update_id(
                self.agency_id, self.policy_scope, self.round_id
            )

    @property
    def scope_tag(self) -> str:
        return "global" if self.policy_scope is PolicyScope.GLOBAL else "regional"

    def param_hash(self) -> str:
        h = hashlib.sha256()
        for p in self.parameters:
            h.update(np.ascontiguousarray(p, dtype=np.float32).tobytes())
        return h.hexdigest()[:12]

    def metadata_record(self) -> dict:
        return {
            "update_id": self.update_id,
            "agency_id": self.agency_id,
            "region_id": self.region_id,
            "round_id": self.round_id,
            "policy_scope": self.policy_scope.value,
            "num_examples": self.num_examples,
            "destination": self.destination,
            "parent_version": self.parent_version,
            "param_hash": self.param_hash(),
            "timestamp": self.timestamp,
        }


def updates_from_flower(
    results: Sequence[tuple[list[np.ndarray], int, dict]],
) -> list[Update]:
    updates: list[Update] = []
    for params, num_examples, metrics in results:
        updates.append(
            Update(
                parameters=list(params),
                agency_id=str(metrics["agency_id"]),
                region_id=str(metrics["region_id"]),
                round_id=int(metrics["round_id"]),
                policy_scope=parse_scope(metrics["policy_scope"]),
                num_examples=int(num_examples),
                parent_version=metrics.get("parent_version"),
            )
        )
    return updates
