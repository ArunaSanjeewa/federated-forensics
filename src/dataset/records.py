from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.policy.scope import PolicyScope, parse_scope


@dataclass
class Record:
    record_id: str
    features: np.ndarray
    family: str
    category: str
    label: int
    scope: PolicyScope
    agency_id: Optional[str] = None
    region_id: Optional[str] = None
    split: Optional[str] = None

    def __post_init__(self) -> None:
        self.scope = parse_scope(self.scope)
        self.features = np.asarray(self.features, dtype=np.float32)

    def as_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "family": self.family,
            "category": self.category,
            "label": self.label,
            "scope": self.scope.value,
            "agency_id": self.agency_id,
            "region_id": self.region_id,
            "split": self.split,
            "feature_dim": int(self.features.shape[0]),
        }


def summarise(records: list[Record]) -> dict:
    out: dict[str, int] = {}

    def bump(key: str) -> None:
        out[key] = out.get(key, 0) + 1

    for r in records:
        bump("total")
        bump(f"scope:{r.scope.value}")
        bump(f"family:{r.family}")
        if r.agency_id:
            bump(f"agency:{r.agency_id}")
            bump(f"agency:{r.agency_id}:{r.scope.value}")
            bump(f"agency:{r.agency_id}:family:{r.family}")
        if r.region_id:
            bump(f"region:{r.region_id}")
        if r.split:
            bump(f"split:{r.split}")
            if r.agency_id:
                bump(f"agency:{r.agency_id}:split:{r.split}")
                bump(f"agency:{r.agency_id}:{r.scope.value}:{r.split}")
                bump(f"agency:{r.agency_id}:family:{r.family}:{r.split}")
    return out
