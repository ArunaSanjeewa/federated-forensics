from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from src.policy.gate import Destination, GateDecision, GateEvaluation, PolicyGate
from src.policy.scope import PolicyScope
from src.update.metadata import Update


Params = list[np.ndarray]


def fedavg(items: Sequence[tuple[Params, float]]) -> Params:
    items = [(p, float(w)) for p, w in items if w > 0]
    if not items:
        raise ValueError("fedavg received no contributions with positive weight")
    total = sum(w for _, w in items)
    n_tensors = len(items[0][0])
    for params, _ in items:
        if len(params) != n_tensors:
            raise ValueError("parameter lists have inconsistent length")
    out: Params = []
    for t in range(n_tensors):
        acc = np.zeros_like(np.asarray(items[0][0][t], dtype=np.float64))
        for params, w in items:
            acc += np.asarray(params[t], dtype=np.float64) * w
        out.append((acc / total).astype(np.asarray(items[0][0][t]).dtype))
    return out


@dataclass
class RegionAggregate:
    region_id: str
    regional_model: Params
    outward_model: Optional[Params]
    outward_weight: float
    included_update_ids: list[str]
    excluded_update_ids: list[str]


@dataclass
class HierarchicalResult:
    regional: dict[str, RegionAggregate]
    national_model: Optional[Params]
    national_accepted_update_ids: list[str]
    gate_evaluations: list[GateEvaluation]

    @property
    def policy_violations(self) -> int:
        return sum(
            1
            for e in self.gate_evaluations
            if e.stage == "regional_to_national"
            and e.policy_scope is PolicyScope.REGIONAL_ONLY
            and e.decision is GateDecision.INCLUDE
        )


def hierarchical_aggregate(
    updates: Sequence[Update],
    round_id: int,
    gate: Optional[PolicyGate] = None,
) -> HierarchicalResult:
    gate = gate or PolicyGate()

    by_region: dict[str, list[Update]] = defaultdict(list)
    for u in updates:
        by_region[u.region_id].append(u)

    regional: dict[str, RegionAggregate] = {}
    outward_for_national: list[tuple[Params, float, str]] = []

    for region_id, region_updates in sorted(by_region.items()):
        for u in region_updates:
            gate.evaluate(
                update_id=u.update_id,
                agency_id=u.agency_id,
                region_id=u.region_id,
                round_id=round_id,
                policy_scope=u.policy_scope,
                source=u.agency_id,
                destination=Destination.REGIONAL,
                stage="agency_to_regional",
            )
        regional_model = fedavg([(u.parameters, u.num_examples) for u in region_updates])

        global_updates: list[Update] = []
        included_ids: list[str] = []
        excluded_ids: list[str] = []
        for u in region_updates:
            ev = gate.evaluate(
                update_id=u.update_id,
                agency_id=u.agency_id,
                region_id=u.region_id,
                round_id=round_id,
                policy_scope=u.policy_scope,
                source=u.region_id,
                destination=Destination.NATIONAL,
                stage="regional_to_national",
            )
            if ev.decision is GateDecision.INCLUDE:
                global_updates.append(u)
                included_ids.append(u.update_id)
            else:
                excluded_ids.append(u.update_id)

        outward_model: Optional[Params] = None
        outward_weight = 0.0
        if global_updates:
            outward_model = fedavg(
                [(u.parameters, u.num_examples) for u in global_updates]
            )
            outward_weight = float(sum(u.num_examples for u in global_updates))
            outward_for_national.append((outward_model, outward_weight, region_id))

        regional[region_id] = RegionAggregate(
            region_id=region_id,
            regional_model=regional_model,
            outward_model=outward_model,
            outward_weight=outward_weight,
            included_update_ids=included_ids,
            excluded_update_ids=excluded_ids,
        )

    national_model: Optional[Params] = None
    national_accepted: list[str] = []
    if outward_for_national:
        national_model = fedavg(
            [(p, w) for p, w, _ in outward_for_national]
        )
        for agg in regional.values():
            national_accepted.extend(agg.included_update_ids)

    return HierarchicalResult(
        regional=regional,
        national_model=national_model,
        national_accepted_update_ids=sorted(national_accepted),
        gate_evaluations=list(gate.evaluations),
    )
