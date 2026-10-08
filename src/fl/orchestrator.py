from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from src.fl.aggregation import (
    HierarchicalResult,
    fedavg,
    hierarchical_aggregate,
)
from src.fl.client import AgencyClient
from src.fl.return_path import build_return_path
from src.policy.gate import Destination, PolicyGate
from src.policy.scope import PolicyScope
from src.update.metadata import Update

Params = list[np.ndarray]


@dataclass
class RoundClientResult:
    client_id: str
    agency_id: str
    region_id: str
    policy_scope: PolicyScope
    parameters: Params
    num_examples: int
    round_id: int


def _fit_clients(
    clients: list[AgencyClient],
    parameters_for: Callable[[AgencyClient], Params],
    round_id: int,
) -> list[RoundClientResult]:
    results: list[RoundClientResult] = []
    for c in clients:
        params, n, metrics = c.fit(parameters_for(c), {"round_id": round_id})
        results.append(
            RoundClientResult(
                client_id=metrics["client_id"],
                agency_id=metrics["agency_id"],
                region_id=metrics["region_id"],
                policy_scope=PolicyScope(metrics["policy_scope"]),
                parameters=params,
                num_examples=n,
                round_id=round_id,
            )
        )
    return results


@dataclass
class FlatOutcome:
    global_model: Params
    per_round_accepted: dict[int, list[str]] = field(default_factory=dict)
    agency_params: dict[str, Params] = field(default_factory=dict)


def run_flat(
    clients: list[AgencyClient],
    init_parameters: Params,
    num_rounds: int,
    logger: Optional[object] = None,
) -> FlatOutcome:
    global_model = [np.array(p, copy=True) for p in init_parameters]
    outcome = FlatOutcome(global_model=global_model)

    for rnd in range(1, num_rounds + 1):
        results = _fit_clients(clients, lambda c: global_model, rnd)
        global_model = fedavg(
            [(r.parameters, r.num_examples) for r in results]
        )
        outcome.per_round_accepted[rnd] = sorted(
            Update(
                parameters=r.parameters,
                agency_id=r.agency_id,
                region_id=r.region_id,
                round_id=rnd,
                policy_scope=r.policy_scope,
                num_examples=r.num_examples,
            ).update_id
            for r in results
        )
        if logger is not None:
            for r in results:
                logger.log(
                    {
                        "round_id": rnd,
                        "update_id": f"{r.agency_id}_{('global' if r.policy_scope is PolicyScope.GLOBAL else 'regional')}_r{rnd}",
                        "agency_id": r.agency_id,
                        "region_id": r.region_id,
                        "policy_scope": r.policy_scope.value,
                        "source": r.agency_id,
                        "destination": "NATIONAL",
                        "decision": "INCLUDE",
                        "stage": "flat_national",
                    }
                )

    outcome.global_model = global_model
    for c in clients:
        outcome.agency_params[c.agency_id] = [np.array(p, copy=True) for p in global_model]
    return outcome


@dataclass
class HierarchicalOutcome:
    national_model: Params
    regional_models: dict[str, Params]
    agency_params: dict[str, Params]
    per_round_results: list[HierarchicalResult] = field(default_factory=list)
    per_round_delivered: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_policy_violations(self) -> int:
        return sum(r.policy_violations for r in self.per_round_results)


def run_hierarchical(
    clients: list[AgencyClient],
    init_parameters: Params,
    num_rounds: int,
    agency_regions: dict[str, str],
    logger: Optional[object] = None,
    blend_weighting: str = "equal",
) -> HierarchicalOutcome:
    national_model = [np.array(p, copy=True) for p in init_parameters]
    agency_start: dict[str, Params] = {
        a: [np.array(p, copy=True) for p in init_parameters] for a in agency_regions
    }
    regional_models: dict[str, Params] = {}
    outcome = HierarchicalOutcome(
        national_model=national_model, regional_models={}, agency_params={}
    )

    for rnd in range(1, num_rounds + 1):
        results = _fit_clients(
            clients, lambda c: agency_start[c.agency_id], rnd
        )
        updates = [
            Update(
                parameters=r.parameters,
                agency_id=r.agency_id,
                region_id=r.region_id,
                round_id=rnd,
                policy_scope=r.policy_scope,
                num_examples=r.num_examples,
            )
            for r in results
        ]

        gate = PolicyGate(logger=logger)
        result = hierarchical_aggregate(updates, round_id=rnd, gate=gate)
        outcome.per_round_results.append(result)

        national_model = result.national_model
        regional_models = {
            rid: agg.regional_model for rid, agg in result.regional.items()
        }

        delivered = build_return_path(
            result, updates, agency_regions, logger=logger, round_id=rnd,
            blend_weighting=blend_weighting,
        )
        outcome.per_round_delivered.append(
            {a: dm.components for a, dm in delivered.items()}
        )
        agency_start = {
            a: [np.array(p, copy=True) for p in dm.parameters]
            for a, dm in delivered.items()
        }

    outcome.national_model = national_model
    outcome.regional_models = regional_models
    outcome.agency_params = {a: p for a, p in agency_start.items()}
    return outcome
