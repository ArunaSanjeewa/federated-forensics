from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from src.fl.aggregation import HierarchicalResult, fedavg
from src.policy.scope import PolicyScope
from src.update.metadata import Update

Params = list[np.ndarray]


@dataclass
class DeliveredModel:
    agency_id: str
    region_id: str
    parameters: Params
    path: list[str]
    components: list[str]


def build_return_path(
    result: HierarchicalResult,
    updates: Sequence[Update],
    agency_regions: dict[str, str],
    logger: Optional["object"] = None,
    round_id: int = 0,
    blend_weighting: str = "equal",
) -> dict[str, DeliveredModel]:
    if result.national_model is None:
        raise ValueError("no national model was produced; nothing to return")
    if blend_weighting not in ("equal", "sample"):
        raise ValueError(f"blend_weighting must be 'equal' or 'sample', got {blend_weighting!r}")

    regional_only: dict[str, tuple[Params, float]] = {}
    by_region_scope: dict[tuple[str, PolicyScope], list[Update]] = defaultdict(list)
    for u in updates:
        by_region_scope[(u.region_id, u.policy_scope)].append(u)
    for (region_id, scope), group in by_region_scope.items():
        if scope is PolicyScope.REGIONAL_ONLY and group:
            regional_only[region_id] = (
                fedavg([(u.parameters, u.num_examples) for u in group]),
                float(sum(u.num_examples for u in group)),
            )

    national_sample_weight = float(
        sum(
            u.num_examples
            for u in updates
            if u.policy_scope is PolicyScope.GLOBAL
        )
    ) or 1.0

    delivered: dict[str, DeliveredModel] = {}
    for agency_id, region_id in agency_regions.items():
        if region_id in regional_only:
            ro_model, ro_sample_weight = regional_only[region_id]
            if blend_weighting == "equal":
                national_w, ro_w = 0.5, 0.5
            else:
                national_w, ro_w = national_sample_weight, ro_sample_weight
            params = fedavg(
                [
                    (result.national_model, national_w),
                    (ro_model, ro_w),
                ]
            )
            components = ["national_model", f"{region_id}_regional_only"]
        else:
            params = [np.array(p, copy=True) for p in result.national_model]
            components = ["national_model"]

        model = DeliveredModel(
            agency_id=agency_id,
            region_id=region_id,
            parameters=params,
            path=["NATIONAL", region_id, agency_id],
            components=components,
        )
        delivered[agency_id] = model

        if logger is not None:
            logger.log(
                {
                    "round_id": round_id,
                    "update_id": f"return_{agency_id}_r{round_id}",
                    "agency_id": agency_id,
                    "region_id": region_id,
                    "policy_scope": "GLOBAL",
                    "source": region_id,
                    "destination": agency_id,
                    "decision": "INCLUDE",
                    "stage": "national_to_regional_to_local",
                    "components": components,
                }
            )

    _assert_no_direct_national_to_local(delivered)
    return delivered


def _assert_no_direct_national_to_local(delivered: dict[str, DeliveredModel]) -> None:
    for model in delivered.values():
        if model.path[0] != "NATIONAL" or model.path[1] == model.agency_id:
            raise AssertionError(
                f"agency {model.agency_id} received a model without passing "
                f"through a regional coordinator: path={model.path}"
            )
        if len(model.path) != 3 or not model.path[1].startswith("R"):
            raise AssertionError(
                f"return path for {model.agency_id} is malformed: {model.path}"
            )
