from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from src.dataset.records import Record
from src.policy.scope import PolicyScope

REGION_OF_AGENCY = {"A1": "R1", "A2": "R1", "A3": "R2", "A4": "R2"}
RESTRICTED_AGENCIES = ("A1", "A2")
ALL_AGENCIES = ("A1", "A2", "A3", "A4")


@dataclass
class PartitionConfig:
    data_seed: int = 42
    agency_budget_per_class: int = 200
    restricted_agencies: tuple[str, ...] = RESTRICTED_AGENCIES
    all_agencies: tuple[str, ...] = ALL_AGENCIES


def partition_records(records: list[Record], cfg: PartitionConfig) -> list[Record]:
    by_family: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_family[r.family].append(r)

    assigned: list[Record] = []
    for family, pool in sorted(by_family.items()):
        scope = pool[0].scope
        eligible = (
            cfg.restricted_agencies if scope is PolicyScope.REGIONAL_ONLY else cfg.all_agencies
        )
        rng = random.Random(f"{cfg.data_seed}:{family}")
        pool = list(pool)
        rng.shuffle(pool)

        needed = cfg.agency_budget_per_class * len(eligible)
        if len(pool) < needed:
            raise ValueError(
                f"family {family!r} has only {len(pool)} records, needs "
                f"{needed} ({cfg.agency_budget_per_class} x {len(eligible)} agencies)"
            )

        offset = 0
        for agency in eligible:
            block = pool[offset : offset + cfg.agency_budget_per_class]
            offset += cfg.agency_budget_per_class
            for r in block:
                r.agency_id = agency
                r.region_id = REGION_OF_AGENCY[agency]
                assigned.append(r)

    return assigned


def validate_partition(records: list[Record]) -> None:
    holdings: dict[str, set[str]] = {a: set() for a in ALL_AGENCIES}
    for r in records:
        if r.agency_id is None:
            raise AssertionError(f"record {r.record_id} was not assigned to an agency")
        if REGION_OF_AGENCY[r.agency_id] != r.region_id:
            raise AssertionError(f"record {r.record_id} has inconsistent agency/region")
        holdings[r.agency_id].add(r.scope.value)

    for agency in ("A1", "A2"):
        if holdings[agency] != {"GLOBAL", "REGIONAL_ONLY"}:
            raise AssertionError(
                f"{agency} must hold GLOBAL + REGIONAL_ONLY, holds {holdings[agency]}"
            )
    for agency in ("A3", "A4"):
        if holdings[agency] != {"GLOBAL"}:
            raise AssertionError(
                f"{agency} must hold GLOBAL only, holds {holdings[agency]}"
            )
