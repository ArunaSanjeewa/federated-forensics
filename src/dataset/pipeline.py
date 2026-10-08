from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

import numpy as np

from src.dataset.loader import load_corpus, synthetic_corpus
from src.dataset.partition import PartitionConfig, partition_records, validate_partition
from src.dataset.records import Record, summarise
from src.dataset.split import SplitConfig, restricted_test_records, split_records


def _partition_cfg(config: dict) -> PartitionConfig:
    d = config.get("data", {})
    return PartitionConfig(
        data_seed=int(d.get("data_seed", 42)),
        agency_budget_per_class=int(d.get("agency_budget_per_class", 200)),
    )


def _split_cfg(config: dict) -> SplitConfig:
    d = config.get("data", {})
    return SplitConfig(
        train_per_class=int(d.get("train_per_class", 150)),
        test_per_class=int(d.get("test_per_class", 50)),
        split_per_seed=bool(d.get("split_per_seed", True)),
        data_seed=int(d.get("data_seed", 42)),
    )


def fit_scaler(records: list[Record]) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([r.features for r in records]).astype(np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def prepare_corpus(
    config: dict, seed: int, use_synthetic: bool = False
) -> tuple[list[Record], dict[str, Any]]:
    data_cfg = config.get("data", {})
    global_families = list(data_cfg.get("global_families", []))
    restricted_families = list(data_cfg.get("restricted_families", []))

    if use_synthetic:
        records = synthetic_corpus(
            global_families=global_families or None,
            restricted_families=restricted_families or None,
            seed=int(data_cfg.get("data_seed", 42)),
        )
        chosen = [r.family for r in records]
        from src.dataset.policy_assign import assign_scopes

        _, policy_report = assign_scopes(chosen, restricted_families or ["fam_r1"])
    else:
        records, policy_report = load_corpus(global_families, restricted_families)

    part_cfg = _partition_cfg(config)
    records = partition_records(records, part_cfg)
    validate_partition(records)

    split_cfg = _split_cfg(config)
    split_report = split_records(records, split_cfg, seed=seed)

    train_records = [r for r in records if r.split == "train"]
    scaler_mean, scaler_std = fit_scaler(train_records) if train_records else (None, None)

    report = {
        "seed": seed,
        "n_records": len(records),
        "global_families": global_families,
        "restricted_families": restricted_families,
        "policy_assignment": policy_report.as_dict(),
        "partition": {
            "data_seed": part_cfg.data_seed,
            "agency_budget_per_class": part_cfg.agency_budget_per_class,
            "counts": summarise(records),
        },
        "split": split_report.as_dict(),
        "restricted_test_pool_size": len(restricted_test_records(records)),
        "n_classes": len(global_families) + len(restricted_families),
    }
    return records, report, (scaler_mean, scaler_std)


def agency_train_records(records: list[Record], agency_id: Optional[str] = None) -> list[Record]:
    return [
        r
        for r in records
        if r.split == "train" and (agency_id is None or r.agency_id == agency_id)
    ]


def agency_test_records(records: list[Record], agency_id: Optional[str] = None) -> list[Record]:
    return [
        r
        for r in records
        if r.split == "test" and (agency_id is None or r.agency_id == agency_id)
    ]
