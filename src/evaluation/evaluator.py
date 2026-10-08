from __future__ import annotations

from typing import Any

import numpy as np

from src.dataset.records import Record
from src.dataset.split import restricted_test_records
from src.model import backend

REGION_OF_AGENCY = {"A1": "R1", "A2": "R1", "A3": "R2", "A4": "R2"}


def evaluate_all_agencies(
    bundle,
    agency_params: dict[str, list[np.ndarray]],
    records: list[Record],
    config: dict,
    *,
    condition: str,
    seed: int,
) -> list[dict[str, Any]]:
    restricted_pool = restricted_test_records(records)
    rows: list[dict[str, Any]] = []

    for agency_id, params in sorted(agency_params.items()):
        backend.set_parameters(config, bundle, params)

        own_test = [r for r in records if r.split == "test" and r.agency_id == agency_id]
        own = backend.evaluate(config, bundle, own_test)
        rows.append(_row(condition, seed, agency_id, "all", own))

        restricted = backend.evaluate(config, bundle, restricted_pool)
        rows.append(_row(condition, seed, agency_id, "restricted", restricted))

    return rows


def _row(condition, seed, agency_id, split, metrics: dict) -> dict:
    return {
        "condition": condition,
        "seed": seed,
        "agency_id": agency_id,
        "region_id": REGION_OF_AGENCY[agency_id],
        "test_split": split,
        "n": metrics.get("n", 0),
        "eval_loss": metrics.get("eval_loss", float("nan")),
        "accuracy": metrics.get("accuracy", float("nan")),
        "precision_macro": metrics.get("precision_macro", float("nan")),
        "recall_macro": metrics.get("recall_macro", float("nan")),
        "f1_macro": metrics.get("f1_macro", float("nan")),
    }
