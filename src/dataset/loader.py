from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.config import REPO_ROOT
from src.dataset.policy_assign import assign_scopes
from src.dataset.records import Record

RAW_DIR = REPO_ROOT / "data" / "bodmas_raw"
FEATURE_DIM = 2381


def load_bodmas_arrays(raw_dir: Path = RAW_DIR):
    import pandas as pd

    npz_path = raw_dir / "bodmas.npz"
    meta_path = raw_dir / "bodmas_metadata.csv"
    cat_path = raw_dir / "bodmas_malware_category.csv"
    for p in (npz_path, meta_path, cat_path):
        if not p.exists():
            raise FileNotFoundError(
                f"missing {p}; run `python scripts/prepare_data.py` first"
            )

    data = np.load(npz_path)
    X, y = data["X"], data["y"]

    meta = pd.read_csv(meta_path)
    cat = pd.read_csv(cat_path)
    mal_idx = meta.index[meta["family"].notna()].to_numpy()
    mal = meta.loc[mal_idx].reset_index(drop=True)
    mal["_row"] = mal_idx

    merged = mal.merge(cat, left_on="sha", right_on="sha256", how="left")
    return X, y, merged


def build_records(
    X: np.ndarray,
    merged_meta,
    global_families: list[str],
    restricted_families: list[str],
) -> tuple[list[Record], "PolicyAssignmentReport"]:
    chosen = list(global_families) + list(restricted_families)
    sel = merged_meta[merged_meta["family"].isin(chosen)].reset_index(drop=True)
    if sel.empty:
        raise ValueError(
            f"none of the configured families {chosen} were found in the dataset"
        )

    scopes, report = assign_scopes(sel["family"].tolist(), restricted_families)
    label_of = {f: i for i, f in enumerate(chosen)}

    records: list[Record] = []
    for i, row in sel.iterrows():
        row_idx = int(row["_row"])
        records.append(
            Record(
                record_id=str(row["sha"])[:16],
                features=X[row_idx],
                family=str(row["family"]),
                category=str(row.get("category", "")),
                label=label_of[str(row["family"])],
                scope=scopes[i],
            )
        )
    return records, report


def load_corpus(
    global_families: list[str],
    restricted_families: list[str],
    raw_dir: Path = RAW_DIR,
):
    X, y, merged = load_bodmas_arrays(raw_dir)
    return build_records(X, merged, global_families, restricted_families)


def synthetic_corpus(
    global_families: Optional[list[str]] = None,
    restricted_families: Optional[list[str]] = None,
    n_per_family: int = 1000,
    dim: int = FEATURE_DIM,
    seed: int = 0,
) -> list[Record]:
    global_families = global_families or ["fam_g1", "fam_g2", "fam_g3"]
    restricted_families = restricted_families or ["fam_r1"]
    chosen = global_families + restricted_families
    label_of = {f: i for i, f in enumerate(chosen)}

    rng = np.random.default_rng(seed)
    records: list[Record] = []
    scope_of = {f: ("REGIONAL_ONLY" if f in restricted_families else "GLOBAL") for f in chosen}
    for fam in chosen:
        centre = rng.normal(0, 1, dim).astype(np.float32)
        for i in range(n_per_family):
            vec = centre + rng.normal(0, 0.5, dim).astype(np.float32)
            records.append(
                Record(
                    record_id=f"{fam}_{i:04d}",
                    features=vec,
                    family=fam,
                    category="synthetic",
                    label=label_of[fam],
                    scope=scope_of[fam],
                )
            )
    return records
