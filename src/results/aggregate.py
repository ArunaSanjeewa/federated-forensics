from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.config import REPO_ROOT

RAW = REPO_ROOT / "results" / "raw"
OUT = REPO_ROOT / "results"
METRICS = ["eval_loss", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
CONDITIONS = ["B1", "B2", "B3", "B4"]


def _load_raw_metrics() -> pd.DataFrame:
    frames = []
    for csv_path in sorted(RAW.glob("*/seed_*/final_metrics.csv")):
        if csv_path.stat().st_size == 0:
            continue
        frames.append(pd.read_csv(csv_path))
    if not frames:
        raise FileNotFoundError(f"no final_metrics.csv found under {RAW}")
    return pd.concat(frames, ignore_index=True)


def _load_policy() -> pd.DataFrame:
    rows = []
    for jpath in sorted(RAW.glob("*/seed_*/policy_enforcement.json")):
        data = json.loads(jpath.read_text(encoding="utf-8"))
        cond = jpath.parts[-3]
        seed = int(jpath.parts[-2].split("_")[1])
        rows.append(
            {
                "condition": cond,
                "seed": seed,
                "regional_only_forward_attempts": data.get("regional_only_forward_attempts", 0),
                "regional_only_blocked_nationally": data.get("regional_only_blocked_nationally", 0),
                "global_updates_forwarded_nationally": data.get("global_updates_forwarded_nationally", 0),
                "policy_violations": data.get("policy_violations", 0),
                "policy_enforcement_rate": data.get("policy_enforcement_rate", float("nan")),
            }
        )
    return pd.DataFrame(rows)


def aggregate() -> dict[str, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = _load_raw_metrics()

    long = raw.melt(
        id_vars=["condition", "seed", "agency_id", "region_id", "test_split", "n"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )
    agg = (
        long.groupby(["condition", "agency_id", "region_id", "test_split", "metric"])
        .agg(mean=("value", "mean"), std=("value", "std"), n_seeds=("value", "count"))
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)
    agg_path = OUT / "aggregated_results.csv"
    agg.to_csv(agg_path, index=False)

    wide_rows = []
    for (cond, agency, region, split), grp in long.groupby(
        ["condition", "agency_id", "region_id", "test_split"]
    ):
        row = {"condition": cond, "agency_id": agency, "region_id": region, "test_split": split}
        for metric in ("f1_macro", "accuracy", "eval_loss"):
            vals = grp[grp["metric"] == metric]["value"]
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        wide_rows.append(row)
    per_agency = pd.DataFrame(wide_rows).sort_values(["agency_id", "test_split", "condition"])
    per_agency_path = OUT / "per_agency_results.csv"
    per_agency.to_csv(per_agency_path, index=False)

    policy = _load_policy()
    policy_path = OUT / "policy_enforcement.csv"
    policy.to_csv(policy_path, index=False)

    primary_path = _comparison(long, "B4", "B3", OUT / "primary_comparison.csv", restrict_agencies=("A1", "A2"))
    secondary_path = _comparison(
        long, "B2", "B4", OUT / "secondary_comparison.csv", restrict_agencies=("A1", "A2", "A3", "A4")
    )

    return {
        "aggregated_results": agg_path,
        "per_agency_results": per_agency_path,
        "policy_enforcement": policy_path,
        "primary_comparison": primary_path,
        "secondary_comparison": secondary_path,
    }


def _comparison(long: pd.DataFrame, cond_a: str, cond_b: str, out_path: Path, restrict_agencies) -> Path:
    rows = []
    sub = long[long["agency_id"].isin(restrict_agencies)]
    for (agency, region, split, metric), grp in sub.groupby(["agency_id", "region_id", "test_split", "metric"]):
        a = grp[grp["condition"] == cond_a].set_index("seed")["value"]
        b = grp[grp["condition"] == cond_b].set_index("seed")["value"]
        common = sorted(set(a.index) & set(b.index))
        if not common:
            continue
        diff = a.loc[common].values - b.loc[common].values
        n = len(diff)
        if n > 1 and np.std(diff, ddof=1) > 0:
            t_stat, p_value = stats.ttest_rel(a.loc[common].values, b.loc[common].values)
            t_stat, p_value = float(t_stat), float(p_value)
        elif n > 1:
            t_stat, p_value = float("nan"), float("nan")
        else:
            t_stat, p_value = float("nan"), float("nan")
        rows.append(
            {
                "agency_id": agency,
                "region_id": region,
                "test_split": split,
                "metric": metric,
                f"{cond_a}_mean": float(a.loc[common].mean()),
                f"{cond_b}_mean": float(b.loc[common].mean()),
                "delta_mean": float(np.mean(diff)),
                "delta_std": float(np.std(diff, ddof=1)) if n > 1 else 0.0,
                "n_seeds": n,
                "t_stat": t_stat,
                "p_value": p_value,
                "df": n - 1 if n > 1 else 0,
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path
