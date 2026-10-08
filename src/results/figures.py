from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import REPO_ROOT

OUT = REPO_ROOT / "results" / "figures"
CONDITION_COLOURS = {
    "B1": "#8c8c8c",
    "B2": "#0072B2",
    "B3": "#D55E00",
    "B4": "#009E73",
}
CONDITION_LABELS = {
    "B1": "B1 local-only",
    "B2": "B2 flat, unrestricted",
    "B3": "B3 flat, restricted excluded",
    "B4": "B4 hierarchical (proposed)",
}


def _load_per_agency() -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "results" / "per_agency_results.csv")


def main_results(metric: str = "f1_macro") -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    df = _load_per_agency()

    groups = [("A1", "all"), ("A2", "all"), ("A3", "all"), ("A4", "all"), ("A1", "restricted"), ("A2", "restricted")]
    group_labels = ["A1", "A2", "A3", "A4", "A1\n(restricted)", "A2\n(restricted)"]
    conditions = ["B1", "B2", "B3", "B4"]

    x = np.arange(len(groups))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, cond in enumerate(conditions):
        means, errs = [], []
        for agency, split in groups:
            row = df[(df["condition"] == cond) & (df["agency_id"] == agency) & (df["test_split"] == split)]
            if row.empty:
                means.append(np.nan)
                errs.append(0.0)
            else:
                means.append(float(row[f"{metric}_mean"].iloc[0]))
                errs.append(float(row[f"{metric}_std"].iloc[0]))
        ax.bar(x + (i - 1.5) * width, means, width, yerr=errs, capsize=3, label=CONDITION_LABELS[cond], color=CONDITION_COLOURS[cond])

    ax.axvline(3.5, color="0.7", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel(f"{metric} (mean +/- 1 SD over seeds)")
    ax.set_title("Per-agency malware-family classification quality by condition")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()

    paths = []
    for ext in ("pdf", "png"):
        p = OUT / f"main_results.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def policy_flow(condition: str = "B4", seed: int | None = None, round_id: int = 1) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = REPO_ROOT / "results" / "raw" / condition
    seed_dirs = sorted(raw.glob("seed_*"))
    if not seed_dirs:
        return []
    chosen = seed_dirs[0] if seed is None else raw / f"seed_{seed}"
    records = [
        json.loads(line)
        for line in (chosen / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    boundary = [r for r in records if r.get("stage") == "regional_to_national" and r.get("round_id") == round_id]
    accepted = [r["update_id"] for r in boundary if r["decision"] == "INCLUDE"]
    excluded = [r["update_id"] for r in boundary if r["decision"] == "EXCLUDE"]

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    ax.text(0.02, 0.85, f"{condition}  round {round_id}  regional -> national gate", fontsize=11, weight="bold")
    ax.text(0.02, 0.55, "INCLUDE (reached national aggregation):", color=CONDITION_COLOURS["B4"], fontsize=9)
    ax.text(0.05, 0.42, ", ".join(accepted) or "(none)", fontsize=9, family="monospace")
    ax.text(0.02, 0.22, "EXCLUDE (withheld at the regional boundary):", color=CONDITION_COLOURS["B3"], fontsize=9)
    ax.text(0.05, 0.09, ", ".join(excluded) or "(none)", fontsize=9, family="monospace")
    fig.tight_layout()

    paths = []
    for ext in ("pdf", "png"):
        p = OUT / f"policy_flow.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths
