from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="+", default=["B1", "B2", "B3", "B4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="default: the seed list in configs/base.yaml")
    ap.add_argument("--synthetic", action="store_true",
                    help="use the offline synthetic corpus (no download, no real training quality)")
    ap.add_argument("--backend", choices=["hf", "mock"], default=None,
                    help="override model.backend; 'mock' runs the whole pipeline with no ML stack")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    if not args.aggregate_only:
        from src.experiment.runner import run_condition

        base = load_config("base")
        seeds = args.seeds or base.get("federated", {}).get("seeds", [42, 43, 44, 45, 46])

        for condition in args.conditions:
            config = load_config(condition)
            if args.backend:
                config.setdefault("model", {})["backend"] = args.backend
            for seed in seeds:
                print(f"=== {condition} seed {seed} ===", flush=True)
                try:
                    summary = run_condition(
                        condition, seed, config, use_synthetic=args.synthetic
                    )
                    per = summary["policy_enforcement"]
                    print(
                        f"    done -> {summary['run_dir']}  "
                        f"violations={per['policy_violations']} "
                        f"PER={per['policy_enforcement_rate']}",
                        flush=True,
                    )
                except Exception:
                    print(f"    FAILED {condition} seed {seed}", flush=True)
                    traceback.print_exc()

    try:
        from src.results.aggregate import aggregate

        paths = aggregate()
        print("\nAggregated:")
        for name, p in paths.items():
            print(f"  {name}: {p}")
    except FileNotFoundError as exc:
        print(f"\n[aggregate skipped] {exc}")
        return 0

    if not args.no_figures:
        from src.results.figures import main_results, policy_flow

        for p in main_results():
            print(f"  figure: {p}")
        for p in policy_flow():
            print(f"  figure: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
