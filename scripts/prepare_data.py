from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BODMAS_FOLDER_URL = "https://drive.google.com/drive/folders/1Uf-LebLWyi9eCv97iBal7kL1NgiGEsv_"


def _ensure_downloaded(raw_dir: Path) -> None:
    need = ["bodmas.npz", "bodmas_metadata.csv", "bodmas_malware_category.csv"]
    if all((raw_dir / f).exists() for f in need):
        print(f"[prepare_data] found existing files in {raw_dir}")
        return
    import gdown

    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prepare_data] downloading BODMAS to {raw_dir} ...")
    gdown.download_folder(BODMAS_FOLDER_URL, output=str(raw_dir), quiet=False)


def main() -> int:
    from src.config import load_config
    from src.dataset.pipeline import prepare_corpus

    config = load_config("base")
    raw_dir = REPO_ROOT / config.get("data", {}).get("raw_dir", "data/bodmas_raw")
    _ensure_downloaded(raw_dir)

    records, report, _ = prepare_corpus(config, seed=config["data"]["data_seed"])

    print("\n=== Section 6.1 / 6.2 dataset report ===")
    print(f"total records used: {report['n_records']}")
    print(f"classes ({report['n_classes']}): "
          f"GLOBAL={report['global_families']}  REGIONAL_ONLY={report['restricted_families']}")
    print("family counts (selected records only):")
    for fam, n in sorted(report["policy_assignment"]["family_counts"].items()):
        scope = report["policy_assignment"]["family_to_scope"][fam]
        print(f"  {fam:12s} {scope:14s} n={n}")
    print(f"\npartition: agency_budget_per_class={report['partition']['agency_budget_per_class']}")
    for k, v in sorted(report["partition"]["counts"].items()):
        if k.startswith("agency:") and ":split:" not in k and ":family:" not in k:
            print(f"  {k}: {v}")
    print(f"\nsplit: train_per_class={report['split']['train_per_class']} "
          f"test_per_class={report['split']['test_per_class']} seed_used={report['split']['seed_used']}")
    print(f"restricted test pool size (A1+A2 pooled): {report['restricted_test_pool_size']}")

    out_path = REPO_ROOT / "results" / "dataset_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
