from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config import REPO_ROOT
from src.dataset.pipeline import agency_test_records, agency_train_records, prepare_corpus
from src.dataset.records import Record, summarise
from src.evaluation.evaluator import evaluate_all_agencies
from src.fl.client import AgencyClient
from src.fl.orchestrator import run_flat, run_hierarchical
from src.policy.gate import GateDecision
from src.policy.scope import PolicyScope
from src.provenance.logger import ProvenanceLogger, render_national_round_summary

AGENCY_REGIONS = {"A1": "R1", "A2": "R1", "A3": "R2", "A4": "R2"}
RESULTS_ROOT = REPO_ROOT / "results" / "raw"


def _log(msg: str) -> None:
    print(f"    [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run_dir(condition: str, seed: int) -> Path:
    d = RESULTS_ROOT / condition / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _lib_versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    from importlib.metadata import PackageNotFoundError, version

    for mod in ("numpy", "pandas", "torch", "scikit-learn", "flwr"):
        try:
            out[mod] = version(mod)
        except PackageNotFoundError:
            out[mod] = "not installed"
    return out


def _num_rounds(config: dict) -> int:
    return int(config.get("federated", {}).get("num_rounds", 5))


def _n_classes(config: dict) -> int:
    d = config.get("data", {})
    return len(d.get("global_families", [])) + len(d.get("restricted_families", []))


def _b4_clients(bundle, records: list[Record], config: dict, seed: int) -> list[AgencyClient]:
    clients: list[AgencyClient] = []
    for agency in ("A1", "A2"):
        for scope in (PolicyScope.GLOBAL, PolicyScope.REGIONAL_ONLY):
            recs = [r for r in agency_train_records(records, agency) if r.scope is scope]
            clients.append(
                AgencyClient(
                    client_id=f"{agency}|{scope.value}",
                    agency_id=agency,
                    region_id=AGENCY_REGIONS[agency],
                    policy_scope=scope,
                    bundle=bundle,
                    train_records=recs,
                    config=config,
                    seed=seed,
                )
            )
    for agency in ("A3", "A4"):
        clients.append(
            AgencyClient(
                client_id=f"{agency}|GLOBAL",
                agency_id=agency,
                region_id=AGENCY_REGIONS[agency],
                policy_scope=PolicyScope.GLOBAL,
                bundle=bundle,
                train_records=agency_train_records(records, agency),
                config=config,
                seed=seed,
            )
        )
    return clients


def _flat_clients(
    bundle, records: list[Record], config: dict, seed: int, drop_restricted: bool
) -> list[AgencyClient]:
    clients: list[AgencyClient] = []
    for agency in ("A1", "A2", "A3", "A4"):
        recs = agency_train_records(records, agency)
        if drop_restricted:
            recs = [r for r in recs if r.scope is PolicyScope.GLOBAL]
        clients.append(
            AgencyClient(
                client_id=f"{agency}|MIXED",
                agency_id=agency,
                region_id=AGENCY_REGIONS[agency],
                policy_scope=PolicyScope.GLOBAL,
                bundle=bundle,
                train_records=recs,
                config=config,
                seed=seed,
            )
        )
    return clients


def run_condition(
    condition: str,
    seed: int,
    config: dict,
    *,
    use_synthetic: bool = False,
) -> dict[str, Any]:
    from src.model import backend

    t0 = time.time()
    run_dir = _run_dir(condition, seed)
    records, data_report, (scaler_mean, scaler_std) = prepare_corpus(
        config, seed, use_synthetic=use_synthetic
    )
    n_classes = data_report["n_classes"]

    _log(f"{condition} seed {seed}: loading model backend ({n_classes} classes)")
    bundle = backend.build_bundle(
        config, seed=seed, n_classes=n_classes, scaler_mean=scaler_mean, scaler_std=scaler_std
    )
    init_params = backend.get_parameters(config, bundle)
    num_rounds = _num_rounds(config)
    _log(f"model ready on {getattr(bundle, 'device', '?')}; {num_rounds} rounds")

    provenance_path = run_dir / "provenance.jsonl"
    logger = ProvenanceLogger(provenance_path)
    agency_params: dict[str, list[np.ndarray]]
    policy_summary: dict[str, Any] = {}

    if condition == "B1":
        agency_params = {}
        for agency in ("A1", "A2", "A3", "A4"):
            backend.set_parameters(config, bundle, init_params)
            recs = agency_train_records(records, agency)
            for rnd in range(1, num_rounds + 1):
                backend.train(config, bundle, recs, seed + rnd)
            agency_params[agency] = backend.get_parameters(config, bundle)
            _log(f"{agency} local training done ({len(recs)} records)")

    elif condition in ("B2", "B3"):
        clients = _flat_clients(bundle, records, config, seed, drop_restricted=(condition == "B3"))
        outcome = run_flat(clients, init_params, num_rounds, logger=logger)
        agency_params = outcome.agency_params
        policy_summary = {"per_round_accepted": outcome.per_round_accepted}

    elif condition == "B4":
        clients = _b4_clients(bundle, records, config, seed)
        blend_weighting = config.get("federated", {}).get("return_path_blend", "equal")
        outcome = run_hierarchical(
            clients, init_params, num_rounds, AGENCY_REGIONS, logger=logger,
            blend_weighting=blend_weighting,
        )
        agency_params = outcome.agency_params
        policy_summary = {
            "total_policy_violations": outcome.total_policy_violations,
            "per_round_delivered_components": outcome.per_round_delivered,
        }
        for rnd in range(1, num_rounds + 1):
            (run_dir / f"national_round_{rnd}.txt").write_text(
                render_national_round_summary(logger.records, rnd), encoding="utf-8"
            )
    else:
        raise ValueError(f"unknown condition {condition!r}")

    _log(f"training complete ({time.time() - t0:.0f}s); evaluating per agency")
    eval_rows = evaluate_all_agencies(bundle, agency_params, records, config, condition=condition, seed=seed)

    per = _policy_enforcement(logger.records)

    run_config = {
        "condition": condition,
        "seed": seed,
        "data_seed": config.get("data", {}).get("data_seed", 42),
        "num_rounds": num_rounds,
        "model": config.get("model", {}),
        "train": config.get("train", {}),
        "federated": config.get("federated", {}),
        "policy": config.get("policy", {}),
        "data": config.get("data", {}),
        "n_classes": n_classes,
        "trainable_parameters": backend.count_trainable(config, bundle),
        "device": bundle.device,
        "library_versions": _lib_versions(),
        "wall_time_sec": round(time.time() - t0, 1),
        "data_report": data_report,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
    _write_csv(run_dir / "final_metrics.csv", eval_rows)
    (run_dir / "policy_enforcement.json").write_text(
        json.dumps({**per, **policy_summary}, indent=2, default=str), encoding="utf-8"
    )

    return {
        "condition": condition,
        "seed": seed,
        "run_dir": str(run_dir),
        "eval_rows": eval_rows,
        "policy_enforcement": per,
        "data_report": data_report,
    }


def _policy_enforcement(records: list[dict]) -> dict[str, Any]:
    boundary = [r for r in records if r.get("stage") == "regional_to_national"]
    ro_attempts = [r for r in boundary if r.get("policy_scope") == "REGIONAL_ONLY"]
    ro_excluded = [r for r in ro_attempts if r.get("decision") == "EXCLUDE"]
    global_forwarded = [
        r for r in boundary if r.get("policy_scope") == "GLOBAL" and r.get("decision") == "INCLUDE"
    ]
    violations = [r for r in ro_attempts if r.get("decision") == "INCLUDE"]
    per = (len(ro_excluded) / len(ro_attempts)) if ro_attempts else float("nan")
    return {
        "regional_only_forward_attempts": len(ro_attempts),
        "regional_only_blocked_nationally": len(ro_excluded),
        "global_updates_forwarded_nationally": len(global_forwarded),
        "policy_violations": len(violations),
        "policy_enforcement_rate": per,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
