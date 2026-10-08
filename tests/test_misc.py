import json

import numpy as np
import pytest

from src.config import load_config
from src.fl.orchestrator import run_flat
from src.policy.scope import PolicyScope
from src.provenance.logger import ProvenanceLogger
from src.update.metadata import Update, make_update_id


def test_config_inheritance():
    base = load_config("base")
    b4 = load_config("B4")
    assert b4["model"]["hidden_dims"] == base["model"]["hidden_dims"]
    assert b4["federated"]["seeds"] == [42, 43, 44, 45, 46]
    assert b4["condition"]["id"] == "B4"
    assert b4["condition"]["policy_gate"] == "active"


def test_all_condition_configs_load():
    for c in ("B1", "B2", "B3", "B4"):
        cfg = load_config(c)
        assert cfg["condition"]["id"] == c
        assert cfg["policy"]["scopes"] == ["GLOBAL", "REGIONAL_ONLY"]


def test_global_and_restricted_families_share_no_names():
    base = load_config("base")
    g = set(base["data"]["global_families"])
    r = set(base["data"]["restricted_families"])
    assert g.isdisjoint(r)
    assert len(g) == 6 and len(r) == 2


def test_update_id_format():
    assert make_update_id("A1", PolicyScope.GLOBAL, 5) == "A1_global_r5"
    assert make_update_id("A2", PolicyScope.REGIONAL_ONLY, 3) == "A2_regional_r3"


def test_update_metadata_record():
    u = Update([np.ones((2, 2), dtype=np.float32)], "A1", "R1", 1, "GLOBAL", 7)
    rec = u.metadata_record()
    assert rec["update_id"] == "A1_global_r1"
    assert rec["policy_scope"] == "GLOBAL"
    assert rec["num_examples"] == 7
    assert len(rec["param_hash"]) == 12


def test_provenance_logger_roundtrip(tmp_path):
    path = tmp_path / "p.jsonl"
    logger = ProvenanceLogger(path)
    logger.log({"round_id": 1, "update_id": "A1_global_r1", "decision": "INCLUDE"})
    logger.log({"round_id": 1, "update_id": "A1_regional_r1", "decision": "EXCLUDE"})
    loaded = ProvenanceLogger.load(path)
    assert len(loaded) == 2
    assert loaded[1]["decision"] == "EXCLUDE"


class _StubClient:
    def __init__(self, agency_id, region_id, value):
        self.agency_id = agency_id
        self.region_id = region_id
        self._value = value

    def fit(self, parameters, config):
        params = [np.full((3,), float(self._value), dtype=np.float32)]
        return params, 10, {
            "client_id": f"{self.agency_id}|MIXED",
            "agency_id": self.agency_id,
            "region_id": self.region_id,
            "round_id": int(config["round_id"]),
            "policy_scope": "GLOBAL",
            "final_loss": 0.0,
            "steps": 1,
        }


def test_run_flat_all_agencies_end_with_same_model(tmp_path):
    clients = [
        _StubClient("A1", "R1", 1.0),
        _StubClient("A2", "R1", 2.0),
        _StubClient("A3", "R2", 3.0),
        _StubClient("A4", "R2", 4.0),
    ]
    logger = ProvenanceLogger(tmp_path / "p.jsonl")
    outcome = run_flat(clients, [np.zeros((3,), dtype=np.float32)], num_rounds=2, logger=logger)
    models = list(outcome.agency_params.values())
    for m in models[1:]:
        assert np.allclose(m[0], models[0][0])
    assert all(r["stage"] == "flat_national" for r in logger.records)
