import numpy as np
import pytest

from src.fl.orchestrator import run_hierarchical
from src.policy.scope import PolicyScope
from src.provenance.logger import ProvenanceLogger, render_national_round_summary

AGENCY_REGIONS = {"A1": "R1", "A2": "R1", "A3": "R2", "A4": "R2"}


class StubClient:
    def __init__(self, client_id, agency_id, region_id, scope, value):
        self.client_id = client_id
        self.agency_id = agency_id
        self.region_id = region_id
        self.policy_scope = PolicyScope(scope)
        self._value = value

    def fit(self, parameters, config):
        params = [np.full((2, 2), float(self._value), dtype=np.float32)]
        metrics = {
            "client_id": self.client_id,
            "agency_id": self.agency_id,
            "region_id": self.region_id,
            "round_id": int(config["round_id"]),
            "policy_scope": self.policy_scope.value,
            "final_loss": 0.1,
            "steps": 1,
        }
        return params, 10, metrics


def _b4_clients():
    return [
        StubClient("A1|GLOBAL", "A1", "R1", "GLOBAL", 1.0),
        StubClient("A1|REGIONAL_ONLY", "A1", "R1", "REGIONAL_ONLY", 2.0),
        StubClient("A2|GLOBAL", "A2", "R1", "GLOBAL", 3.0),
        StubClient("A2|REGIONAL_ONLY", "A2", "R1", "REGIONAL_ONLY", 4.0),
        StubClient("A3|GLOBAL", "A3", "R2", "GLOBAL", 5.0),
        StubClient("A4|GLOBAL", "A4", "R2", "GLOBAL", 6.0),
    ]


def test_no_regional_only_update_in_any_national_round(tmp_path):
    logger = ProvenanceLogger(tmp_path / "prov.jsonl")
    init = [np.zeros((2, 2), dtype=np.float32)]
    outcome = run_hierarchical(_b4_clients(), init, num_rounds=3, agency_regions=AGENCY_REGIONS, logger=logger)

    assert outcome.total_policy_violations == 0

    for rnd in range(1, 4):
        summary = render_national_round_summary(logger.records, rnd)
        assert "regional" not in summary.split("accepted_updates")[1].split("]")[0]
        assert f"A1_regional_r{rnd}" in summary
        excluded_line = [l for l in summary.splitlines() if l.startswith("excluded_updates")][0]
        assert f"A1_regional_r{rnd}" in excluded_line
        assert f"A2_regional_r{rnd}" in excluded_line


def test_regional_only_updates_are_accepted_regionally(tmp_path):
    logger = ProvenanceLogger(tmp_path / "prov.jsonl")
    init = [np.zeros((2, 2), dtype=np.float32)]
    run_hierarchical(_b4_clients(), init, num_rounds=1, agency_regions=AGENCY_REGIONS, logger=logger)

    regional_records = [
        r for r in logger.records
        if r["stage"] == "agency_to_regional" and r["policy_scope"] == "REGIONAL_ONLY"
    ]
    assert regional_records
    assert all(r["decision"] == "INCLUDE" for r in regional_records)


def test_provenance_shows_full_lifecycle(tmp_path):
    logger = ProvenanceLogger(tmp_path / "prov.jsonl")
    init = [np.zeros((2, 2), dtype=np.float32)]
    run_hierarchical(_b4_clients(), init, num_rounds=1, agency_regions=AGENCY_REGIONS, logger=logger)

    a1_regional = [r for r in logger.records if r["update_id"] == "A1_regional_r1"]
    stages = {r["stage"]: r["decision"] for r in a1_regional}
    assert stages["agency_to_regional"] == "INCLUDE"
    assert stages["regional_to_national"] == "EXCLUDE"
