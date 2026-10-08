import numpy as np
import pytest

from src.fl.aggregation import fedavg, hierarchical_aggregate
from src.policy.scope import PolicyScope
from src.update.metadata import Update


def _params(value, shape=(2, 2)):
    return [np.full(shape, float(value), dtype=np.float32)]


def test_fedavg_is_sample_weighted():
    out = fedavg([(_params(0.0), 1), (_params(3.0), 3)])
    assert np.allclose(out[0], 2.25)


def test_fedavg_rejects_empty():
    with pytest.raises(ValueError):
        fedavg([])


def _round_updates(round_id=1):
    return [
        Update(_params(1.0), "A1", "R1", round_id, PolicyScope.GLOBAL, 10),
        Update(_params(2.0), "A1", "R1", round_id, PolicyScope.REGIONAL_ONLY, 10),
        Update(_params(3.0), "A2", "R1", round_id, PolicyScope.GLOBAL, 10),
        Update(_params(4.0), "A2", "R1", round_id, PolicyScope.REGIONAL_ONLY, 10),
        Update(_params(5.0), "A3", "R2", round_id, PolicyScope.GLOBAL, 10),
        Update(_params(6.0), "A4", "R2", round_id, PolicyScope.GLOBAL, 10),
    ]


def test_regional_model_includes_regional_only():
    result = hierarchical_aggregate(_round_updates(), round_id=1)
    assert np.allclose(result.regional["R1"].regional_model[0], 2.5)


def test_outward_model_excludes_regional_only():
    result = hierarchical_aggregate(_round_updates(), round_id=1)
    assert np.allclose(result.regional["R1"].outward_model[0], 2.0)
    assert result.regional["R1"].excluded_update_ids == ["A1_regional_r1", "A2_regional_r1"]


def test_national_model_has_no_regional_only_contribution():
    result = hierarchical_aggregate(_round_updates(), round_id=1)
    ids = result.national_accepted_update_ids
    assert "A1_regional_r1" not in ids
    assert "A2_regional_r1" not in ids
    assert set(ids) == {"A1_global_r1", "A2_global_r1", "A3_global_r1", "A4_global_r1"}
    assert np.allclose(result.national_model[0], 3.75)


def test_zero_policy_violations_by_construction():
    result = hierarchical_aggregate(_round_updates(), round_id=1)
    assert result.policy_violations == 0


def test_gate_evaluations_cover_both_boundaries():
    result = hierarchical_aggregate(_round_updates(), round_id=1)
    stages = {e.stage for e in result.gate_evaluations}
    assert stages == {"agency_to_regional", "regional_to_national"}
    boundary = [e for e in result.gate_evaluations if e.stage == "regional_to_national"]
    excluded = [e for e in boundary if e.decision.value == "EXCLUDE"]
    assert {e.update_id for e in excluded} == {"A1_regional_r1", "A2_regional_r1"}
