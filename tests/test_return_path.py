import numpy as np
import pytest

from src.fl.aggregation import hierarchical_aggregate
from src.fl.return_path import build_return_path
from src.policy.scope import PolicyScope
from src.update.metadata import Update

AGENCY_REGIONS = {"A1": "R1", "A2": "R1", "A3": "R2", "A4": "R2"}


def _params(v, shape=(2, 2)):
    return [np.full(shape, float(v), dtype=np.float32)]


def _updates(round_id=1):
    return [
        Update(_params(1.0), "A1", "R1", round_id, PolicyScope.GLOBAL, 10),
        Update(_params(2.0), "A1", "R1", round_id, PolicyScope.REGIONAL_ONLY, 10),
        Update(_params(3.0), "A2", "R1", round_id, PolicyScope.GLOBAL, 10),
        Update(_params(4.0), "A2", "R1", round_id, PolicyScope.REGIONAL_ONLY, 10),
        Update(_params(5.0), "A3", "R2", round_id, PolicyScope.GLOBAL, 10),
        Update(_params(6.0), "A4", "R2", round_id, PolicyScope.GLOBAL, 10),
    ]


def test_every_agency_path_goes_through_a_region():
    updates = _updates()
    result = hierarchical_aggregate(updates, round_id=1)
    delivered = build_return_path(result, updates, AGENCY_REGIONS, round_id=1)
    for agency, model in delivered.items():
        assert model.path[0] == "NATIONAL"
        assert model.path[1] == AGENCY_REGIONS[agency]
        assert model.path[2] == agency
        assert len(model.path) == 3


def test_region1_model_blends_national_and_regional_only():
    updates = _updates()
    result = hierarchical_aggregate(updates, round_id=1)
    delivered = build_return_path(result, updates, AGENCY_REGIONS, round_id=1)
    assert delivered["A1"].components == ["national_model", "R1_regional_only"]
    assert delivered["A3"].components == ["national_model"]
    assert np.allclose(delivered["A3"].parameters[0], result.national_model[0])
    assert not np.allclose(delivered["A1"].parameters[0], result.national_model[0])


def test_equal_blend_ignores_sample_counts():
    updates = _updates()
    result = hierarchical_aggregate(updates, round_id=1)
    sample_delivered = build_return_path(
        result, updates, AGENCY_REGIONS, round_id=1, blend_weighting="sample"
    )
    equal_delivered = build_return_path(
        result, updates, AGENCY_REGIONS, round_id=1, blend_weighting="equal"
    )
    assert np.allclose(sample_delivered["A1"].parameters[0], 3.5)
    assert np.allclose(equal_delivered["A1"].parameters[0], 3.375)
    assert abs(3.375 - 3.0) < abs(3.5 - 3.0)


def test_return_path_rejects_unknown_weighting():
    updates = _updates()
    result = hierarchical_aggregate(updates, round_id=1)
    with pytest.raises(ValueError):
        build_return_path(result, updates, AGENCY_REGIONS, round_id=1, blend_weighting="bogus")


def test_return_path_logs_records(tmp_path):
    from src.provenance.logger import ProvenanceLogger

    updates = _updates()
    result = hierarchical_aggregate(updates, round_id=1)
    logger = ProvenanceLogger(tmp_path / "prov.jsonl")
    build_return_path(result, updates, AGENCY_REGIONS, logger=logger, round_id=1)
    stages = {r["stage"] for r in logger.records}
    assert "national_to_regional_to_local" in stages
    assert len(logger.records) == 4
