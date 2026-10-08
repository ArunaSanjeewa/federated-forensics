import pytest

from src.policy.gate import Destination, GateDecision, PolicyGate, can_forward
from src.policy.scope import PolicyScope, parse_scope


@pytest.mark.parametrize(
    "scope, dest, expected",
    [
        (PolicyScope.GLOBAL, Destination.REGIONAL, GateDecision.INCLUDE),
        (PolicyScope.GLOBAL, Destination.NATIONAL, GateDecision.INCLUDE),
        (PolicyScope.REGIONAL_ONLY, Destination.REGIONAL, GateDecision.INCLUDE),
        (PolicyScope.REGIONAL_ONLY, Destination.NATIONAL, GateDecision.EXCLUDE),
    ],
)
def test_truth_table(scope, dest, expected):
    assert can_forward(scope, dest) is expected


def test_truth_table_accepts_strings():
    assert can_forward("GLOBAL", "NATIONAL") is GateDecision.INCLUDE
    assert can_forward("REGIONAL_ONLY", "NATIONAL") is GateDecision.EXCLUDE


def test_only_regional_only_to_national_is_excluded():
    decisions = {
        (s, d): can_forward(s, d)
        for s in PolicyScope
        for d in Destination
    }
    excluded = [k for k, v in decisions.items() if v is GateDecision.EXCLUDE]
    assert excluded == [(PolicyScope.REGIONAL_ONLY, Destination.NATIONAL)]


def test_parse_scope_rejects_garbage():
    with pytest.raises(ValueError):
        parse_scope("SEMI_GLOBAL")


def test_parse_scope_aliases():
    assert parse_scope("regional-only") is PolicyScope.REGIONAL_ONLY
    assert parse_scope("REGION_ONLY") is PolicyScope.REGIONAL_ONLY


def test_gate_records_every_decision():
    gate = PolicyGate()
    gate.evaluate(
        update_id="A1_regional_r1", agency_id="A1", region_id="R1", round_id=1,
        policy_scope="REGIONAL_ONLY", source="R1", destination="NATIONAL",
        stage="regional_to_national",
    )
    gate.evaluate(
        update_id="A1_global_r1", agency_id="A1", region_id="R1", round_id=1,
        policy_scope="GLOBAL", source="R1", destination="NATIONAL",
        stage="regional_to_national",
    )
    assert len(gate.evaluations) == 2
    assert len(gate.excluded()) == 1
    assert gate.excluded()[0].update_id == "A1_regional_r1"
    assert len(gate.included()) == 1
