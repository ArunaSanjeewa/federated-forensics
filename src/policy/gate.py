from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .scope import PolicyScope, parse_scope


class Destination(str, Enum):
    REGIONAL = "REGIONAL"
    NATIONAL = "NATIONAL"

    def __str__(self) -> str:
        return self.value


class GateDecision(str, Enum):
    INCLUDE = "INCLUDE"
    EXCLUDE = "EXCLUDE"

    def __str__(self) -> str:
        return self.value


_RULES: dict[tuple[PolicyScope, Destination], GateDecision] = {
    (PolicyScope.GLOBAL, Destination.REGIONAL): GateDecision.INCLUDE,
    (PolicyScope.GLOBAL, Destination.NATIONAL): GateDecision.INCLUDE,
    (PolicyScope.REGIONAL_ONLY, Destination.REGIONAL): GateDecision.INCLUDE,
    (PolicyScope.REGIONAL_ONLY, Destination.NATIONAL): GateDecision.EXCLUDE,
}


def can_forward(
    scope: PolicyScope | str, destination: Destination | str
) -> GateDecision:
    scope = parse_scope(scope)
    if not isinstance(destination, Destination):
        destination = Destination(str(destination).strip().upper())
    try:
        return _RULES[(scope, destination)]
    except KeyError:
        raise ValueError(f"no gate rule for ({scope}, {destination})")


@dataclass(frozen=True)
class GateEvaluation:

    update_id: str
    agency_id: str
    region_id: str
    round_id: int
    policy_scope: PolicyScope
    source: str
    destination: Destination
    decision: GateDecision
    stage: str

    def as_record(self) -> dict:
        return {
            "update_id": self.update_id,
            "agency_id": self.agency_id,
            "region_id": self.region_id,
            "round_id": self.round_id,
            "policy_scope": self.policy_scope.value,
            "source": self.source,
            "destination": self.destination.value,
            "decision": self.decision.value,
            "stage": self.stage,
        }


class PolicyGate:

    def __init__(self, logger: Optional["object"] = None) -> None:
        self._logger = logger
        self.evaluations: list[GateEvaluation] = []

    def evaluate(
        self,
        *,
        update_id: str,
        agency_id: str,
        region_id: str,
        round_id: int,
        policy_scope: PolicyScope | str,
        source: str,
        destination: Destination | str,
        stage: str,
    ) -> GateEvaluation:
        scope = parse_scope(policy_scope)
        dest = (
            destination
            if isinstance(destination, Destination)
            else Destination(str(destination).strip().upper())
        )
        decision = can_forward(scope, dest)
        evaluation = GateEvaluation(
            update_id=update_id,
            agency_id=agency_id,
            region_id=region_id,
            round_id=round_id,
            policy_scope=scope,
            source=source,
            destination=dest,
            decision=decision,
            stage=stage,
        )
        self.evaluations.append(evaluation)
        if self._logger is not None:
            self._logger.log(evaluation.as_record())
        return evaluation

    def included(self, evaluations: Optional[list[GateEvaluation]] = None) -> list[GateEvaluation]:
        source = self.evaluations if evaluations is None else evaluations
        return [e for e in source if e.decision is GateDecision.INCLUDE]

    def excluded(self, evaluations: Optional[list[GateEvaluation]] = None) -> list[GateEvaluation]:
        source = self.evaluations if evaluations is None else evaluations
        return [e for e in source if e.decision is GateDecision.EXCLUDE]
