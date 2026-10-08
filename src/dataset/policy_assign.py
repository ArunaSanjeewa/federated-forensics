from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from src.policy.scope import PolicyScope


def scope_for_family(family: str, restricted_families: Sequence[str]) -> PolicyScope:
    return (
        PolicyScope.REGIONAL_ONLY
        if family in set(restricted_families)
        else PolicyScope.GLOBAL
    )


@dataclass
class PolicyAssignmentReport:
    total: int
    n_global: int
    n_regional_only: int
    family_to_scope: dict[str, str]
    family_counts: dict[str, int]

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "n_global": self.n_global,
            "n_regional_only": self.n_regional_only,
            "family_to_scope": self.family_to_scope,
            "family_counts": self.family_counts,
        }


def assign_scopes(
    families: Iterable[str],
    restricted_families: Sequence[str],
) -> tuple[list[PolicyScope], PolicyAssignmentReport]:
    families = [str(f) for f in families]
    scopes = [scope_for_family(f, restricted_families) for f in families]

    counts = Counter(families)
    fam_scope: dict[str, str] = {
        f: scope_for_family(f, restricted_families).value for f in counts
    }

    report = PolicyAssignmentReport(
        total=len(families),
        n_global=sum(1 for s in scopes if s is PolicyScope.GLOBAL),
        n_regional_only=sum(1 for s in scopes if s is PolicyScope.REGIONAL_ONLY),
        family_to_scope=fam_scope,
        family_counts=dict(counts),
    )
    return scopes, report
