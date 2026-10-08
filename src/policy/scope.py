from __future__ import annotations

from enum import Enum


class PolicyScope(str, Enum):
    GLOBAL = "GLOBAL"
    REGIONAL_ONLY = "REGIONAL_ONLY"

    def __str__(self) -> str:
        return self.value


def parse_scope(value: str | "PolicyScope") -> "PolicyScope":
    if isinstance(value, PolicyScope):
        return value
    key = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "GLOBAL": PolicyScope.GLOBAL,
        "REGIONAL_ONLY": PolicyScope.REGIONAL_ONLY,
        "REGION_ONLY": PolicyScope.REGIONAL_ONLY,
        "REGIONALONLY": PolicyScope.REGIONAL_ONLY,
    }
    if key not in aliases:
        raise ValueError(
            f"unknown policy scope {value!r}; expected GLOBAL or REGIONAL_ONLY"
        )
    return aliases[key]
