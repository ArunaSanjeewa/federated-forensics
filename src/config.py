from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(name_or_path: str) -> dict[str, Any]:
    path = Path(name_or_path)
    if not path.exists():
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no config at {name_or_path!r} or {path}")

    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    parent_name = raw.pop("defaults", None)
    if parent_name:
        parent = load_config(parent_name)
        raw = _deep_merge(parent, raw)
    return raw


@dataclass
class ResolvedRun:

    condition: str
    seed: int
    data_seed: int
    values: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        rec = {"condition": self.condition, "seed": self.seed, "data_seed": self.data_seed}
        rec.update(self.values)
        return rec
