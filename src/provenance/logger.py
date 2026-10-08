from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable


class ProvenanceLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._records: list[dict] = []
        self.path.write_text("", encoding="utf-8")

    def log(self, record: dict) -> None:
        with self._lock:
            self._records.append(record)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")

    def log_many(self, records: Iterable[dict]) -> None:
        for record in records:
            self.log(record)

    @property
    def records(self) -> list[dict]:
        return list(self._records)

    def filter(self, **equals) -> list[dict]:
        out = []
        for r in self._records:
            if all(r.get(k) == v for k, v in equals.items()):
                out.append(r)
        return out

    @staticmethod
    def load(path: str | Path) -> list[dict]:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]


def render_national_round_summary(records: list[dict], round_id: int) -> str:
    at_boundary = [
        r
        for r in records
        if r.get("round_id") == round_id
        and r.get("stage") == "regional_to_national"
    ]
    accepted = sorted(r["update_id"] for r in at_boundary if r["decision"] == "INCLUDE")
    excluded = sorted(r["update_id"] for r in at_boundary if r["decision"] == "EXCLUDE")
    lines = [
        f"National Round {round_id}:",
        f"accepted_updates = [{', '.join(accepted)}]",
        f"excluded_updates = [{', '.join(excluded)}]",
    ]
    return "\n".join(lines)
