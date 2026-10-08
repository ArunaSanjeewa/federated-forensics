from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from src.dataset.records import Record


@dataclass
class SplitConfig:
    train_per_class: int = 150
    test_per_class: int = 50
    split_per_seed: bool = True
    data_seed: int = 42


@dataclass
class SplitReport:
    train_per_class: int
    test_per_class: int
    seed_used: int
    n_train: int
    n_test: int
    strata_short: list[str]

    def as_dict(self) -> dict:
        return {
            "train_per_class": self.train_per_class,
            "test_per_class": self.test_per_class,
            "seed_used": self.seed_used,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "strata_short": self.strata_short,
        }


def split_records(records: list[Record], cfg: SplitConfig, seed: int) -> SplitReport:
    seed_used = seed if cfg.split_per_seed else cfg.data_seed
    rng = random.Random(seed_used)

    strata: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for r in records:
        strata[(r.agency_id or "?", r.family)].append(r)

    short: list[str] = []
    n_train = n_test = 0
    need = cfg.train_per_class + cfg.test_per_class
    for (agency, family), group in sorted(strata.items()):
        group = list(group)
        rng.shuffle(group)
        if len(group) < need:
            short.append(f"{agency}/{family} (n={len(group)}, need={need})")
        train_n = min(cfg.train_per_class, len(group))
        test_n = min(cfg.test_per_class, max(len(group) - train_n, 0))
        for r in group[:train_n]:
            r.split = "train"
        for r in group[train_n : train_n + test_n]:
            r.split = "test"
        n_train += train_n
        n_test += test_n

    return SplitReport(
        train_per_class=cfg.train_per_class,
        test_per_class=cfg.test_per_class,
        seed_used=seed_used,
        n_train=n_train,
        n_test=n_test,
        strata_short=short,
    )


def restricted_test_records(records: list[Record]) -> list[Record]:
    return [
        r
        for r in records
        if r.split == "test" and r.scope.value == "REGIONAL_ONLY"
    ]
