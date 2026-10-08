import pytest

from src.config import load_config
from src.dataset.partition import PartitionConfig, partition_records, validate_partition
from src.dataset.pipeline import agency_test_records, agency_train_records, prepare_corpus
from src.dataset.policy_assign import assign_scopes, scope_for_family
from src.dataset.records import summarise
from src.dataset.split import SplitConfig, restricted_test_records, split_records
from src.dataset.loader import synthetic_corpus
from src.policy.scope import PolicyScope

RESTRICTED = ["fam_r1"]
GLOBAL = ["fam_g1", "fam_g2", "fam_g3"]


def test_scope_for_family():
    assert scope_for_family("fam_r1", RESTRICTED) is PolicyScope.REGIONAL_ONLY
    assert scope_for_family("fam_g1", RESTRICTED) is PolicyScope.GLOBAL


def test_assign_scopes_report():
    fams = ["fam_g1", "fam_r1", "fam_g1", "fam_r1", "fam_r1"]
    scopes, report = assign_scopes(fams, RESTRICTED)
    assert report.n_regional_only == 3
    assert report.n_global == 2
    assert report.family_to_scope["fam_r1"] == "REGIONAL_ONLY"


def _synthetic(seed=1, n_per_family=100):
    return synthetic_corpus(
        global_families=GLOBAL, restricted_families=RESTRICTED, n_per_family=n_per_family, seed=seed
    )


def test_partition_topology_contract():
    records = _synthetic(seed=1)
    records = partition_records(records, PartitionConfig(data_seed=42, agency_budget_per_class=20))
    validate_partition(records)

    holdings = summarise(records)
    assert holdings["agency:A1:REGIONAL_ONLY"] > 0
    assert holdings["agency:A2:REGIONAL_ONLY"] > 0
    assert "agency:A3:REGIONAL_ONLY" not in holdings
    assert "agency:A4:REGIONAL_ONLY" not in holdings


def test_partition_is_deterministic():
    a = _synthetic(seed=1)
    b = _synthetic(seed=1)
    a = partition_records(a, PartitionConfig(data_seed=7, agency_budget_per_class=20))
    b = partition_records(b, PartitionConfig(data_seed=7, agency_budget_per_class=20))
    assert [r.agency_id for r in a] == [r.agency_id for r in b]


def test_partition_agencies_get_disjoint_records():
    records = _synthetic(seed=1)
    records = partition_records(records, PartitionConfig(data_seed=42, agency_budget_per_class=20))
    ids_by_agency = {}
    for r in records:
        if r.family in GLOBAL:
            ids_by_agency.setdefault(r.agency_id, set()).add(r.record_id)
    all_ids = [i for s in ids_by_agency.values() for i in s]
    assert len(all_ids) == len(set(all_ids))


def test_partition_raises_if_budget_exceeds_supply():
    records = _synthetic(seed=1, n_per_family=10)
    with pytest.raises(ValueError):
        partition_records(records, PartitionConfig(data_seed=42, agency_budget_per_class=20))


def test_split_exact_counts_and_no_leakage():
    records = _synthetic(seed=2)
    records = partition_records(records, PartitionConfig(data_seed=42, agency_budget_per_class=20))
    split_records(records, SplitConfig(train_per_class=12, test_per_class=8), seed=42)

    train = [r for r in records if r.split == "train"]
    test = [r for r in records if r.split == "test"]
    assert train and test
    assert any(r.scope is PolicyScope.REGIONAL_ONLY for r in train)
    assert len(restricted_test_records(records)) == 8 * 2

    train_ids = {id(r) for r in train}
    test_ids = {id(r) for r in test}
    assert train_ids.isdisjoint(test_ids)


def test_split_reshuffles_per_seed():
    a = _synthetic(seed=3)
    a = partition_records(a, PartitionConfig(data_seed=42, agency_budget_per_class=20))
    split_records(a, SplitConfig(train_per_class=12, test_per_class=8, split_per_seed=True), seed=42)
    a1_ids_42 = {r.record_id for r in a if r.split == "test" and r.agency_id == "A1"}

    b = _synthetic(seed=3)
    b = partition_records(b, PartitionConfig(data_seed=42, agency_budget_per_class=20))
    split_records(b, SplitConfig(train_per_class=12, test_per_class=8, split_per_seed=True), seed=43)
    a1_ids_43 = {r.record_id for r in b if r.split == "test" and r.agency_id == "A1"}

    assert a1_ids_42 != a1_ids_43


def test_prepare_corpus_synthetic_end_to_end():
    config = load_config("B4")
    config["data"]["global_families"] = GLOBAL
    config["data"]["restricted_families"] = RESTRICTED
    config["data"]["agency_budget_per_class"] = 20
    config["data"]["train_per_class"] = 12
    config["data"]["test_per_class"] = 8

    records, report, (mean, std) = prepare_corpus(config, seed=42, use_synthetic=True)
    assert report["n_records"] == len(records)
    assert report["policy_assignment"]["n_regional_only"] > 0
    assert report["restricted_test_pool_size"] >= 1
    assert len(agency_train_records(records, "A1")) > 0
    assert all(r.scope is PolicyScope.GLOBAL for r in agency_train_records(records, "A3"))
    assert mean is not None and mean.shape[0] == records[0].features.shape[0]
