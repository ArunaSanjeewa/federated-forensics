import pytest

from src.config import load_config


@pytest.fixture()
def mock_results_dir(tmp_path, monkeypatch):
    import src.experiment.runner as runner
    import src.results.aggregate as aggregate

    raw = tmp_path / "raw"
    monkeypatch.setattr(runner, "RESULTS_ROOT", raw)
    monkeypatch.setattr(aggregate, "RAW", raw)
    monkeypatch.setattr(aggregate, "OUT", tmp_path)
    return tmp_path


def _cfg(condition):
    c = load_config(condition)
    c.setdefault("model", {})["backend"] = "mock"
    c["data"]["agency_budget_per_class"] = 20
    c["data"]["train_per_class"] = 12
    c["data"]["test_per_class"] = 8
    return c


def test_all_conditions_run_and_aggregate(mock_results_dir):
    from src.experiment.runner import run_condition
    from src.results.aggregate import aggregate

    for condition in ("B1", "B2", "B3", "B4"):
        for seed in (42, 43):
            summary = run_condition(condition, seed, _cfg(condition), use_synthetic=True)
            assert summary["eval_rows"]
            assert summary["policy_enforcement"]["policy_violations"] == 0

    paths = aggregate()
    for p in paths.values():
        assert p.exists()


def test_b4_enforces_policy(mock_results_dir):
    from src.experiment.runner import run_condition

    summary = run_condition("B4", 42, _cfg("B4"), use_synthetic=True)
    per = summary["policy_enforcement"]
    assert per["policy_enforcement_rate"] == 1.0
    assert per["policy_violations"] == 0
    assert per["regional_only_blocked_nationally"] == per["regional_only_forward_attempts"] > 0

    run_dir = mock_results_dir / "raw" / "B4" / "seed_42"
    for txt in run_dir.glob("national_round_*.txt"):
        body = txt.read_text()
        accepted_line = [l for l in body.splitlines() if l.startswith("accepted_updates")][0]
        assert "regional" not in accepted_line


def test_b3_excludes_restricted_from_training_not_eval(mock_results_dir):
    from src.experiment.runner import run_condition

    summary = run_condition("B3", 42, _cfg("B3"), use_synthetic=True)
    restricted_rows = [r for r in summary["eval_rows"] if r["test_split"] == "restricted"]
    assert restricted_rows
    assert all(r["n"] > 0 for r in restricted_rows)
