import csv
import json
import math
from pathlib import Path

import pytest

from game2048 import comparison
from game2048.comparison import paired_metrics, select_planner


def rows(exponents, start=0):
    return [
        {"seed": start + i, "max_exponent": e, "max_tile": 1 << e, "tile_sum": 512, "moves": 200}
        for i, e in enumerate(exponents)
    ]


def test_selection_requires_evidence_for_action_search_and_detects_regression():
    baseline = rows([8, 9] * 20)
    assert select_planner(baseline, baseline, baseline)["selected"] == "rollout"
    improved = rows([9, 10] * 20)
    result = select_planner(baseline, baseline, improved)
    assert result["selected"] == "mppi" and not result["screened_out"]
    worse = rows([7, 8] * 20)
    assert select_planner(baseline, worse, worse)["screened_out"]
    with pytest.raises(ValueError, match="identical seeds"):
        paired_metrics(baseline, rows([8, 9] * 20, start=1))


def test_comparison_freezes_models_and_selects_only_one_formal_planner(tmp_path, monkeypatch):
    experiment = tmp_path / "old"
    for seed in range(3):
        checkpoint = experiment / f"seed{seed}" / "td" / "best"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.json").write_text(json.dumps({"training": "td_lambda", "seed": seed}))
        (checkpoint / "params.msgpack").write_bytes(bytes([seed]))
    calls = []

    def evaluate(checkpoint, destination, mode, games, seed, planner, log_directory):
        calls.append((mode, games, seed, planner.scenarios))
        destination.mkdir(parents=True, exist_ok=True)
        # The same alternating gain repeats across models: these are correlated
        # game seeds, not three times as many independent observations.
        records = rows(
            [8 + int(seed == 300_000 and mode == "rollout" and i % 2 == 1) for i in range(games)],
            start=seed,
        )
        with (destination / "games.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        return {"mode": mode, "games": games}

    monkeypatch.setattr(comparison, "evaluate_checkpoint", evaluate)
    output = tmp_path / "new"
    result = comparison.compare_planners(experiment, output)
    assert result["status"] == "complete" and len(calls) == 9
    assert calls[:3] == [
        ("greedy", 256, 210_000, 8),
        ("rollout", 256, 210_000, 384),
        ("mppi", 256, 210_000, 8),
    ]
    assert {mode for mode, games, _, _ in calls if games == 2000} == {"greedy", "rollout"}
    assert result["aggregate_paired"]["max_exponent"]["mean"] == 0.5
    assert result["aggregate_paired"]["max_exponent"]["standard_error"] == pytest.approx(
        0.5 / math.sqrt(1999)
    )
    with pytest.raises(ValueError, match="outside historical"):
        comparison.compare_planners(experiment, experiment / "new")
    (Path(experiment) / "seed0" / "td" / "best" / "params.msgpack").write_bytes(b"changed")
    with pytest.raises(ValueError, match="specification changed"):
        comparison.compare_planners(experiment, output)
