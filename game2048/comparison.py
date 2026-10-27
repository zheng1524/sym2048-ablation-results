"""Frozen TD checkpoints: screen two planners, evaluate only the selected one.

No teacher or JAX import: each GPU evaluation runs in its own process. Existing
historical runs supply checkpoints only and are never written by this workflow.
"""

import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from game2048.artifacts import atomic_json
from game2048.config import INFERENCE_VERSIONS, PlannerConfig
from game2048.evaluation_jobs import evaluate_checkpoint


def paired_interval(differences) -> dict:
    values = np.asarray(differences, np.float64)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "standard_error": se,
        "lower": mean - 1.96 * se,
        "upper": mean + 1.96 * se,
    }


def read_games(directory: Path) -> list[dict[str, int]]:
    with (directory / "games.csv").open(newline="") as stream:
        return [{key: int(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def paired_differences(baseline, proposed) -> dict[str, np.ndarray]:
    """One difference per shared game seed, before any model aggregation."""
    if [row["seed"] for row in baseline] != [row["seed"] for row in proposed]:
        raise ValueError("Paired games must have identical seeds in identical order")
    fields = ("max_exponent", "max_tile", "tile_sum", "moves")
    before, after = [
        np.array([[row[field] for field in fields] for row in records], dtype=np.int64)
        for records in (baseline, proposed)
    ]
    result = dict(zip(fields, (after - before).T, strict=True))
    for exponent in (8, 9, 10):
        result[f"reach_{1 << exponent}"] = (after[:, 0] >= exponent).astype(np.int8) - (
            before[:, 0] >= exponent
        ).astype(np.int8)
    return result


def paired_metrics(baseline, proposed) -> dict:
    return {
        field: paired_interval(values)
        for field, values in paired_differences(baseline, proposed).items()
    }


def select_planner(greedy, rollout, mppi) -> dict:
    comparison = paired_metrics(rollout, mppi)["max_exponent"]
    against_greedy = {
        name: paired_metrics(greedy, rows)["max_exponent"]
        for name, rows in (("rollout", rollout), ("mppi", mppi))
    }
    return {
        "selected": "mppi" if comparison["lower"] > 0 else "rollout",
        "screened_out": all(delta["upper"] < 0 for delta in against_greedy.values()),
        "mppi_minus_rollout": comparison,
        "versus_greedy": against_greedy,
        "rule": "Select MPPI only if paired 95% lower bound over rollout is positive",
    }


def compare_planners(experiment: Path, output: Path) -> dict:
    experiment, output = experiment.resolve(), output.resolve()
    if output == experiment or experiment in output.parents:
        raise ValueError("Use a separate output directory, outside historical experiments")
    checkpoints = [experiment / f"seed{seed}" / "td" / "best" for seed in range(3)]
    configs = {"mppi": PlannerConfig(), "rollout": replace(PlannerConfig(), scenarios=384)}
    configs["greedy"] = PlannerConfig()
    models = []
    for checkpoint in checkpoints:
        metadata = json.loads((checkpoint / "model.json").read_text())
        if metadata.get("training") != "td_lambda":
            raise ValueError(f"Expected a TD checkpoint: {checkpoint}")
        models.append(
            {
                "path": str(checkpoint),
                "sha256": hashlib.sha256((checkpoint / "params.msgpack").read_bytes()).hexdigest(),
                "metadata": metadata,
            }
        )
    specification = {
        "schema": 1,
        "versions": INFERENCE_VERSIONS,
        "models": models,
        "configs": {name: asdict(config) for name, config in configs.items()},
        "validation": {"model_seed": 0, "seed": 210_000, "games": 256},
        "test": {"seed": 300_000, "games": 2000},
        "selection": "paired_normal_95_lower_gt_zero_else_rollout",
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "experiment.json"
    if manifest.exists() and json.loads(manifest.read_text()) != specification:
        raise ValueError("Comparison specification changed; use a new output directory")
    atomic_json(manifest, specification)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    validation = {}
    for mode in ("greedy", "rollout", "mppi"):
        validation[mode] = evaluate_checkpoint(
            checkpoints[0],
            output / "validation" / mode,
            mode,
            256,
            210_000,
            configs[mode],
            logs,
        )
    selection = select_planner(
        *(read_games(output / "validation" / mode) for mode in ("greedy", "rollout", "mppi"))
    )
    atomic_json(output / "selection.json", {**selection, "validation": validation})
    if selection["screened_out"]:
        result = {
            "status": "screened_out",
            "selection": selection,
            "note": "Both planners regressed in screening. Diagnose before a formal evaluation.",
        }
        atomic_json(output / "results.json", result)
        return result
    selected = selection["selected"]
    results = []
    differences_by_model = []
    for seed, checkpoint in enumerate(checkpoints):
        run = output / f"seed{seed}"
        metrics = {
            mode: evaluate_checkpoint(
                checkpoint, run / mode, mode, 2000, 300_000, configs[mode], logs
            )
            for mode in ("greedy", selected)
        }
        baseline, proposed = [read_games(run / mode) for mode in ("greedy", selected)]
        differences = paired_differences(baseline, proposed)
        differences_by_model.append(differences)
        results.append(
            {
                "training_seed": seed,
                "metrics": metrics,
                "paired": {field: paired_interval(values) for field, values in differences.items()},
            }
        )
        atomic_json(output / "progress.json", {"selection": selection, "runs": results})
    # Average across models within each shared game seed before computing intervals.
    aggregate = {
        field: paired_interval(
            np.stack([delta[field] for delta in differences_by_model]).mean(axis=0)
        )
        for field in differences_by_model[0]
    }
    result = {
        "status": "complete",
        "selection": selection,
        "runs": results,
        "aggregate_paired": aggregate,
        "interval_note": (
            "Normal 95% intervals conditional on three fixed models; paired by game seed."
        ),
    }
    atomic_json(output / "results.json", result)
    return result
