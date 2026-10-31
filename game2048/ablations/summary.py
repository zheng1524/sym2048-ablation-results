"""Paired game uncertainty, separately from the three training-seed replicates."""

import json

import numpy as np

from game2048.ablations.config import GROUPS, Protocol
from game2048.ablations.evaluation import verify_evaluation
from game2048.ablations.models import load_model, parameter_hash
from game2048.artifacts import atomic_json
from game2048.comparison import paired_differences, paired_interval, read_games


def summarize_experiment(output, groups=GROUPS):
    experiment = json.loads((output / "experiment.json").read_text())
    protocol = experiment["protocol"]
    seeds = protocol["seeds"]
    data, metrics = {}, {}
    for group in groups:
        data[group], metrics[group] = {}, []
        for seed in seeds:
            run = output / group / f"seed{seed}"
            marker = json.loads((run / "td" / "complete.json").read_text())
            request = json.loads((run / "td" / "request.json").read_text())
            expected_protocol = protocol
            if group == "d4_scratch_practical":
                expected_protocol = json.loads(
                    (output / "practical" / "selection.json").read_text()
                )["protocol"]
            if (
                request["protocol"] != expected_protocol
                or request["source"] != experiment["source"]
                or request["group"] != group
                or request["seed"] != seed
                or marker["request"] != request
                or marker["updates"] != expected_protocol["td"]["updates"]
            ):
                raise ValueError("Completed training does not match its experiment protocol")
            _, params, _, _ = load_model(run / "td" / "best")
            result = verify_evaluation(run / "test")
            if result["params_sha256"] != parameter_hash(params) or marker[
                "best_params_sha256"
            ] != parameter_hash(params):
                raise ValueError("Evaluation does not correspond to the frozen TD best")
            if (
                result["games"] != protocol["test_games"]
                or result["seed_start"] != protocol["test_seed"]
                or result["smoke"] != protocol["smoke"]
            ):
                raise ValueError("Evaluation uses the wrong protocol")
            if (
                result["source_sha256"] != experiment["source"]["source_sha256"]
                or result["runtime"] != experiment["source"]["runtime"]
            ):
                raise ValueError("Evaluation source or numerical runtime mismatch")
            data[group][seed] = read_games(run / "test")
            metrics[group].append({"training_seed": seed, **result})
    comparisons = {}
    if GROUPS[0] in groups:
        for other in (GROUPS[1], GROUPS[2], "d4_scratch_practical"):
            if other not in groups:
                continue
            diffs = [paired_differences(data[other][s], data[GROUPS[0]][s]) for s in seeds]
            fields = diffs[0].keys()
            seed_means = [float(d["max_exponent"].mean()) for d in diffs]
            comparisons[f"{GROUPS[0]}_minus_{other}"] = {
                "per_training_seed_mean_max_exponent_difference": dict(
                    zip(map(str, seeds), seed_means, strict=True)
                ),
                "mean_across_training_seeds": float(np.mean(seed_means)),
                "sd_across_training_seeds": float(np.std(seed_means, ddof=1))
                if len(seeds) > 1
                else None,
                "paired_game_intervals": {
                    field: paired_interval(np.stack([d[field] for d in diffs]).mean(axis=0))
                    for field in fields
                },
                "interval_scope": "Game randomness conditional on these fitted models; "
                "does not cover training randomness. Normal approximation.",
            }
    aggregates = {}
    for group, results in metrics.items():
        means = [r["mean_max_exponent"] for r in results]
        aggregates[group] = {
            "mean_max_exponent": float(np.mean(means)),
            "training_seed_sd": float(np.std(means, ddof=1)) if len(means) > 1 else None,
        }
    initialization_audit = {}
    if all(g in groups for g in GROUPS[:2]):
        for seed in seeds:
            a = json.loads(
                (output / GROUPS[0] / f"seed{seed}" / "supervised/request.json").read_text()
            )
            b = json.loads((output / GROUPS[1] / f"seed{seed}" / "td/request.json").read_text())
            if (
                a["initial_params_sha256"] != b["initial_params_sha256"]
                or b["upstream_sha256"] is not None
                or "teacher_sha256" in b
            ):
                raise ValueError("A/B initialization or teacher-free scratch audit failed")
            initialization_audit[str(seed)] = {
                "shared_initial_params_sha256": a["initial_params_sha256"],
                "B_upstream_sha256": None,
            }
    result = {
        "smoke": protocol["smoke"],
        "runs": metrics,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "initialization_audit": initialization_audit,
        "formal_core_complete": not protocol["smoke"]
        and all(g in groups for g in GROUPS)
        and protocol == Protocol().json(),
    }
    atomic_json(output / "summary.json", result)
    return result
