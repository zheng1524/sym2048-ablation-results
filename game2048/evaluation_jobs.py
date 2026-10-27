"""Shared, teacher-free subprocess evaluation and completion-cache validation."""

import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from game2048.config import INFERENCE_VERSIONS, PlannerConfig


def planner_arguments(config: PlannerConfig) -> list[str]:
    return [
        argument
        for field, value in asdict(config).items()
        for argument in ("--" + field.replace("_", "-"), str(value))
    ]


def cached_evaluation(
    directory: Path, checkpoint: Path, mode: str, games: int, seed: int, planner: PlannerConfig
) -> dict | None:
    """Reuse only complete results produced by these weights and evaluation settings."""
    try:
        metrics = json.loads((directory / "metrics.json").read_text())
        expected = {
            "mode": mode,
            "algorithm_version": INFERENCE_VERSIONS[mode],
            "games": games,
            "seed_start": seed,
            "checkpoint_sha256": hashlib.sha256(
                (checkpoint / "params.msgpack").read_bytes()
            ).hexdigest(),
        }
        if any(metrics.get(k) != v for k, v in expected.items()):
            return None
        if mode != "greedy" and metrics.get("planner") != asdict(planner):
            return None
        if not isinstance(metrics.get("checkpoint_metadata"), dict):
            return None
        with (directory / "games.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != games or [int(row["seed"]) for row in rows] != list(
            range(seed, seed + games)
        ):
            return None
        for row in rows:
            # A truncated final CSV row can have all column names but missing values.
            for field in ("max_exponent", "max_tile", "tile_sum", "moves"):
                if int(row[field]) < 0:
                    return None
        return metrics
    except (OSError, ValueError, KeyError, TypeError, csv.Error):
        return None


def evaluate_checkpoint(checkpoint, destination, mode, games, seed, planner, log_directory):
    cached = cached_evaluation(destination, checkpoint, mode, games, seed, planner)
    if cached is not None:
        return cached
    batches = (256,) if mode == "greedy" else (4, 2, 1)
    for batch in batches:
        arguments = [
            "evaluate",
            str(checkpoint),
            str(destination),
            "--mode",
            mode,
            "--games",
            str(games),
            "--seed",
            str(seed),
            "--batch-size",
            str(batch),
            *planner_arguments(planner),
        ]
        log_path = log_directory / f"{destination.parent.name}-{destination.name}-batch{batch}.log"
        print(json.dumps({"stage": str(destination), "arguments": arguments}), flush=True)
        with log_path.open("w") as stream:
            completed = subprocess.run(
                [sys.executable, "-m", "game2048", *arguments],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode == 0:
            result = cached_evaluation(destination, checkpoint, mode, games, seed, planner)
            if result is None:
                raise RuntimeError(f"Incomplete evaluation artifacts: {destination}")
            return result
        log = log_path.read_text().lower()
        if not any(marker in log for marker in ("resource_exhausted", "out of memory")):
            raise RuntimeError(f"Evaluation failed; see {log_path}")
    raise RuntimeError(f"Evaluation exhausted memory even at batch size 1: {destination}")
