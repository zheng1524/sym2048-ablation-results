"""Resumable experiment orchestration; each stage gets a fresh GPU process."""

import json
import subprocess
import sys
from pathlib import Path

from game2048.artifacts import atomic_json
from game2048.config import INFERENCE_VERSIONS, PlannerConfig
from game2048.evaluation_jobs import evaluate_checkpoint, planner_arguments


def run_experiment(teacher: Path, output: Path, seeds: list[int], smoke: bool = False) -> dict:
    from game2048.teacher import Teacher

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least one training seed, with no duplicates")
    verified = Teacher(teacher)
    if verified.size != 3:
        raise ValueError("Experiments require the complete 3x3 teacher")
    output.mkdir(parents=True, exist_ok=True)
    test_games, test_seed = (16 if smoke else 2000), 100_000
    specification = {
        "inference_versions": INFERENCE_VERSIONS,
        "teacher_sha256": verified.manifest["content_sha256"],
        "seeds": seeds,
        "smoke": smoke,
        "test_seed": test_seed,
        "test_games": test_games,
        "validation_seed": 10_000,
        "validation_games": 16 if smoke else 512,
    }
    manifest = output / "experiment.json"
    if manifest.exists() and json.loads(manifest.read_text()) != specification:
        raise ValueError("Existing experiment has a different specification; use a new directory")
    atomic_json(manifest, specification)

    def stage(name, arguments):
        print(json.dumps({"stage": name, "arguments": arguments}), flush=True)
        with (output / f"{name}.log").open("a") as log:
            subprocess.run(
                [sys.executable, "-m", "game2048", *arguments],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def evaluation_stage(checkpoint, destination, mode, planner_config):
        return evaluate_checkpoint(
            checkpoint, destination, mode, test_games, test_seed, planner_config, output
        )

    results = []
    for seed in seeds:
        run = output / f"seed{seed}"
        train = run / "supervised"
        if not (train / "complete.json").exists():
            args = ["train", str(teacher), str(train), "--seed", str(seed)]
            if smoke:
                args += [
                    "--updates",
                    "20",
                    "--warmup",
                    "5",
                    "--validate-every",
                    "10",
                    "--validation-games",
                    "16",
                    "--batch-size",
                    "64",
                    "--width",
                    "8",
                    "--blocks",
                    "1",
                    "--collect-every",
                    "5",
                ]
            if (train / "training.msgpack").exists():
                args += ["--resume"]
            stage(f"seed{seed}-train", args)
        frozen = train / "best"
        planner = (
            PlannerConfig(
                horizon=4, candidates=8, scenarios=2, iterations=2, verification_scenarios=4
            )
            if smoke
            else PlannerConfig()
        )
        for mode in ("greedy", "shooting", "mppi"):
            destination = run / mode
            metrics = evaluation_stage(frozen, destination, mode, planner)
            results.append({"training_seed": seed, "variant": mode, **metrics})

        td = run / "td"
        if not (td / "complete.json").exists():
            args = ["finetune", str(frozen), str(td), "--seed", str(seed)]
            if smoke:
                args += [
                    "--updates",
                    "2",
                    "--envs",
                    "8",
                    "--horizon",
                    "4",
                    "--validate-every",
                    "1",
                    "--validation-games",
                    "16",
                ]
            if (td / "training.msgpack").exists():
                args += ["--resume"]
            stage(f"seed{seed}-td", args)
        destination = run / "td-greedy"
        metrics = evaluation_stage(td / "best", destination, "greedy", PlannerConfig())
        results.append({"training_seed": seed, "variant": "td-greedy", **metrics})
        if not (run / "replay.json").exists():
            stage(
                f"seed{seed}-replay",
                [
                    "replay",
                    str(frozen),
                    str(run / "replay.json"),
                    "--mode",
                    "mppi",
                    *planner_arguments(planner),
                ],
            )
        atomic_json(output / "results.json", {"specification": specification, "runs": results})
    return {"output": str(output), "runs": len(results), "smoke": smoke}
