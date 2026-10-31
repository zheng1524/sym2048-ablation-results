"""Fresh-process stages and validation-only practical scratch diagnostic."""

import json
import subprocess
import sys
import time
from dataclasses import asdict, replace

from game2048.ablations.config import GROUPS, model_spec
from game2048.ablations.provenance import claim, source_identity
from game2048.artifacts import atomic_json


def stage(root, name, arguments):
    logs = root / "stage-logs"
    logs.mkdir(exist_ok=True)
    print(json.dumps({"stage": name, "arguments": arguments}), flush=True)
    started = time.monotonic()
    returncode = None
    try:
        with (logs / f"{name}.txt").open("a") as stream:
            completed = subprocess.run(
                [sys.executable, "-m", "game2048.ablations", *arguments],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
            returncode = completed.returncode
            completed.check_returncode()
    finally:
        path = root / "stage-timings.json"
        timings = json.loads(path.read_text()) if path.exists() else {"attempts": []}
        timings["attempts"].append(
            {"stage": name, "seconds": time.monotonic() - started, "returncode": returncode}
        )
        atomic_json(path, timings)


def run(teacher, output, protocol, groups=GROUPS):
    if not groups or len(set(groups)) != len(groups) or any(g not in GROUPS for g in groups):
        raise ValueError("Choose unique core groups")
    specification = {
        "protocol": protocol.json(),
        "groups": list(groups),
        "source": source_identity(),
        "models": {g: asdict(model_spec(g)) for g in groups},
        "random_streams": "See state.STREAMS; initialization is split(PRNGKey(seed))[1]",
    }
    claim(output, "experiment.json", specification)
    protocol_file = output / "protocol.json"
    atomic_json(protocol_file, protocol.json())
    # B can run without teacher files and is not gated by supervised assets.
    ordered = [g for g in (GROUPS[1], GROUPS[0], GROUPS[2]) if g in groups]
    for group in ordered:
        for seed in protocol.seeds:
            root = output / group / f"seed{seed}"
            shared = ["--group", group, "--seed", str(seed), "--protocol-file", str(protocol_file)]
            upstream = None
            if group != GROUPS[1]:
                destination = root / "supervised"
                args = [
                    "stage",
                    "--phase",
                    "supervised",
                    "--teacher",
                    str(teacher),
                    "--output",
                    str(destination),
                    *shared,
                ]
                if (destination / "request.json").exists():
                    args += ["--resume"]
                stage(output, f"{group}-seed{seed}-supervised", args)
                upstream = destination / "best"
            destination = root / "td"
            args = ["stage", "--phase", "td", "--output", str(destination), *shared]
            if upstream is not None:
                args += ["--upstream", str(upstream)]
            if (destination / "request.json").exists():
                args += ["--resume"]
            stage(output, f"{group}-seed{seed}-td", args)
    # Freeze all core checkpoints before opening any test results.
    frozen = {}
    for group in groups:
        for seed in protocol.seeds:
            root = output / group / f"seed{seed}"
            done = json.loads((root / "td" / "complete.json").read_text())
            frozen[f"{group}/{seed}"] = done["best_params_sha256"]
    atomic_json(output / "frozen.json", frozen)
    for group in groups:
        for seed in protocol.seeds:
            root = output / group / f"seed{seed}"
            stage(
                output,
                f"{group}-seed{seed}-test",
                [
                    "evaluate",
                    str(root / "td" / "best"),
                    str(root / "test"),
                    "--protocol-file",
                    str(protocol_file),
                ],
            )
    stage(output, "summary", ["summarize", str(output), "--groups", *groups])


def practical(output, protocol):
    """Decide and tune exclusively on validation, before practical test evaluation."""
    if protocol.smoke:
        raise ValueError("Practical sufficiency diagnosis is a formal-only experiment")
    if tuple(protocol.seeds) != (0, 1, 2):
        raise ValueError("Practical diagnosis requires formal seeds 0,1,2")
    scores = {}
    for group in GROUPS[:2]:
        scores[group] = []
        for seed in protocol.seeds:
            info = json.loads((output / group / f"seed{seed}" / "td" / "training.json").read_text())
            if info["step"] != protocol.td.updates:
                raise ValueError("Complete the strict A/B validation curves before diagnosis")
            scores[group].append(info["best_score"])
    diagnostic = output / "practical"
    request = {
        "source": source_identity(),
        "protocol": protocol.json(),
        "validation_scores": scores,
    }
    claim(diagnostic, "request.json", request)
    if sum(scores[GROUPS[1]]) >= sum(scores[GROUPS[0]]):
        atomic_json(diagnostic / "selection.json", {"required": False, "validation_scores": scores})
        return
    candidates = []
    for rate in (3e-5, 1e-4, 3e-4):
        pilot_protocol = replace(
            protocol, seeds=(3,), td=replace(protocol.td, updates=2000, learning_rate=rate)
        )
        path = diagnostic / f"pilot-{rate:g}"
        path.mkdir(exist_ok=True)
        protocol_file = path / "protocol.json"
        atomic_json(protocol_file, pilot_protocol.json())
        args = [
            "stage",
            "--phase",
            "td",
            "--output",
            str(path / "td"),
            "--group",
            "lr_pilot",
            "--seed",
            "3",
            "--protocol-file",
            str(protocol_file),
        ]
        if (path / "td" / "request.json").exists():
            args += ["--resume"]
        stage(output, f"pilot-{rate:g}", args)
        info = json.loads((path / "td" / "training.json").read_text())
        candidates.append({"learning_rate": rate, "validation_score": info["best_score"]})
    chosen = min(candidates, key=lambda c: (-c["validation_score"], c["learning_rate"]))
    selected_protocol = replace(
        protocol, td=replace(protocol.td, updates=100_000, learning_rate=chosen["learning_rate"])
    )
    selection = {
        "required": True,
        "candidates": candidates,
        "chosen": chosen,
        "rule": "Maximum best validation mean_max_exponent; tie -> lower learning rate",
        "validation_scores": scores,
        "protocol": selected_protocol.json(),
    }
    marker = diagnostic / "selection.json"
    if marker.exists() and json.loads(marker.read_text()) != selection:
        raise ValueError("Practical learning rate was already frozen differently")
    atomic_json(marker, selection)
    selected_file = diagnostic / "protocol.json"
    atomic_json(selected_file, selected_protocol.json())
    for seed in protocol.seeds:
        path = output / "d4_scratch_practical" / f"seed{seed}" / "td"
        args = [
            "stage",
            "--phase",
            "td",
            "--output",
            str(path),
            "--seed",
            str(seed),
            "--group",
            "d4_scratch_practical",
            "--protocol-file",
            str(selected_file),
        ]
        if (path / "request.json").exists():
            args += ["--resume"]
        stage(output, f"practical-seed{seed}", args)
    # All learning-rate and checkpoint choices have finished before testing.
    for seed in protocol.seeds:
        root = output / "d4_scratch_practical" / f"seed{seed}"
        stage(
            output,
            f"practical-seed{seed}-test",
            [
                "evaluate",
                str(root / "td" / "best"),
                str(root / "test"),
                "--protocol-file",
                str(selected_file),
            ],
        )
    stage(
        output,
        "practical-summary",
        ["summarize", str(output), "--groups", *GROUPS, "d4_scratch_practical"],
    )
