"""Run the frozen formal protocol, collect real artifacts, and create a local delivery commit.

No network writes or credential use. A detached invocation survives the chat turn.
On failure, preserve resumable state and publish an explicitly incomplete report.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "6eb6db44f0a3d70c15b92b5200404bb2b033d686"
GROUPS = ("d4_distill_td", "d4_scratch_td", "plain_distill_td")


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def collect(output, delivery, status):
    """Copy only compact views, best models and completed evaluation CSVs."""
    delivery.mkdir(parents=True, exist_ok=True)
    status["disk_free_bytes"] = shutil.disk_usage(ROOT).free
    (delivery / "results").mkdir(exist_ok=True)
    summary = read(output / "summary.json", {})
    protocol = read(output / "protocol.json", {})
    stages = []
    for group in (*GROUPS, "d4_scratch_practical"):
        for seed in (0, 1, 2):
            root = output / group / f"seed{seed}"
            for phase in ("supervised", "td"):
                source = root / phase
                info = read(source / "training.json")
                if info is None:
                    continue
                stages.append(
                    {
                        "group": group,
                        "seed": seed,
                        "phase": phase,
                        "updates": info["step"],
                        "best_step": info["best_step"],
                        "best_validation": info["best_score"],
                        "seconds": info["elapsed_seconds"],
                        "environment_transitions": info["environment_transitions"],
                        "complete": (source / "complete.json").exists(),
                        "observed_step": read(source / "progress.json", {}).get(
                            "step", info["step"]
                        ),
                    }
                )
                target = delivery / "results" / group / f"seed{seed}" / phase
                target.mkdir(parents=True, exist_ok=True)
                for name in ("training.json", "complete.json", "request.json", "interrupted.json"):
                    if (source / name).exists():
                        shutil.copyfile(source / name, target / name)
                if (source / "best" / "model.json").exists():
                    checkpoint = delivery / "checkpoints" / group / f"seed{seed}" / phase
                    checkpoint.mkdir(parents=True, exist_ok=True)
                    for name in ("model.json", "params.msgpack"):
                        shutil.copyfile(source / "best" / name, checkpoint / name)
            if (root / "test" / "metrics.json").exists():
                target = delivery / "results" / group / f"seed{seed}" / "test"
                target.mkdir(parents=True, exist_ok=True)
                for name in ("games.csv", "metrics.json", "request.json"):
                    shutil.copyfile(root / "test" / name, target / name)
    for name in (
        "summary.json",
        "frozen.json",
        "experiment.json",
        "protocol.json",
        "stage-timings.json",
    ):
        if (output / name).exists():
            shutil.copyfile(output / name, delivery / "results" / name)
    selection = read(output / "practical" / "selection.json", {})
    if selection:
        target = delivery / "results" / "practical"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output / "practical" / "selection.json", target / "selection.json")
        for path in (output / "practical").glob("pilot-*/td/training.json"):
            shutil.copyfile(path, target / f"{path.parent.parent.name}-training.json")
    evidence = delivery / "evidence"
    evidence.mkdir(exist_ok=True)
    for path in (output / "stage-logs").glob("*.txt"):
        shutil.copyfile(path, evidence / path.name)
    core_complete = bool(summary.get("formal_core_complete"))
    practical_complete = bool(selection) and (
        not selection.get("required")
        or all(
            (output / "d4_scratch_practical" / f"seed{s}" / "test" / "metrics.json").exists()
            for s in (0, 1, 2)
        )
    )
    status.update(core_complete=core_complete, practical_complete=practical_complete, stages=stages)
    atomic(delivery / "JOB_STATUS.json", status)
    lines = [
        "# Execution report: distillation and D₄ ablation",
        "",
        f"Status: **{status['status']}**. Formal core complete: {core_complete}. "
        f"Practical sufficiency diagnosis complete: {practical_complete}.",
        "",
        f"Base: `{BASE}`. Branch: `experiment/distillation-d4-ablation-20261031`.",
        f"Latest free disk: {status['disk_free_bytes'] / (1024**3):.3f} GiB.",
        "Main and all baseline tracked files are preserved. GitHub connector and local "
        "credential both report push=false: **not uploaded; no PR created**.",
        "",
        "Hardware: local RTX 3060 Laptop 6GB, Ryzen 7 6800H, 14GiB RAM. "
        "JAX 0.12.0.1 uses CUDA 12 plugins and cuDNN 9.10.2.21 in the isolated uv environment; "
        "pyproject.toml and uv.lock are unchanged. Original recorded 4090 SSH endpoints "
        "closed during handshake. See evidence/ and PROTOCOL.md for checks and runtime details.",
        "",
        "The new complete teacher was built and fully verified locally: 41,325,017 "
        "states, Bellman residual 0, SHA256 "
        "`6a66f3b52bd5bef3e270c777e4da063cae7e7038a0c5c858eb7bb7274466fc8e`. "
        "Its duplicate intermediates were byte-verified, archived, and kept locally.",
        "",
        "Architectures: original D₄ 32/2/64 (304,833 parameters); plain CNN 83/2/64 "
        "(305,403; +0.187%). Plain uses spatial flatten and ordinary convolutions. "
        "Both receive uniformly oriented scalar afterstates after orbit holdout masking. "
        "A/B initialization hashes are recorded per seed; tests make teacher construction "
        "and lookup throw while B still trains. Recovery is compared with continuous training.",
        "",
        "Strict A/B matches TD algorithm and 10000-update online budget. A also incurs "
        "100000 supervised updates and teacher cost. A/C matches parameter counts and "
        "training budgets; it does not match execution time, FLOPs, or student trajectories.",
        "",
        "| Group | Seed | Phase | Actual updates | Best step | Validation best | Seconds |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for s in stages:
        score = "n/a" if s["best_validation"] is None else f"{s['best_validation']:.6f}"
        lines.append(
            f"| {s['group']} | {s['seed']} | {s['phase']} | {s['updates']} | "
            f"{s['best_step']} | {score} | {s['seconds']:.1f} |"
        )
    if core_complete:
        lines.extend(
            [
                "",
                "Formal greedy test results (frozen validation-selected models):",
                "",
                "| Group | Training seed | Mean max exponent | 256 | 512 | 1024 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for group, runs in summary["runs"].items():
            for r in runs:
                rates = r["reach_rates"]
                lines.append(
                    f"| {group} | {r['training_seed']} | {r['mean_max_exponent']:.6f} | "
                    f"{rates['256']:.2%} | {rates['512']:.2%} | {rates['1024']:.2%} |"
                )
        lines.extend(["", "Paired A-minus-control differences:", ""])
        for name, comparison in summary["comparisons"].items():
            ci = comparison["paired_game_intervals"]["max_exponent"]
            lines.append(
                f"- {name}: {comparison['mean_across_training_seeds']:.6f}; "
                f"paired game 95% interval [{ci['lower']:.6f}, {ci['upper']:.6f}]; "
                f"training-seed SD {comparison['sd_across_training_seeds']}; "
                f"seed differences {comparison['per_training_seed_mean_max_exponent_difference']}."
            )
        lines.extend(
            [
                "",
                "Intervals describe game randomness conditional on the fitted models; "
                "they do not cover training randomness. Summary JSON contains secondary "
                "metrics, per-model means and training-seed dispersion. Equal game seeds "
                "do not imply identical trajectories.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "**No completed formal core comparison is available.** Partial curves, "
                "smoke scores and historical scores are not method conclusions.",
            ]
        )
    if not practical_complete:
        lines.extend(
            [
                "",
                "Scratch-training sufficiency has not yet been established; "
                "a weak strict-budget B cannot establish failure to learn or convergence.",
            ]
        )
    if status.get("error"):
        lines.extend(["", "Execution interruption: `" + status["error"].replace("`", "'") + "`."])
    lines.extend(
        [
            "",
            "Checks: baseline 29 tests passed. New correctness tests cover factories, "
            "parameter counts, plain symmetry freedom, legacy loading, legal masks, "
            "analytic boundary, holdout/augmentation, teacher-free TD, and exact TD and "
            "supervision/pool recovery. Actual final commands and outcomes are in "
            "`evidence/checks-*.txt` and `evidence/checks-gpu-pytest-final.txt`; do not infer "
            "success from this description. Smoke is independently labeled "
            "and is not formal evidence.",
            "",
            "Artifacts: results/ contains complete CSVs and validation curves; checkpoints/ "
            "contains only best weights and metadata. Recovery payloads, teacher and caches "
            "remain under ignored new directories. `git diff --name-status "
            + BASE
            + "` is the additive-change audit. A local bundle is supplied "
            "because push is unavailable.",
            "",
            f"Frozen protocol: `{json.dumps(protocol, ensure_ascii=False, sort_keys=True)}`",
            "",
        ]
    )
    (delivery / "REPORT.md").write_text("\n".join(lines))
    (delivery / "PR_BODY.md").write_text(
        "Adds isolated distillation-versus-scratch TD and D4-versus-plain-CNN experiments. "
        "The original model, commands, rules and historical artifacts are preserved.\n\n"
        f"Execution status: {status['status']}; formal core complete: {core_complete}; "
        f"practical sufficiency diagnosis complete: {practical_complete}. "
        "REPORT.md and results/ contain the actual scope and paired statistics. "
        "Smoke is separately labeled. Ruff, formatting, ty and CPU/GPU pytest evidence "
        "is in evidence/. No merge is requested.\n"
    )
    hashes = {
        str(p.relative_to(delivery)): hashlib.sha256(p.read_bytes()).hexdigest()
        for prefix in ("results", "checkpoints")
        for p in (delivery / prefix).rglob("*")
        if p.is_file()
    }
    if status["status"] == "complete":
        for path in (delivery / "checkpoints").rglob("model.json"):
            expected = read(path)["params_sha256"]
            if (
                hashlib.sha256((path.parent / "params.msgpack").read_bytes()).hexdigest()
                != expected
            ):
                raise ValueError(f"Copied checkpoint hash mismatch: {path}")
        for path in (delivery / "results").rglob("metrics.json"):
            if (
                hashlib.sha256((path.parent / "games.csv").read_bytes()).hexdigest()
                != read(path)["csv_sha256"]
            ):
                raise ValueError(f"Copied evaluation hash mismatch: {path}")
    atomic(delivery / "ARTIFACTS_SHA256.json", hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, default=ROOT / "docs/ablations/20261031")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    status = {
        "status": "running",
        "pid": os.getpid(),
        "started_utc": datetime.now(UTC).isoformat(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "output": str(args.output),
        "uploaded": False,
    }
    if args.collect_only:
        status = read(args.delivery / "JOB_STATUS.json", status)
    atomic(args.delivery / "JOB_STATUS.json", status)
    error = None
    if not args.collect_only:
        log = args.output.with_name(args.output.name + ".workflow.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log.open("a") as stream:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "game2048.ablations",
                        "run",
                        "--teacher",
                        str(args.teacher),
                        "--output",
                        str(args.output),
                        "--seeds",
                        "0",
                        "1",
                        "2",
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                while child.poll() is None:
                    status["updated_utc"] = datetime.now(UTC).isoformat()
                    collect(args.output, args.delivery, status)
                    time.sleep(30)
                if child.returncode:
                    raise subprocess.CalledProcessError(child.returncode, child.args)
            status["status"] = "complete"
        except BaseException as exc:
            error = exc
            status.update(status="interrupted", error=str(exc))
            (args.delivery / "evidence" / "job-exception.txt").write_text(traceback.format_exc())
    else:
        status["status"] = "running" if not (args.output / "summary.json").exists() else "collected"
    status["updated_utc"] = datetime.now(UTC).isoformat()
    collect(args.output, args.delivery, status)
    if not args.collect_only:
        subprocess.run(["git", "add", "--", str(args.delivery)], check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Codex",
                "-c",
                "user.email=codex@openai.com",
                "commit",
                "-m",
                f"Record ablation execution: {status['status']}",
            ],
            check=True,
        )
        bundle = ROOT.parent / "sym2048-ablation-20261031.bundle"
        subprocess.run(
            [
                "git",
                "bundle",
                "create",
                str(bundle),
                "experiment/distillation-d4-ablation-20261031",
            ],
            check=True,
        )
        subprocess.run(["git", "bundle", "verify", str(bundle)], check=True)
    if error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
