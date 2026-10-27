"""Command line entry points; teacher commands do not import JAX."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from game2048.config import INFERENCE_VERSIONS, PlannerConfig


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3x3 2048 exact-teacher neural agent")
    commands = parser.add_subparsers(dest="command", required=True)
    teacher = commands.add_parser("teacher")
    teacher.add_argument("operation", choices=("build", "verify", "inspect"))
    teacher.add_argument("directory", type=Path)
    teacher.add_argument("--size", type=int, default=3, choices=(2, 3))
    teacher.add_argument("--threads", type=int, default=16)
    teacher.add_argument("--chunk", type=int, default=8192)

    train = commands.add_parser("train")
    train.add_argument("teacher", type=Path)
    train.add_argument("output", type=Path)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--updates", type=int, default=100_000)
    train.add_argument("--batch-size", type=int, default=1024)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--width", type=int, default=32)
    train.add_argument("--blocks", type=int, default=2)
    train.add_argument("--warmup", type=int, default=5000)
    train.add_argument("--validate-every", type=int, default=5000)
    train.add_argument("--validation-games", type=int, default=512)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--collect-every", type=int, default=1000)
    train.add_argument("--resume", action="store_true")

    td = commands.add_parser("finetune")
    td.add_argument("checkpoint", type=Path)
    td.add_argument("output", type=Path)
    td.add_argument("--seed", type=int, default=0)
    td.add_argument("--updates", type=int, default=10_000)
    td.add_argument("--trace-lambda", type=float, default=0.8)
    td.add_argument("--learning-rate", type=float, default=3e-5)
    td.add_argument("--envs", type=int, default=128)
    td.add_argument("--horizon", type=int, default=32)
    td.add_argument("--validate-every", type=int, default=1000)
    td.add_argument("--validation-games", type=int, default=512)
    td.add_argument("--resume", action="store_true")

    experiment = commands.add_parser("experiment")
    experiment.add_argument("teacher", type=Path)
    experiment.add_argument("output", type=Path)
    experiment.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    experiment.add_argument("--smoke", action="store_true")

    comparison = commands.add_parser("compare-planners")
    comparison.add_argument("experiment", type=Path)
    comparison.add_argument("output", type=Path)

    planner_defaults = PlannerConfig()
    for name in ("evaluate", "replay"):
        sub = commands.add_parser(name)
        sub.add_argument("checkpoint", type=Path)
        sub.add_argument("output", type=Path)
        sub.add_argument("--mode", default="greedy", choices=tuple(INFERENCE_VERSIONS))
        sub.add_argument("--seed", type=int, default=100_000)
        if name == "evaluate":
            sub.add_argument("--games", type=int, default=2000)
            sub.add_argument("--batch-size", type=int)
        for field, default in asdict(planner_defaults).items():
            sub.add_argument("--" + field.replace("_", "-"), type=type(default), default=default)

    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "teacher":
        from game2048.teacher import build, verify

        if args.operation == "build":
            build(args.directory, args.size, args.threads, args.chunk)
            result = verify(args.directory, threads=args.threads)
        elif args.operation == "verify":
            result = verify(args.directory, threads=args.threads)
        else:
            result = json.loads((args.directory / "manifest.json").read_text())
    elif args.command == "train":
        from game2048.network import ModelConfig
        from game2048.training import TrainConfig, train

        config = TrainConfig(
            seed=args.seed,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_updates=args.updates,
            warmup_updates=args.warmup,
            validation_every=args.validate_every,
            validation_games=args.validation_games,
            patience=args.patience,
            collect_every=args.collect_every,
        )
        result = train(
            args.teacher,
            args.output,
            config,
            ModelConfig(width=args.width, blocks=args.blocks),
            args.resume,
        )
    elif args.command == "finetune":
        from game2048.training import finetune

        result = finetune(
            args.checkpoint,
            args.output,
            updates=args.updates,
            seed=args.seed,
            trace_lambda=args.trace_lambda,
            learning_rate=args.learning_rate,
            envs=args.envs,
            horizon=args.horizon,
            validation_every=args.validate_every,
            validation_games=args.validation_games,
            resume=args.resume,
        )
    elif args.command == "experiment":
        from game2048.experiments import run_experiment

        result = run_experiment(args.teacher, args.output, args.seeds, args.smoke)
    elif args.command == "compare-planners":
        from game2048.comparison import compare_planners

        result = compare_planners(args.experiment, args.output)
    else:
        from game2048.evaluation import evaluate, replay
        from game2048.network import load_model

        model, params, metadata = load_model(args.checkpoint)
        planner = PlannerConfig(
            **{field: getattr(args, field) for field in asdict(PlannerConfig())}
        )
        if args.command == "evaluate":
            checkpoint_info = {
                "checkpoint_sha256": hashlib.sha256(
                    (args.checkpoint / "params.msgpack").read_bytes()
                ).hexdigest(),
                "checkpoint_metadata": metadata,
            }
            result = evaluate(
                model,
                params,
                games=args.games,
                seed=args.seed,
                mode=args.mode,
                planner=planner,
                batch_size=args.batch_size,
                output=args.output,
                checkpoint_info=checkpoint_info,
            )
        else:
            replay(
                model, params, seed=args.seed, mode=args.mode, planner=planner, output=args.output
            )
            result = {"replay": str(args.output)}
    print(json.dumps(result, indent=2, ensure_ascii=False))
