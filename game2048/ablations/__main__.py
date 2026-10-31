"""Isolated entry point: python -m game2048.ablations --help."""

import argparse
import json
from pathlib import Path


def main():
    from game2048.ablations.config import GROUPS, from_json, model_spec, protocol

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--teacher", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    run.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    run.add_argument("--smoke", action="store_true")
    stage = sub.add_parser("stage")
    stage.add_argument("--phase", choices=("supervised", "td"), required=True)
    stage.add_argument("--teacher", type=Path)
    stage.add_argument("--upstream", type=Path)
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument(
        "--group", choices=(*GROUPS, "d4_scratch_practical", "lr_pilot"), required=True
    )
    stage.add_argument("--seed", type=int, required=True)
    stage.add_argument("--protocol-file", type=Path, required=True)
    stage.add_argument("--resume", action="store_true")
    stage.add_argument("--stop-after", type=int)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("checkpoint", type=Path)
    evaluation.add_argument("output", type=Path)
    evaluation.add_argument("--protocol-file", type=Path, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("output", type=Path)
    summary.add_argument("--groups", nargs="+", default=list(GROUPS))
    practical = sub.add_parser("practical")
    practical.add_argument("output", type=Path)
    practical.add_argument("--protocol-file", type=Path, required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("output", type=Path)
    audit.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.command == "run":
        from game2048.ablations.workflow import practical as diagnose
        from game2048.ablations.workflow import run as execute

        cfg = protocol(args.seeds, args.smoke)
        execute(args.teacher, args.output, cfg, args.groups)
        if not args.smoke and set(args.groups) == set(GROUPS):
            diagnose(args.output, cfg)
    elif args.command == "stage":
        cfg = from_json(json.loads(args.protocol_file.read_text()))
        if args.seed not in cfg.seeds:
            parser.error("Seed is absent from the frozen protocol")
        options = dict(
            group=args.group,
            seed=args.seed,
            spec=model_spec(args.group),
            protocol=cfg,
            resume=args.resume,
            stop_after=args.stop_after,
        )
        if args.phase == "supervised":
            from game2048.ablations.supervised import train

            if args.teacher is None:
                parser.error("Supervised training requires --teacher")
            train(args.teacher, args.output, **options)
        else:
            from game2048.ablations.td import train

            if args.teacher is not None:
                parser.error("TD stages do not accept teacher paths")
            train(args.output, upstream=args.upstream, **options)
    elif args.command == "evaluate":
        from game2048.ablations.evaluation import evaluate_checkpoint

        cfg = from_json(json.loads(args.protocol_file.read_text()))
        evaluate_checkpoint(
            args.checkpoint,
            args.output,
            games=cfg.test_games,
            seed=cfg.test_seed,
            batch_size=cfg.evaluation_batch,
            smoke=cfg.smoke,
        )
    elif args.command == "summarize":
        from game2048.ablations.summary import summarize_experiment

        summarize_experiment(args.output, args.groups)
    elif args.command == "practical":
        from game2048.ablations.workflow import practical as diagnose

        diagnose(args.output, from_json(json.loads(args.protocol_file.read_text())))
    elif args.command == "audit":
        from game2048.ablations.audit import audit

        audit(args.output, args.batch_size)


if __name__ == "__main__":
    main()
