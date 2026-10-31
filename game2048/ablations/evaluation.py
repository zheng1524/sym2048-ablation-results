"""Architecture-aware greedy games; adapted from evaluation.make_game_runner/evaluate."""

import csv
import hashlib
import json
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from game2048 import jax_env
from game2048.ablations.config import VERSION, ModelSpec
from game2048.ablations.models import load_model, make_model, parameter_hash
from game2048.ablations.provenance import claim, source_identity
from game2048.artifacts import atomic_json
from game2048.comparison import read_games
from game2048.network import greedy_action


@lru_cache(maxsize=16)
def make_game_runner(spec: ModelSpec, version: str = VERSION):
    if version != VERSION:
        raise ValueError("Unsupported inference version")
    model = make_model(spec)

    def run(params, seeds):
        env_keys = jax.vmap(jax.random.PRNGKey)(seeds)
        action_keys = jax.vmap(lambda k: jax.random.fold_in(k, 1_234_567))(env_keys)
        boards = jax.vmap(jax_env.reset)(env_keys)
        alive = jnp.ones(len(seeds), jnp.bool_)
        moves = jnp.zeros(len(seeds), jnp.int32)

        def condition(carry):
            _, active, _, t = carry
            return jnp.any(active) & (t < 2048)

        def advance(carry):
            current, active, counts, t = carry
            ties = jax.vmap(lambda k: jax.random.fold_in(k, t))(action_keys)
            tiles = jax.vmap(lambda k: jax.random.fold_in(k, t + 1))(env_keys)
            actions = jax.vmap(lambda b, k: greedy_action(model, params, b, k))(current, ties)
            nxt, _, done = jax.vmap(jax_env.step)(current, actions, tiles)
            return (
                jnp.where(active[:, None, None], nxt, current),
                active & ~done,
                counts + active,
                t + 1,
            )

        boards, alive, moves, _ = jax.lax.while_loop(
            condition,
            advance,
            (boards, alive, moves, jnp.int32(0)),
        )
        return jnp.max(boards, axis=(1, 2)), jax.vmap(jax_env.tile_sum)(boards), moves, alive

    return jax.jit(run)


def summarize(records):
    levels = np.array([r["max_exponent"] for r in records], np.float64)
    return {
        "mean_max_exponent": float(levels.mean()),
        "mean_max_tile": float(np.mean([r["max_tile"] for r in records])),
        "mean_tile_sum": float(np.mean([r["tile_sum"] for r in records])),
        "mean_moves": float(np.mean([r["moves"] for r in records])),
        "standard_error_max_exponent": float(levels.std(ddof=1) / np.sqrt(len(levels)))
        if len(levels) > 1
        else 0.0,
        "reach_rates": {str(1 << k): float(np.mean(levels >= k)) for k in (8, 9, 10)},
    }


def validate_records(records, games, seed):
    if len(records) != games or [r["seed"] for r in records] != list(range(seed, seed + games)):
        raise ValueError("Incomplete, duplicated, or unexpected game seeds")
    for row in records:
        if not 1 <= row["max_exponent"] <= 10 or row["max_tile"] != 1 << row["max_exponent"]:
            raise ValueError("Invalid tile/exponent correspondence")
        if row["moves"] < 1 or row["tile_sum"] < row["max_tile"]:
            raise ValueError("Invalid terminal game record")


def evaluate(spec, params, *, games, seed, batch_size=256, output=None, smoke=False):
    if min(games, batch_size) < 1:
        raise ValueError("Evaluation counts must be positive")
    identity = source_identity()
    request = {
        "algorithm_version": VERSION,
        "mode": "greedy",
        "model": asdict(spec),
        "params_sha256": parameter_hash(params),
        "games": games,
        "seed_start": seed,
        "batch_size": batch_size,
        "smoke": smoke,
        "source_sha256": identity["source_sha256"],
        "runtime": identity["runtime"],
    }
    if output is not None:
        claim(output, "request.json", request)
        if (output / "metrics.json").exists():
            return verify_evaluation(output, request)
        atomic_json(output / "progress.json", {"status": "compiling", "games": 0})
    runner = make_game_runner(spec)
    seeds = np.arange(seed, seed + games, dtype=np.uint32)
    started = time.monotonic()
    compiled = runner.lower(params, jnp.asarray(np.resize(seeds, batch_size))).compile()
    compile_seconds = time.monotonic() - started
    records = []
    started = time.monotonic()
    if output is not None:
        atomic_json(output / "progress.json", {"status": "running", "games": 0})
    try:
        for offset in range(0, games, batch_size):
            selected = seeds[offset : offset + batch_size]
            values = jax.device_get(compiled(params, jnp.asarray(np.resize(selected, batch_size))))
            exponents, totals, moves, alive = [np.asarray(x)[: len(selected)] for x in values]
            if alive.any():
                raise RuntimeError("Evaluation exceeded conservative episode bound")
            records.extend(
                {
                    "seed": int(s),
                    "max_exponent": int(e),
                    "max_tile": 1 << int(e),
                    "tile_sum": int(total),
                    "moves": int(count),
                }
                for s, e, total, count in zip(selected, exponents, totals, moves, strict=True)
            )
            if output is not None:
                atomic_json(output / "progress.json", {"status": "running", "games": len(records)})
        validate_records(records, games, seed)
        metrics = {
            **request,
            **summarize(records),
            "compile_seconds": compile_seconds,
            "execution_seconds": time.monotonic() - started,
        }
        if output is not None:
            temporary = output / "games.csv.tmp"
            with temporary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            temporary.replace(output / "games.csv")
            metrics["csv_sha256"] = hashlib.sha256((output / "games.csv").read_bytes()).hexdigest()
            atomic_json(output / "metrics.json", metrics)
            atomic_json(output / "progress.json", {"status": "complete", "games": games})
        return metrics
    except BaseException as error:
        if output is not None:
            atomic_json(output / "interrupted.json", {"games": len(records), "error": str(error)})
        raise


def verify_evaluation(directory: Path, expected=None):
    request = json.loads((directory / "request.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    if expected is not None and expected != request:
        raise ValueError("Evaluation request mismatch")
    if any(metrics.get(k) != v for k, v in request.items()):
        raise ValueError("Evaluation metadata mismatch")
    digest = hashlib.sha256((directory / "games.csv").read_bytes()).hexdigest()
    if metrics.get("csv_sha256") != digest:
        raise ValueError("CSV hash mismatch")
    records = read_games(directory)
    validate_records(records, request["games"], request["seed_start"])
    if any(metrics.get(k) != v for k, v in summarize(records).items()):
        raise ValueError("Evaluation summary disagrees with CSV")
    return metrics


def evaluate_checkpoint(checkpoint, output, *, games, seed, batch_size, smoke):
    _, params, _, spec = load_model(checkpoint)
    return evaluate(
        spec, params, games=games, seed=seed, batch_size=batch_size, output=output, smoke=smoke
    )
