"""Teacher-free evaluation with separate, seed-addressed environment/planner RNGs."""

import csv
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from game2048 import jax_env
from game2048.artifacts import atomic_json
from game2048.network import ModelConfig, ValueNetwork
from game2048.planner import PlannerConfig, make_actor


@lru_cache(maxsize=8)
def make_game_runner(config: ModelConfig, mode: str, planner: PlannerConfig):
    model = ValueNetwork(config)
    actor = make_actor(model, mode, planner)

    def run(params, seeds):
        env_keys = jax.vmap(jax.random.PRNGKey)(seeds)
        plan_keys = jax.vmap(lambda k: jax.random.fold_in(k, 1_234_567))(env_keys)
        boards = jax.vmap(jax_env.reset)(env_keys)
        alive = jnp.ones(len(seeds), dtype=jnp.bool_)
        moves = jnp.zeros(len(seeds), dtype=jnp.int32)

        def condition(carry):
            _, active, _, t = carry
            return jnp.any(active) & (t < 2048)

        def advance(carry):
            current, active, counts, t = carry
            action_keys = jax.vmap(lambda k: jax.random.fold_in(k, t))(plan_keys)
            tile_keys = jax.vmap(lambda k: jax.random.fold_in(k, t + 1))(env_keys)
            actions = jax.vmap(actor, in_axes=(None, 0, 0))(params, current, action_keys)
            nxt, _, done = jax.vmap(jax_env.step)(current, actions, tile_keys)
            return (
                jnp.where(active[:, None, None], nxt, current),
                active & ~done,
                counts + active,
                t + 1,
            )

        boards, alive, moves, _ = jax.lax.while_loop(
            condition, advance, (boards, alive, moves, jnp.int32(0))
        )
        return jnp.max(boards, axis=(1, 2)), jax.vmap(jax_env.tile_sum)(boards), moves, alive

    return jax.jit(run)


def evaluate(
    model: ValueNetwork,
    params,
    *,
    games: int = 2000,
    seed: int = 100_000,
    mode: str = "greedy",
    planner: PlannerConfig = PlannerConfig(),
    batch_size: int | None = None,
    output: Path | None = None,
    checkpoint_info: dict | None = None,
) -> dict:
    if games < 1 or (batch_size is not None and batch_size < 1):
        raise ValueError("Evaluation games and batch size must be positive")
    batch_size = batch_size if batch_size is not None else (256 if mode == "greedy" else 4)
    runner = make_game_runner(model.config, mode, planner)
    seeds = np.arange(seed, seed + games, dtype=np.uint32)
    example = jnp.asarray(np.resize(seeds, batch_size))
    started = time.monotonic()
    compiled = runner.lower(params, example).compile()
    compile_seconds = time.monotonic() - started
    records = []
    started = time.monotonic()
    for offset in range(0, games, batch_size):
        selected = seeds[offset : offset + batch_size]
        values = jax.device_get(compiled(params, jnp.asarray(np.resize(selected, batch_size))))
        exponents, totals, moves, active = [np.asarray(x)[: len(selected)] for x in values]
        if active.any():
            raise RuntimeError("Evaluation exceeded the conservative episode bound")
        for i, game_seed in enumerate(selected):
            records.append(
                {
                    "seed": int(game_seed),
                    "max_exponent": int(exponents[i]),
                    "max_tile": 1 << int(exponents[i]),
                    "tile_sum": int(totals[i]),
                    "moves": int(moves[i]),
                }
            )
    elapsed = time.monotonic() - started
    levels = np.array([r["max_exponent"] for r in records], dtype=np.float64)
    result = {
        "mode": mode,
        "games": games,
        "seed_start": seed,
        "mean_max_exponent": float(levels.mean()),
        "mean_max_tile": float(np.mean([r["max_tile"] for r in records])),
        "mean_tile_sum": float(np.mean([r["tile_sum"] for r in records])),
        "mean_moves": float(np.mean([r["moves"] for r in records])),
        "standard_error_max_exponent": float(levels.std(ddof=1) / np.sqrt(games))
        if games > 1
        else 0.0,
        "reach_rates": {str(1 << k): float(np.mean(levels >= k)) for k in (8, 9, 10)},
        "compile_seconds": compile_seconds,
        "execution_seconds": elapsed,
        "amortized_seconds_per_move": elapsed / max(1, sum(r["moves"] for r in records)),
        "batch_size": batch_size,
        "planner": asdict(planner),
    }
    if checkpoint_info is not None:
        result["checkpoint_sha256"] = checkpoint_info["checkpoint_sha256"]
        result["checkpoint_metadata"] = checkpoint_info["checkpoint_metadata"]
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        # metrics.json is the completion marker: publish it only after the CSV.
        (output / "metrics.json").unlink(missing_ok=True)
        temporary = output / "games.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        temporary.replace(output / "games.csv")
        atomic_json(output / "metrics.json", result)
    return result


def replay(
    model: ValueNetwork, params, *, seed: int, mode: str, planner: PlannerConfig, output: Path
) -> None:
    actor = make_actor(model, mode, planner)
    env_key = jax.random.PRNGKey(seed)
    plan_key = jax.random.fold_in(env_key, 1_234_567)
    board = jax_env.reset(env_key)
    frames = []
    for t in range(2048):
        action = actor(params, board, jax.random.fold_in(plan_key, t))
        frames.append({"board_exponents": np.asarray(board).tolist(), "action": int(action)})
        board, _, done = jax_env.step(board, action, jax.random.fold_in(env_key, t + 1))
        if bool(done):
            break
    else:
        raise RuntimeError("Replay exceeded episode bound")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output,
        {
            "seed": seed,
            "mode": mode,
            "planner": asdict(planner),
            "frames": frames,
            "terminal_board_exponents": np.asarray(board).tolist(),
        },
    )
