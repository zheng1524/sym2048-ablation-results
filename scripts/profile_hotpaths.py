"""Small warm-kernel timing check, not a model-quality or tuning experiment."""

import argparse
import time
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from game2048 import jax_env
from game2048.artifacts import atomic_json
from game2048.network import ModelConfig, ValueNetwork
from game2048.planner import PlannerConfig, make_actor
from game2048.rules import decode
from game2048.teacher import Teacher


def measure(function, *args, repeats=3):
    started = time.monotonic()
    jax.block_until_ready(function(*args))
    first_seconds = time.monotonic() - started
    started = time.monotonic()
    for _ in range(repeats):
        jax.block_until_ready(function(*args))
    return {"first_seconds": first_seconds, "warm_seconds": (time.monotonic() - started) / repeats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    teacher = Teacher(args.teacher)
    rng = np.random.default_rng(123)
    rows = rng.integers(0, len(teacher.keys), 32768)
    boards = np.stack([decode(code) for code in teacher.keys[rows]])
    result = {"teacher_lookup_32768": measure(teacher.lookup, boards)}
    model = ValueNetwork(ModelConfig())
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 3, 3), jnp.int32))["params"]
    afters, _ = jax.vmap(jax_env.afterstates)(jnp.asarray(boards[:1024]))
    x = afters.reshape(-1, 3, 3)
    labels = jnp.asarray(teacher.q[rows[:1024]].reshape(-1), dtype=jnp.float32)
    mask = jnp.asarray(teacher.legal[rows[:1024]].reshape(-1))
    predict = jax.jit(lambda p, b: model.apply({"params": p}, b))
    result["network_forward_4096"] = measure(predict, params, x)

    def loss(p, x, labels, mask):
        values = cast(jax.Array, model.apply({"params": p}, x))
        return jnp.sum(jnp.where(mask, (values - labels) ** 2, 0)) / mask.sum()

    result["network_forward_backward_4096"] = measure(
        jax.jit(jax.value_and_grad(loss)), params, x, labels, mask
    )
    board = jax_env.reset(jax.random.PRNGKey(100_000))
    for mode in ("greedy", "mppi"):
        actor = make_actor(model, mode, PlannerConfig())
        result[f"{mode}_single_action"] = measure(actor, params, board, jax.random.PRNGKey(8))
    result["model_parameters"] = sum(x.size for x in jax.tree.leaves(params))
    result["devices"] = [str(device) for device in jax.devices()]
    result["jax_version"] = jax.__version__
    result["note"] = "Default randomly initialized model; warm repeated-input timing, not quality."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
