"""Exact value supervision, student-distribution sampling, and teacher-free TD(lambda)."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState

from game2048 import jax_env
from game2048.artifacts import atomic_bytes, atomic_json
from game2048.evaluation import evaluate
from game2048.network import ModelConfig, ValueNetwork, greedy_action, load_model, save_model
from game2048.teacher import Teacher


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    batch_size: int = 1024
    learning_rate: float = 3e-4
    max_updates: int = 100_000
    warmup_updates: int = 5000
    validation_every: int = 5000
    validation_games: int = 512
    patience: int = 5
    collect_every: int = 1000
    collection_envs: int = 128
    collection_steps: int = 256
    pool_capacity: int = 262_144

    def __post_init__(self):
        if (
            min(
                self.batch_size,
                self.max_updates,
                self.validation_every,
                self.validation_games,
                self.patience,
                self.collect_every,
                self.collection_envs,
                self.collection_steps,
                self.pool_capacity,
            )
            < 1
            or self.learning_rate <= 0
            or self.warmup_updates < 0
        ):
            raise ValueError("Training counts/rate must be positive; warmup must be nonnegative")


def _decode_words(words):
    shifts = jnp.arange(9, dtype=jnp.uint32) * 4
    low = (words[:, 0, None] >> jnp.minimum(shifts, 28)) & 15
    high = (words[:, 1, None] >> (jnp.maximum(shifts, 32) - 32)) & 15
    return jnp.where(shifts < 32, low, high).astype(jnp.int32).reshape(-1, 3, 3)


def _teacher_arrays(teacher: Teacher):
    keys = np.asarray(teacher.keys)
    words = np.stack(
        [
            (keys & np.uint64(0xFFFFFFFF)).astype(np.uint32),
            (keys >> np.uint64(32)).astype(np.uint32),
        ],
        axis=1,
    )
    maximum = np.zeros(len(keys), dtype=np.uint8)
    empty = np.zeros(len(keys), dtype=np.uint8)
    for i in range(9):
        digit = ((keys >> np.uint64(4 * i)) & 15).astype(np.uint8)
        maximum = np.maximum(maximum, digit)
        empty += digit == 0
    strata = maximum.astype(np.int16) * 10 + empty
    order = np.argsort(strata, kind="stable").astype(np.int32)
    counts = np.bincount(strata, minlength=110)
    starts = np.cumsum(counts) - counts
    valid = np.flatnonzero(counts)
    return tuple(
        jnp.asarray(a)
        for a in (
            words,
            np.asarray(teacher.q, dtype=np.float32),
            np.asarray(teacher.legal),
            order,
            starts[valid].astype(np.int32),
            counts[valid].astype(np.int32),
        )
    )


def _optimizer(rate):
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(rate, weight_decay=1e-5))


def lambda_returns(rewards, dones, next_values, bootstrap, trace_lambda):
    """Finite rollout lambda returns; a terminal cuts all following-episode values."""

    def backup(g, inputs):
        reward, done, following = inputs
        target = reward + jnp.where(done, 0.0, (1 - trace_lambda) * following + trace_lambda * g)
        return target, target

    _, targets = jax.lax.scan(backup, bootstrap, (rewards, dones, next_values), reverse=True)
    return jax.lax.stop_gradient(targets)


def _make_label_validation(model, batch_size=1024):
    @jax.jit
    def validate(params, arrays, key):
        words, labels, masks, order, starts, counts = arrays
        k1, k2 = jax.random.split(key)
        groups = jax.random.randint(k1, (batch_size,), 0, len(starts))
        offsets = (jax.random.uniform(k2, (batch_size,)) * counts[groups]).astype(jnp.int32)
        rows = order[starts[groups] + offsets]
        afters, _ = jax.vmap(jax_env.afterstates)(_decode_words(words[rows]))
        x = afters.reshape(-1, 3, 3)
        valid = masks[rows].reshape(-1) & jax_env.validation_mask(x)
        error = model.apply({"params": params}, x) - labels[rows].reshape(-1)
        return jnp.sum(jnp.where(valid, error**2, 0)), valid.sum()

    return validate


def _make_collection(model: ValueNetwork, envs: int, steps: int):
    def collect(params, key):
        boards = jax.vmap(jax_env.reset)(jax.random.split(key, envs))

        def advance(current, t):
            keys = jax.random.split(jax.random.fold_in(key, t + 1), envs * 5).reshape(envs, 5, 2)
            actions = jax.vmap(lambda b, k: greedy_action(model, params, b, k))(current, keys[:, 0])
            _, legal = jax.vmap(jax_env.afterstates)(current)
            random_actions = jnp.argmax(
                jnp.where(legal, jax.vmap(lambda k: jax.random.uniform(k, (4,)))(keys[:, 1]), -1.0),
                axis=1,
            )
            explore = jax.vmap(lambda k: jax.random.uniform(k) < 0.05)(keys[:, 2])
            actions = jnp.where(explore, random_actions, actions)
            nxt, _, done = jax.vmap(jax_env.step)(current, actions, keys[:, 3])
            resets = jax.vmap(jax_env.reset)(keys[:, 4])
            return jnp.where(done[:, None, None], resets, nxt), current

        _, observed = jax.lax.scan(advance, boards, jnp.arange(steps))
        return observed.reshape(-1, 3, 3)

    return jax.jit(collect)


def train(
    teacher_dir: Path,
    output: Path,
    config: TrainConfig = TrainConfig(),
    model_config: ModelConfig = ModelConfig(),
    resume: bool = False,
) -> dict:
    teacher = Teacher(teacher_dir)
    if teacher.size != 3:
        raise ValueError("The neural training environment is 3x3")
    output.mkdir(parents=True, exist_ok=True)
    model = ValueNetwork(model_config)
    key = jax.random.PRNGKey(config.seed)
    key, init_key = jax.random.split(key)
    params = model.init(init_key, jnp.zeros((1, 3, 3), jnp.int32))["params"]
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=_optimizer(config.learning_rate)
    )
    history = []
    best_score, stale = -float("inf"), 0
    pool = np.zeros(config.pool_capacity, dtype=np.int32)
    pool_count, pool_position = 0, 0
    start = 0
    checkpoint = output / "training.msgpack"
    if checkpoint.exists() and not resume:
        raise FileExistsError("Training checkpoint exists; use --resume or a new output directory")
    if resume:
        restored = serialization.from_bytes({"state": state, "key": key}, checkpoint.read_bytes())
        state, key = restored["state"], restored["key"]
        info = json.loads((output / "training.json").read_text())
        if info["teacher_sha256"] != teacher.manifest["content_sha256"]:
            raise ValueError("Cannot resume against a different teacher")
        if info["model"] != asdict(model_config) or info["config"] != asdict(config):
            raise ValueError("Resume requires the original model and training configuration")
        start, best_score, stale = info["step"], info["best_score"], info["stale"]
        if int(state.step) != start:
            raise ValueError("Incomplete training checkpoint: state and metadata steps disagree")
        if stale >= config.patience or start >= config.max_updates:
            result = {"best_score": best_score, "updates": start, "output": str(output)}
            atomic_json(output / "complete.json", result)
            return result
        history = info["history"]
        pool = np.load(output / "pool.npy")
        pool_count, pool_position = info["pool_count"], info["pool_position"]
    arrays = _teacher_arrays(teacher)
    pool_device = jnp.asarray(pool)
    collector = _make_collection(model, config.collection_envs, config.collection_steps)
    label_validation = _make_label_validation(model)

    @jax.jit
    def update(state, key, arrays, pool_ids, pool_n, use_pool):
        words, labels, masks, order, starts, counts = arrays
        key, k1, k2, k3 = jax.random.split(key, 4)
        groups = jax.random.randint(k1, (config.batch_size,), 0, len(starts))
        offsets = (jax.random.uniform(k2, (config.batch_size,)) * counts[groups]).astype(jnp.int32)
        rows = order[starts[groups] + offsets]
        student_rows = pool_ids[
            jax.random.randint(k3, (config.batch_size,), 0, jnp.maximum(pool_n, 1))
        ]
        rows = jnp.where(
            use_pool & (jnp.arange(config.batch_size) < config.batch_size // 2), student_rows, rows
        )
        boards = _decode_words(words[rows])
        after, _ = jax.vmap(jax_env.afterstates)(boards)
        x = after.reshape(-1, 3, 3)
        valid = masks[rows].reshape(-1) & ~jax_env.validation_mask(x)
        target = labels[rows].reshape(-1)

        def loss(params):
            predicted = model.apply({"params": params}, x)
            return jnp.sum(jnp.where(valid, (predicted - target) ** 2, 0.0)) / jnp.maximum(
                valid.sum(), 1
            )

        value, grads = jax.value_and_grad(loss)(state.params)
        return state.apply_gradients(grads=grads), key, value

    started = time.monotonic()
    for step in range(start + 1, config.max_updates + 1):
        if step > config.warmup_updates and (step - 1) % config.collect_every == 0:
            key, collect_key = jax.random.split(key)
            observations = np.asarray(collector(state.params, collect_key))
            rows, _, _ = teacher.lookup(observations)
            rows = rows[-config.pool_capacity :].astype(np.int32)
            indices = (np.arange(len(rows)) + pool_position) % config.pool_capacity
            pool[indices] = rows
            pool_position = int((pool_position + len(rows)) % config.pool_capacity)
            pool_count = min(config.pool_capacity, pool_count + len(rows))
            pool_device = jnp.asarray(pool)
        state, key, loss = update(
            state, key, arrays, pool_device, jnp.int32(pool_count), pool_count > 0
        )
        if step % 100 == 0 or step == 1:
            print(
                json.dumps(
                    {"step": step, "loss": float(loss), "seconds": time.monotonic() - started}
                ),
                flush=True,
            )
        if step % config.validation_every == 0 or step == config.max_updates:
            metrics = evaluate(model, state.params, games=config.validation_games, seed=10_000)
            errors = np.asarray(
                [
                    jax.device_get(
                        label_validation(state.params, arrays, jax.random.PRNGKey(20_000 + i))
                    )
                    for i in range(8)
                ]
            ).sum(axis=0)
            metrics["heldout_afterstate_rmse"] = float(np.sqrt(errors[0] / max(errors[1], 1)))
            metrics["heldout_label_samples"] = int(errors[1])
            score = metrics["mean_max_exponent"]
            history.append({"step": step, "loss": float(loss), "validation": metrics})
            if score > best_score:
                best_score, stale = score, 0
                save_model(
                    output / "best",
                    state.params,
                    model_config,
                    {
                        "training": "exact_teacher",
                        "step": step,
                        "seed": config.seed,
                        "validation": metrics,
                        "teacher_sha256": teacher.manifest["content_sha256"],
                    },
                )
            else:
                stale += 1
            save_model(
                output / "latest",
                state.params,
                model_config,
                {"training": "exact_teacher", "step": step, "seed": config.seed},
            )
            atomic_bytes(checkpoint, serialization.to_bytes({"state": state, "key": key}))
            np.save(output / "pool.npy", pool)
            info = {
                "step": step,
                "config": asdict(config),
                "model": asdict(model_config),
                "best_score": best_score,
                "stale": stale,
                "history": history,
                "pool_count": pool_count,
                "pool_position": pool_position,
                "teacher_sha256": teacher.manifest["content_sha256"],
            }
            atomic_json(output / "training.json", info)
            print(
                json.dumps({"validation_step": step, "score": score, "best": best_score}),
                flush=True,
            )
            if stale >= config.patience:
                break
    result = {"best_score": best_score, "updates": int(state.step), "output": str(output)}
    atomic_json(output / "complete.json", result)
    return result


def finetune(
    checkpoint: Path,
    output: Path,
    *,
    updates: int = 10_000,
    seed: int = 0,
    trace_lambda: float = 0.8,
    learning_rate: float = 3e-5,
    envs: int = 128,
    horizon: int = 32,
    validation_every: int = 1000,
    validation_games: int = 512,
    resume: bool = False,
):
    if not 0 <= trace_lambda <= 1:
        raise ValueError("lambda must be in [0, 1]")
    model, params, _ = load_model(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training.msgpack").exists() and not resume:
        raise FileExistsError("TD checkpoint exists; use --resume or a new output directory")
    key = jax.random.PRNGKey(seed)
    boards = jax.vmap(jax_env.reset)(jax.random.split(key, envs))
    state = TrainState.create(apply_fn=model.apply, params=params, tx=_optimizer(learning_rate))
    history = []
    config = dict(
        updates=updates,
        seed=seed,
        trace_lambda=trace_lambda,
        learning_rate=learning_rate,
        envs=envs,
        horizon=horizon,
        validation_every=validation_every,
        validation_games=validation_games,
    )
    if min(updates, envs, horizon, validation_every, validation_games) < 1 or learning_rate <= 0:
        raise ValueError("TD counts and learning rate must be positive")
    start = 0
    if resume:
        info = json.loads((output / "training.json").read_text())
        if info["config"] != config or info["model"] != asdict(model.config):
            raise ValueError("Resume requires the original TD and model configuration")
        restored = serialization.from_bytes(
            {"state": state, "boards": boards, "key": key},
            (output / "training.msgpack").read_bytes(),
        )
        state, boards, key = restored["state"], restored["boards"], restored["key"]
        start, best, history = info["step"], info["best_score"], info["history"]
        if int(state.step) != start:
            raise ValueError("Incomplete TD checkpoint: state and metadata steps disagree")
    else:
        initial_metrics = evaluate(model, params, games=validation_games, seed=10_000)
        best = initial_metrics["mean_max_exponent"]
        save_model(
            output / "best",
            params,
            model.config,
            {
                "training": "td_lambda",
                "step": 0,
                "seed": seed,
                "lambda": trace_lambda,
                "validation": initial_metrics,
            },
        )

    @jax.jit
    def update(state, boards, key):
        key, rollout_key = jax.random.split(key)

        def transition(current, t):
            keys = jax.random.split(jax.random.fold_in(rollout_key, t), envs * 3).reshape(
                envs, 3, 2
            )
            actions = jax.vmap(lambda b, k: greedy_action(model, state.params, b, k))(
                current, keys[:, 0]
            )
            afters, _ = jax.vmap(jax_env.afterstates)(current)
            selected = afters[jnp.arange(envs), actions]
            nxt, reward, done = jax.vmap(jax_env.step)(current, actions, keys[:, 1])
            nxt = jnp.where(done[:, None, None], jax.vmap(jax_env.reset)(keys[:, 2]), nxt)
            return nxt, (selected, reward, done)

        boards, (afters, rewards, dones) = jax.lax.scan(transition, boards, jnp.arange(horizon))
        values = model.apply({"params": state.params}, afters.reshape(-1, 3, 3)).reshape(
            horizon, envs
        )
        options, legal = jax.vmap(jax_env.afterstates)(boards)
        final_values = model.apply({"params": state.params}, options.reshape(-1, 3, 3)).reshape(
            envs, 4
        )
        bootstrap = jnp.max(jnp.where(legal, final_values, -jnp.inf), axis=1)
        next_values = jnp.concatenate([values[1:], bootstrap[None]], axis=0)

        targets = lambda_returns(rewards, dones, next_values, bootstrap, trace_lambda)

        def loss(p):
            prediction = model.apply({"params": p}, afters.reshape(-1, 3, 3)).reshape(horizon, envs)
            return jnp.mean((prediction - targets) ** 2)

        error, grads = jax.value_and_grad(loss)(state.params)
        return state.apply_gradients(grads=grads), boards, key, error

    for step in range(start + 1, updates + 1):
        state, boards, key, loss = update(state, boards, key)
        if step % 100 == 0 or step == 1:
            print(json.dumps({"td_step": step, "loss": float(loss)}), flush=True)
        if step % validation_every == 0 or step == updates:
            metrics = evaluate(model, state.params, games=validation_games, seed=10_000)
            history.append({"step": step, "validation": metrics})
            if metrics["mean_max_exponent"] > best:
                best = metrics["mean_max_exponent"]
                save_model(
                    output / "best",
                    state.params,
                    model.config,
                    {
                        "training": "td_lambda",
                        "step": step,
                        "seed": seed,
                        "lambda": trace_lambda,
                        "validation": metrics,
                    },
                )
            save_model(
                output / "latest",
                state.params,
                model.config,
                {"training": "td_lambda", "step": step, "seed": seed},
            )
            atomic_bytes(
                output / "training.msgpack",
                serialization.to_bytes({"state": state, "boards": boards, "key": key}),
            )
            atomic_json(
                output / "training.json",
                {
                    "history": history,
                    "best_score": best,
                    "lambda": trace_lambda,
                    "learning_rate": learning_rate,
                    "config": config,
                    "model": asdict(model.config),
                    "step": step,
                },
            )
    result = {"best_score": best, "updates": updates}
    atomic_json(output / "complete.json", result)
    return result
