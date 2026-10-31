"""Augmented scalar-afterstate supervision adapted from training.train (baseline 6eb6db4)."""

import time
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from game2048 import jax_env
from game2048.ablations.evaluation import evaluate
from game2048.ablations.models import initialize, parameter_count, parameter_hash
from game2048.ablations.optimization import batch_value_and_grad
from game2048.ablations.provenance import source_identity
from game2048.ablations.state import (
    prepare,
    random_streams,
    read_recovery,
    take,
    write_recovery,
)
from game2048.artifacts import atomic_json
from game2048.rules import permutations_d4
from game2048.training import _decode_words, _make_collection, _optimizer, _teacher_arrays

TEACHER_SHA256 = "6a66f3b52bd5bef3e270c777e4da063cae7e7038a0c5c858eb7bb7274466fc8e"


def checked_teacher(directory):
    from game2048.teacher import Teacher

    teacher = Teacher(directory)
    m = teacher.manifest
    if (
        teacher.size != 3
        or len(teacher.keys) != 41_325_017
        or m.get("verified_states") != len(teacher.keys)
        or m.get("bellman_residual") != 0
        or m.get("content_sha256") != TEACHER_SHA256
        or not m.get("enumeration_complete")
    ):
        raise ValueError("Supervision requires the fully verified, complete 3x3 teacher")
    return teacher


def augment(afters, key):
    """Each scalar input gets its own uniform D4 orientation; labels remain indexed."""
    flat = afters.reshape(-1, 9)
    groups = jax.random.randint(key, (len(flat),), 0, 8)
    return jnp.take_along_axis(flat, jnp.asarray(permutations_d4())[groups], axis=1).reshape(
        -1, 3, 3
    )


def supervised_batch(words, labels, masks, rows, augmentation_key, *, holdout=False):
    boards = _decode_words(words[rows])
    afters, legal = jax.vmap(jax_env.afterstates)(boards)
    x = afters.reshape(-1, 3, 3)
    orbit_holdout = jax_env.validation_mask(x)
    valid = masks[rows].reshape(-1) & legal.reshape(-1)
    valid &= orbit_holdout if holdout else ~orbit_holdout
    return augment(x, augmentation_key), labels[rows].reshape(-1), valid


def masked_loss(predictions, targets, valid):
    # Select before squaring so an invalid NaN label cannot poison gradients.
    error = jnp.where(valid, predictions - targets, 0.0)
    return jnp.sum(error**2) / jnp.maximum(valid.sum(), 1)


def sample_rows(arrays, key, count):
    _, _, _, order, starts, counts = arrays
    k1, k2 = jax.random.split(key)
    groups = jax.random.randint(k1, (count,), 0, len(starts))
    offsets = (jax.random.uniform(k2, (count,)) * counts[groups]).astype(jnp.int32)
    return order[starts[groups] + offsets]


def make_update(model, cfg, microbatch_size):
    @jax.jit
    def update(state, sample_key, augment_key, arrays, pool, pool_count):
        key1, key2 = jax.random.split(sample_key)
        rows = sample_rows(arrays, key1, cfg.batch_size)
        student = pool[jax.random.randint(key2, (cfg.batch_size,), 0, jnp.maximum(pool_count, 1))]
        rows = jnp.where(
            (pool_count > 0) & (jnp.arange(cfg.batch_size) < cfg.batch_size // 2), student, rows
        )
        x, target, valid = supervised_batch(*arrays[:3], rows, augment_key)

        value, grads = batch_value_and_grad(model, state.params, x, target, valid, microbatch_size)
        return state.apply_gradients(grads=grads), value

    return update


def make_label_validation(model, count):
    @jax.jit
    def validate(params, arrays, key):
        sampling, directions = jax.random.split(key)
        rows = sample_rows(arrays, sampling, count)
        x, target, valid = supervised_batch(*arrays[:3], rows, directions, holdout=True)
        predicted = model.apply({"params": params}, x)
        return jnp.sum(jnp.where(valid, (predicted - target) ** 2, 0)), valid.sum()

    return validate


def train(teacher_dir, output, *, group, seed, spec, protocol, resume=False, stop_after=None):
    if group not in ("d4_distill_td", "plain_distill_td"):
        raise ValueError("Only distillation groups may enter supervised training")
    if spec.architecture != ("plain" if group == "plain_distill_td" else "d4"):
        raise ValueError("Training group and architecture mismatch")
    teacher = checked_teacher(teacher_dir)
    model, params = initialize(spec, seed)
    cfg = protocol.supervised
    request = {
        "phase": "supervised",
        "group": group,
        "seed": seed,
        "model": asdict(spec),
        "protocol": protocol.json(),
        "initial_params_sha256": parameter_hash(params),
        "upstream_sha256": None,
        "teacher_sha256": TEACHER_SHA256,
        "source": source_identity(),
    }
    checkpoint = prepare(output, request, resume=resume)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=_optimizer(cfg.learning_rate))
    streams = random_streams(seed)
    pool = np.zeros(cfg.pool_capacity, np.int32)
    boards = jnp.empty((0, 3, 3), jnp.int32)
    best_params = params
    info: dict = {
        "request": request,
        "step": 0,
        "history": [],
        "best_score": None,
        "best_step": 0,
        "stale": 0,
        "pool_count": 0,
        "pool_position": 0,
        "elapsed_seconds": 0.0,
        "environment_transitions": 0,
        "parameter_count": parameter_count(params),
    }
    if checkpoint.exists():
        state, streams, boards, pool, best_params, info = read_recovery(checkpoint, state, request)
    if (output / "complete.json").exists() and int(state.step) != cfg.max_updates:
        raise ValueError("Invalid supervised completion marker")
    arrays = _teacher_arrays(teacher)
    pool_device = jnp.asarray(pool)
    update = make_update(model, cfg, protocol.gradient_microbatch_size)
    collector = _make_collection(model, cfg.collection_envs, cfg.collection_steps)
    label_validation = make_label_validation(model, cfg.batch_size)
    started = time.monotonic()
    previous_seconds = info["elapsed_seconds"]
    compiling = time.monotonic()
    update = update.lower(
        state,
        streams["sample"],
        streams["augment"],
        arrays,
        pool_device,
        jnp.int32(info["pool_count"]),
    ).compile()
    info["training_compile_seconds"] = info.get("training_compile_seconds", 0.0) + (
        time.monotonic() - compiling
    )

    def save():
        info["step"] = int(state.step)
        info["elapsed_seconds"] = previous_seconds + time.monotonic() - started
        write_recovery(output, state, streams, boards, pool, best_params, info, spec)

    if not checkpoint.exists():
        save()
    try:
        for step in range(int(state.step) + 1, cfg.max_updates + 1):
            if step > cfg.warmup_updates and (step - 1) % cfg.collect_every == 0:
                observations = np.asarray(collector(state.params, take(streams, "collect")))
                rows, _, _ = teacher.lookup(observations)
                rows = rows[-cfg.pool_capacity :].astype(np.int32)
                indices = (np.arange(len(rows)) + info["pool_position"]) % cfg.pool_capacity
                pool[indices] = rows
                info["pool_position"] = int((info["pool_position"] + len(rows)) % cfg.pool_capacity)
                info["pool_count"] = min(cfg.pool_capacity, info["pool_count"] + len(rows))
                info["environment_transitions"] += len(observations)
                pool_device = jnp.asarray(pool)
            update_started = time.monotonic()
            state, loss = update(
                state,
                take(streams, "sample"),
                take(streams, "augment"),
                arrays,
                pool_device,
                jnp.int32(info["pool_count"]),
            )
            jax.block_until_ready(loss)
            info["training_execution_seconds"] = info.get("training_execution_seconds", 0.0) + (
                time.monotonic() - update_started
            )
            if step == 1 or step % 100 == 0:
                print({"group": group, "seed": seed, "step": step, "loss": float(loss)}, flush=True)
                atomic_json(output / "progress.json", {"status": "training", "step": step})
            if step % cfg.validation_every == 0 or step == cfg.max_updates:
                metrics = evaluate(
                    spec,
                    state.params,
                    games=protocol.validation_games,
                    seed=protocol.validation_seed,
                    batch_size=protocol.evaluation_batch,
                    smoke=protocol.smoke,
                )
                # Dedicated heldout stream, independent of training and augmentation.
                heldout_key = take(streams, "heldout")
                errors = np.asarray(
                    [
                        jax.device_get(
                            label_validation(
                                state.params, arrays, jax.random.fold_in(heldout_key, i)
                            )
                        )
                        for i in range(8)
                    ]
                ).sum(axis=0)
                metrics["heldout_afterstate_rmse"] = float(np.sqrt(errors[0] / max(errors[1], 1)))
                metrics["heldout_label_samples"] = int(errors[1])
                info["history"].append({"step": step, "loss": float(loss), "validation": metrics})
                score = metrics["mean_max_exponent"]
                if info["best_score"] is None or score > info["best_score"]:
                    info.update(best_score=score, best_step=step, stale=0)
                    best_params = state.params
                else:
                    info["stale"] += 1
                save()
                print({"validation_step": step, "score": score}, flush=True)
            if stop_after is not None and step >= stop_after:
                save()
                atomic_json(
                    output / "interrupted.json", {"step": step, "reason": "requested chunk"}
                )
                return info
        save()
        atomic_json(
            output / "complete.json",
            {
                "request": request,
                "updates": int(state.step),
                "best_params_sha256": parameter_hash(best_params),
            },
        )
        return info
    except BaseException as error:
        atomic_json(
            output / "interrupted.json",
            {
                "observed_step": int(state.step),
                "error": str(error),
                "recovery": "last atomic validation snapshot",
            },
        )
        raise
