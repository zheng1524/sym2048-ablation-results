"""Teacher-free TD(lambda), adapted from training.finetune; shared by all groups."""

import json
import time
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from game2048 import jax_env
from game2048.ablations.evaluation import evaluate
from game2048.ablations.models import initialize, load_model, parameter_count, parameter_hash
from game2048.ablations.provenance import source_identity
from game2048.ablations.state import prepare, random_streams, read_recovery, take, write_recovery
from game2048.artifacts import atomic_json
from game2048.network import greedy_action
from game2048.training import _optimizer, lambda_returns


def make_update(model, cfg):
    @jax.jit
    def update(state, boards, env_key, tie_key, reset_key):
        def transition(current, t):
            def keys(root):
                return jax.random.split(jax.random.fold_in(root, t), cfg.envs)

            actions = jax.vmap(lambda b, k: greedy_action(model, state.params, b, k))(
                current, keys(tie_key)
            )
            options, _ = jax.vmap(jax_env.afterstates)(current)
            selected = options[jnp.arange(cfg.envs), actions]
            nxt, reward, done = jax.vmap(jax_env.step)(current, actions, keys(env_key))
            nxt = jnp.where(done[:, None, None], jax.vmap(jax_env.reset)(keys(reset_key)), nxt)
            return nxt, (selected, reward, done)

        boards, (afters, rewards, dones) = jax.lax.scan(transition, boards, jnp.arange(cfg.horizon))
        values = model.apply({"params": state.params}, afters.reshape(-1, 3, 3)).reshape(
            cfg.horizon, cfg.envs
        )
        options, legal = jax.vmap(jax_env.afterstates)(boards)
        final_values = model.apply({"params": state.params}, options.reshape(-1, 3, 3)).reshape(
            cfg.envs, 4
        )
        bootstrap = jnp.max(jnp.where(legal, final_values, -jnp.inf), axis=1)
        following = jnp.concatenate([values[1:], bootstrap[None]], axis=0)
        targets = lambda_returns(rewards, dones, following, bootstrap, cfg.trace_lambda)

        def loss(params):
            prediction = model.apply({"params": params}, afters.reshape(-1, 3, 3)).reshape(
                cfg.horizon, cfg.envs
            )
            return jnp.mean((prediction - targets) ** 2)

        error, grads = jax.value_and_grad(loss)(state.params)
        return state.apply_gradients(grads=grads), boards, error

    return update


def train(output, *, group, seed, spec, protocol, upstream=None, resume=False, stop_after=None):
    expected_architecture = "plain" if group == "plain_distill_td" else "d4"
    if spec.architecture != expected_architecture:
        raise ValueError("Training group and architecture mismatch")
    if group in ("d4_distill_td", "plain_distill_td") and upstream is None:
        raise ValueError("Distillation groups require their supervised best checkpoint")
    if group in ("d4_scratch_td", "d4_scratch_practical", "lr_pilot") and upstream is not None:
        raise ValueError("Scratch groups cannot load any upstream checkpoint")
    model, params = initialize(spec, seed)
    initial_hash = parameter_hash(params)
    upstream_hash = None
    if upstream is not None:
        model, params, metadata, loaded_spec = load_model(upstream)
        if loaded_spec != spec:
            raise ValueError("Upstream model family mismatch")
        supervised_request = metadata["request"]
        if (
            supervised_request["group"] != group
            or supervised_request["seed"] != seed
            or supervised_request["initial_params_sha256"] != initial_hash
            or supervised_request["phase"] != "supervised"
        ):
            raise ValueError("Wrong upstream group, seed or initialization")
        if (
            supervised_request["protocol"] != protocol.json()
            or supervised_request["source"] != source_identity()
        ):
            raise ValueError("Upstream protocol or source mismatch")
        upstream_hash = parameter_hash(params)
        completed = json.loads((upstream.parent / "complete.json").read_text())
        if (
            completed["updates"] != protocol.supervised.max_updates
            or completed["best_params_sha256"] != upstream_hash
        ):
            raise ValueError("Upstream completion or checkpoint hash mismatch")
    cfg = protocol.td
    request = {
        "phase": "td",
        "group": group,
        "seed": seed,
        "model": asdict(spec),
        "protocol": protocol.json(),
        "initial_params_sha256": initial_hash,
        "upstream_sha256": upstream_hash,
        "source": source_identity(),
    }
    checkpoint = prepare(output, request, resume=resume)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=_optimizer(cfg.learning_rate))
    streams = random_streams(seed)
    boards = jax.vmap(jax_env.reset)(jax.random.split(take(streams, "td_reset"), cfg.envs))
    pool = np.empty(0, np.int32)
    best_params = params
    info: dict = {
        "request": request,
        "step": 0,
        "history": [],
        "best_score": None,
        "best_step": 0,
        "stale": 0,
        "elapsed_seconds": 0.0,
        "environment_transitions": 0,
        "parameter_count": parameter_count(params),
    }
    if checkpoint.exists():
        state, streams, boards, pool, best_params, info = read_recovery(checkpoint, state, request)
    if (output / "complete.json").exists() and int(state.step) != cfg.updates:
        raise ValueError("Invalid TD completion marker")
    update = make_update(model, cfg)
    started = time.monotonic()
    previous_seconds = info["elapsed_seconds"]
    compiling = time.monotonic()
    update = update.lower(
        state, boards, streams["td_env"], streams["td_tie"], streams["td_reset"]
    ).compile()
    info["training_compile_seconds"] = info.get("training_compile_seconds", 0.0) + (
        time.monotonic() - compiling
    )

    def save():
        info["step"] = int(state.step)
        info["elapsed_seconds"] = previous_seconds + time.monotonic() - started
        write_recovery(output, state, streams, boards, pool, best_params, info, spec)

    try:
        if not info["history"]:
            metrics = evaluate(
                spec,
                state.params,
                games=protocol.validation_games,
                seed=protocol.validation_seed,
                batch_size=protocol.evaluation_batch,
                smoke=protocol.smoke,
            )
            info.update(best_score=metrics["mean_max_exponent"], best_step=0)
            info["history"].append({"step": 0, "validation": metrics})
            save()
        for step in range(int(state.step) + 1, cfg.updates + 1):
            update_started = time.monotonic()
            state, boards, loss = update(
                state,
                boards,
                take(streams, "td_env"),
                take(streams, "td_tie"),
                take(streams, "td_reset"),
            )
            jax.block_until_ready(loss)
            info["training_execution_seconds"] = info.get("training_execution_seconds", 0.0) + (
                time.monotonic() - update_started
            )
            info["environment_transitions"] = step * cfg.envs * cfg.horizon
            if step == 1 or step % 100 == 0:
                print(
                    {"group": group, "seed": seed, "td_step": step, "loss": float(loss)}, flush=True
                )
                atomic_json(output / "progress.json", {"status": "training", "step": step})
            if step % cfg.validation_every == 0 or step == cfg.updates:
                metrics = evaluate(
                    spec,
                    state.params,
                    games=protocol.validation_games,
                    seed=protocol.validation_seed,
                    batch_size=protocol.evaluation_batch,
                    smoke=protocol.smoke,
                )
                info["history"].append({"step": step, "loss": float(loss), "validation": metrics})
                score = metrics["mean_max_exponent"]
                if score > info["best_score"]:
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
        # Roll back to the last atomic validation snapshot. An exception may arrive
        # between drawing keys and committing an update; never snapshot that mixture.
        atomic_json(
            output / "interrupted.json",
            {
                "observed_step": int(state.step),
                "error": str(error),
                "recovery": "last atomic validation snapshot",
            },
        )
        raise
