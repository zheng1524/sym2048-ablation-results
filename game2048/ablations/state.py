"""Single atomic recovery payload: parameters, optimizer, RNGs, pool, boards and best."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import serialization
from flax.training.train_state import TrainState

from game2048.ablations.models import parameter_hash, save_model
from game2048.ablations.provenance import claim
from game2048.artifacts import atomic_bytes, atomic_json

STREAMS = {
    "sample": 101,
    "augment": 102,
    "collect": 103,
    "heldout": 104,
    "td_env": 201,
    "td_tie": 202,
    "td_reset": 203,
}


def random_streams(seed):
    return {
        name: jax.random.fold_in(jax.random.PRNGKey(seed), tag) for name, tag in STREAMS.items()
    }


def take(streams, name):
    streams[name], key = jax.random.split(streams[name])
    return key


def prepare(output, request, *, resume):
    if (output / "request.json").exists() and not resume:
        raise FileExistsError("Existing stage requires --resume")
    claim(output, "request.json", request)
    return output / "recovery.msgpack"


def write_recovery(output, state, streams, boards, pool, best_params, info, spec):
    info = {
        **info,
        "step": int(state.step),
        "params_sha256": parameter_hash(state.params),
        "best_params_sha256": parameter_hash(best_params),
    }
    atomic_bytes(
        output / "recovery.msgpack",
        serialization.to_bytes(
            {
                "state": state,
                "streams": streams,
                "boards": boards,
                "pool": pool,
                "best_params": best_params,
                "info": json.dumps(info, allow_nan=False),
            }
        ),
    )
    # These are reviewable views; recovery.msgpack is the atomic source of truth.
    atomic_json(output / "training.json", info)
    save_model(output / "latest", state.params, spec, info)
    save_model(output / "best", best_params, spec, {**info, "step": info["best_step"]})


def read_recovery(path: Path, state: TrainState, request):
    raw = serialization.msgpack_restore(path.read_bytes())
    info = json.loads(raw["info"])
    if info["request"] != request:
        raise ValueError("Recovery configuration, source, architecture or upstream hash mismatch")
    state = serialization.from_state_dict(state, raw["state"])
    state = jax.tree.map(jnp.asarray, state)
    if int(state.step) != info["step"]:
        raise ValueError("Recovery state and metadata steps disagree")
    if parameter_hash(state.params) != info["params_sha256"]:
        raise ValueError("Recovery parameter hash mismatch")
    best = jax.tree.map(jnp.asarray, raw["best_params"])
    if parameter_hash(best) != info["best_params_sha256"]:
        raise ValueError("Recovery best parameter hash mismatch")
    return (
        state,
        jax.tree.map(jnp.asarray, raw["streams"]),
        jnp.asarray(raw["boards"]),
        raw["pool"].copy(),
        best,
        info,
    )
