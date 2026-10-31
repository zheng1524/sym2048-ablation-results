"""Explicit architecture factory and checkpoints, independent of teacher assets."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import serialization

from game2048.ablations.config import ModelSpec
from game2048.artifacts import atomic_bytes, atomic_json
from game2048.network import ModelConfig, ValueNetwork


class PlainValueNetwork(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, boards):
        features = jax.nn.one_hot(boards, 11, dtype=jnp.float32)
        h = nn.Conv(self.config.width, (3, 3), padding="SAME")(features)
        for _ in range(self.config.blocks):
            residual = h
            h = nn.Conv(self.config.width, (3, 3), padding="SAME")(nn.silu(nn.LayerNorm()(h)))
            h = nn.Conv(self.config.width, (3, 3), padding="SAME")(nn.silu(nn.LayerNorm()(h)))
            h = h + residual
        h = nn.silu(nn.LayerNorm()(h)).reshape(-1, 9 * self.config.width)
        value = nn.Dense(1)(nn.silu(nn.Dense(self.config.head_width)(h)))[:, 0]
        return jnp.where(jnp.max(boards, axis=(1, 2)) >= 10, 10.0, value)


def make_model(spec: ModelSpec):
    cls = ValueNetwork if spec.architecture == "d4" else PlainValueNetwork
    return cls(spec.dimensions())


def initialize(spec: ModelSpec, seed: int):
    model = make_model(spec)
    # Exactly the original train() initializer, shared by A and B.
    _, init_key = jax.random.split(jax.random.PRNGKey(seed))
    params = model.init(init_key, jnp.zeros((1, 3, 3), jnp.int32))["params"]
    return model, params


def parameter_hash(params):
    return hashlib.sha256(serialization.to_bytes(jax.device_get(params))).hexdigest()


def parameter_count(params):
    return sum(x.size for x in jax.tree.leaves(params))


def save_model(directory: Path, params, spec: ModelSpec, metadata: dict):
    directory.mkdir(parents=True, exist_ok=True)
    blob = serialization.to_bytes(jax.device_get(params))
    atomic_bytes(directory / "params.msgpack", blob)
    atomic_json(
        directory / "model.json",
        {
            **metadata,
            "architecture": spec.architecture,
            "model": asdict(spec.dimensions()),
            "params_sha256": hashlib.sha256(blob).hexdigest(),
        },
    )


def load_model(directory: Path):
    metadata = json.loads((directory / "model.json").read_text())
    spec = ModelSpec(architecture=metadata.get("architecture", "d4"), **metadata["model"])
    blob = (directory / "params.msgpack").read_bytes()
    if (
        metadata.get("params_sha256", hashlib.sha256(blob).hexdigest())
        != hashlib.sha256(blob).hexdigest()
    ):
        raise ValueError("Checkpoint parameter hash mismatch")
    params = jax.tree.map(jnp.asarray, serialization.msgpack_restore(blob))
    model = make_model(spec)
    # Validate the actual parameter tree against the declared architecture/dimensions.
    expected = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 3, 3), jnp.int32))["params"]
    if jax.tree.structure(params) != jax.tree.structure(expected) or any(
        a.shape != b.shape
        for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(expected), strict=True)
    ):
        raise ValueError("Checkpoint parameters do not match the declared model")
    return model, params, metadata, spec
