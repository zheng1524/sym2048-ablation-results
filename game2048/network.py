"""D4-equivariant convolutions and an invariant scalar afterstate readout."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from game2048.artifacts import atomic_bytes, atomic_json
from game2048.jax_env import afterstates
from game2048.rules import permutations_d4

_PERMS = permutations_d4()
_COMPOSE = np.array(
    [
        [
            next(k for k in range(8) if np.array_equal(_PERMS[k], _PERMS[b][_PERMS[a]]))
            for b in range(8)
        ]
        for a in range(8)
    ]
)
_INVERSE = np.array([next(b for b in range(8) if _COMPOSE[a, b] == 0) for a in range(8)])
_RELATIVE = _COMPOSE[_INVERSE[:, None], np.arange(8)[None, :]]


@dataclass(frozen=True)
class ModelConfig:
    width: int = 32
    blocks: int = 2
    head_width: int = 64


class ValueModel(Protocol):
    """Minimal value interface, also usable by exact reference tests."""

    def apply(self, variables, boards, /) -> object: ...


class GroupConv(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        groups, channels = x.shape[-2:]
        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (9, groups, channels, self.features),
            jnp.float32,
        )
        spatial = jnp.asarray(_PERMS)
        relative = jnp.asarray(_RELATIVE if groups == 8 else np.zeros((8, 1), np.int32))
        expanded = kernel[spatial[:, :, None], relative[:, None, :]]
        expanded = expanded.transpose(1, 2, 3, 0, 4).reshape(
            3, 3, groups * channels, 8 * self.features
        )
        y = jax.lax.conv_general_dilated(
            x.reshape(*x.shape[:3], groups * channels),
            expanded,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        bias = self.param("bias", nn.initializers.zeros, (self.features,), jnp.float32)
        return y.reshape(*y.shape[:3], 8, self.features) + bias


class ValueNetwork(nn.Module):
    config: ModelConfig = ModelConfig()

    @nn.compact
    def __call__(self, boards):
        features = jax.nn.one_hot(boards, 11, dtype=jnp.float32)[..., None, :]
        h = GroupConv(self.config.width)(features)
        for _ in range(self.config.blocks):
            residual = h
            h = GroupConv(self.config.width)(nn.silu(nn.LayerNorm()(h)))
            h = GroupConv(self.config.width)(nn.silu(nn.LayerNorm()(h)))
            h = h + residual
        h = nn.silu(nn.LayerNorm()(h)).mean(axis=3).reshape(-1, 9, self.config.width)
        pooled = jnp.concatenate(
            [
                h[:, 4],
                h[:, jnp.array([1, 3, 5, 7])].mean(axis=1),
                h[:, jnp.array([0, 2, 6, 8])].mean(axis=1),
            ],
            axis=-1,
        )
        value = nn.Dense(1)(nn.silu(nn.Dense(self.config.head_width)(pooled)))[:, 0]
        # An analytic game-rule boundary, independent of the teacher artifact.
        return jnp.where(jnp.max(boards, axis=(1, 2)) >= 10, 10.0, value)


def action_values(model: ValueModel, params, board):
    states, legal = afterstates(board)
    q = cast(jax.Array, model.apply({"params": params}, states))
    return jnp.where(legal, q, -jnp.inf), legal


def greedy_action(model: ValueModel, params, board, key):
    q, legal = action_values(model, params, board)
    ties = legal & (q == jnp.max(q))
    # Uniform tie breaking preserves equivariance in distribution.
    scores = jnp.where(ties, jax.random.uniform(key, (4,)), -1.0)
    return jnp.argmax(scores).astype(jnp.int32)


def save_model(directory: Path, params, config: ModelConfig, metadata: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    atomic_bytes(directory / "params.msgpack", serialization.to_bytes(jax.device_get(params)))
    atomic_json(directory / "model.json", {"model": asdict(config), **metadata})


def load_model(directory: Path):
    import json

    metadata = json.loads((directory / "model.json").read_text())
    config = ModelConfig(**metadata["model"])
    params = serialization.msgpack_restore((directory / "params.msgpack").read_bytes())
    params = jax.tree.map(jnp.asarray, params)
    return ValueNetwork(config), params, metadata
