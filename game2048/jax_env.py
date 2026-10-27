"""Fixed-shape 3x3 rules. All public functions accept one board; vmap batches them."""

import jax
import jax.numpy as jnp
import numpy as np

from game2048.rules import permutations_d4

_INDICES = np.arange(9, dtype=np.int32).reshape(3, 3)
ORIENT = np.stack([_INDICES, _INDICES[:, ::-1], _INDICES.T, _INDICES.T[:, ::-1]])
UNDO = np.argsort(ORIENT.reshape(4, 9), axis=1)
D4 = permutations_d4().astype(np.int32)


def merge_lines(rows: jax.Array) -> jax.Array:
    a, b, c = rows[..., 0], rows[..., 1], rows[..., 2]
    swap = a == 0
    a, b = jnp.where(swap, b, a), jnp.where(swap, a, b)
    swap = b == 0
    b, c = jnp.where(swap, c, b), jnp.where(swap, b, c)
    swap = a == 0
    a, b = jnp.where(swap, b, a), jnp.where(swap, a, b)
    ab = (a > 0) & (a == b)
    bc = (b > 0) & (b == c) & ~ab
    return jnp.stack(
        [
            jnp.where(ab, a + 1, a),
            jnp.where(ab, c, jnp.where(bc, b + 1, b)),
            jnp.where(ab | bc, 0, c),
        ],
        axis=-1,
    )


def afterstates(board: jax.Array) -> tuple[jax.Array, jax.Array]:
    oriented = board.reshape(9)[jnp.asarray(ORIENT)]
    merged = merge_lines(oriented).reshape(4, 9)
    states = jnp.take_along_axis(merged, jnp.asarray(UNDO), axis=1).reshape(4, 3, 3)
    return states, jnp.any(states != board, axis=(1, 2))


def spawn(board: jax.Array, key: jax.Array) -> jax.Array:
    draws = jax.random.uniform(key, (2,))
    flat = board.reshape(9)
    empty = flat == 0
    count = jnp.sum(empty)
    rank = jnp.minimum((draws[0] * count).astype(jnp.int32), jnp.maximum(count - 1, 0))
    position = jnp.argmax(empty & (jnp.cumsum(empty) == rank + 1))
    value = jnp.where(draws[1] < 0.9, 1, 2).astype(jnp.int32)
    return jnp.where(count > 0, flat.at[position].set(value), flat).reshape(3, 3)


def reset(key: jax.Array) -> jax.Array:
    k1, k2 = jax.random.split(key)
    return spawn(spawn(jnp.zeros((3, 3), dtype=jnp.int32), k1), k2)


def step(board: jax.Array, action: jax.Array, key: jax.Array):
    states, legal = afterstates(board)
    moved = states[action]
    nxt = jnp.where(legal[action], spawn(moved, key), board)
    _, next_legal = afterstates(nxt)
    done = ~jnp.any(next_legal)
    reward = jnp.where(done & legal[action], jnp.max(nxt).astype(jnp.float32), 0.0)
    return nxt, reward, done


def canonical_boards(boards: jax.Array) -> jax.Array:
    """Lexicographic minimum, identical to the low-nibble-first uint64 encoding."""
    versions = boards.reshape(-1, 9)[:, jnp.asarray(D4)]
    best = versions[:, 0]
    for g in range(1, 8):
        candidate = versions[:, g]
        # Highest differing nibble determines packed integer ordering.
        differing = candidate != best
        position = jnp.max(jnp.where(differing, jnp.arange(9), -1), axis=1)
        p = jnp.maximum(position, 0)
        better = (position >= 0) & (
            jnp.take_along_axis(candidate, p[:, None], axis=1)[:, 0]
            < jnp.take_along_axis(best, p[:, None], axis=1)[:, 0]
        )
        best = jnp.where(better[:, None], candidate, best)
    return best.reshape(-1, 3, 3)


def validation_mask(boards: jax.Array) -> jax.Array:
    canonical = canonical_boards(boards).reshape(-1, 9).astype(jnp.uint32)
    h = jnp.full(len(canonical), 2166136261, dtype=jnp.uint32)
    for i in range(9):
        h = (h ^ canonical[:, i]) * jnp.uint32(16777619)
    return (h % 20) == 0


def tile_sum(board: jax.Array) -> jax.Array:
    return jnp.sum(jnp.where(board > 0, jnp.left_shift(1, board), 0))
