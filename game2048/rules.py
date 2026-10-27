"""Board encoding and D4 utilities; moves are implemented in kernels and jax_env."""

import numpy as np
from numpy.typing import NDArray

Board = NDArray[np.int32]


def transform(board: NDArray, group: int) -> NDArray:
    return np.rot90(np.fliplr(board) if group >= 4 else board, group % 4).copy()


def permutations_d4(size: int = 3) -> NDArray:
    indices = np.arange(size * size).reshape(size, size)
    return np.stack([transform(indices, g).ravel() for g in range(8)])


def action_permutations() -> NDArray:
    """result[g, a] is the direction a transformed by g."""
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    out = np.empty((8, 4), dtype=np.int32)
    for g in range(8):
        for a, (dr, dc) in enumerate(directions):
            if g >= 4:
                dc = -dc
            for _ in range(g % 4):
                dr, dc = -dc, dr
            out[g, a] = directions.index((dr, dc))
    return out


def encode(board: NDArray) -> np.uint64:
    value = 0
    for i, x in enumerate(board.ravel()):
        if not 0 <= int(x) <= 15:
            raise ValueError("Exponent does not fit in a nibble")
        value |= int(x) << (4 * i)
    return np.uint64(value)


def decode(code: int | np.uint64, size: int = 3) -> Board:
    return np.asarray(
        [(int(code) >> (4 * i)) & 15 for i in range(size * size)], dtype=np.int32
    ).reshape(size, size)


def canonical(board: Board) -> tuple[np.uint64, int]:
    codes = [encode(transform(board, g)) for g in range(8)]
    group = int(np.argmin(codes))
    return codes[group], group
