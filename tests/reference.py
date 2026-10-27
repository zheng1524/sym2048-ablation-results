"""Adapt Gym's actual tile values to exponents; no extra movement implementation."""

import numpy as np

from game2048.envs import Game2048Env


def exponents(values):
    return np.log2(np.maximum(values, 1)).astype(np.int32)


def gym_afterstates(board):
    env = Game2048Env(size=board.shape[0])
    env.grid = np.where(board > 0, np.left_shift(1, board), 0).astype(np.int32)
    states = np.stack(
        [
            exponents(method())
            for method in (env._move_left, env._move_right, env._move_up, env._move_down)
        ]
    )
    return states, np.any(states != board, axis=(1, 2))
