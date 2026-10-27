"""
@author: Wenyu Li

* the gymnasium game 2048 environment
"""

from enum import Enum

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class Game2048Env(gym.Env):
    """Gymnasium reference environment, using actual tile values as observations.

    Actions: left, right, up, down (0..3). Each legal move spawns 2 or 4
    with probabilities 0.9 and 0.1 in a uniformly selected empty cell.
    Episodes end only when no legal move remains. The terminal reward is
    log2(maximum tile); intermediate and unchanged moves have reward zero.
    Reaching target is reported in info and does not end the episode.
    render() returns a string for ansi, an RGB array for rgb_array, or draws a human window.
    Install with uv sync, then gym.make("Game2048-v0", size=3, target=256).
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode=None, size=3, target=256):
        self.size = size  # The size of the square grid
        self.target = target  # The target score to reach

        self.window_size = 512  # The size of the window to render the game in pixels

        self.observation_space = spaces.Box(
            low=0, high=np.iinfo(np.int32).max, shape=(size, size), dtype=np.int32
        )

        self.action_space = spaces.Discrete(4)  # left, right, up, down

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros(shape=(self.size, self.size), dtype=np.int32)
        self._add_random_tile()
        self._add_random_tile()

        observation = self.grid.copy()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _get_info(self):
        maximum = int(np.max(self.grid))
        return {
            "max_tile": maximum,
            "tile_sum": int(self.grid.sum()),
            "reached_target": maximum >= self.target,
        }

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        prev_grid = self.grid.copy()
        if action == Action.LEFT.value:
            self.grid = self._move_left()
        elif action == Action.RIGHT.value:
            self.grid = self._move_right()
        elif action == Action.UP.value:
            self.grid = self._move_up()
        elif action == Action.DOWN.value:
            self.grid = self._move_down()

        changed = not np.array_equal(prev_grid, self.grid)
        if changed:
            self._add_random_tile()

        terminated = self._check_game_over()
        reward = float(np.log2(np.max(self.grid))) if terminated and changed else 0.0

        truncated = False

        observation = self.grid.copy()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode in ("human", "rgb_array"):
            return self._render_frame()

        if self.render_mode == "ansi":
            return np.array2string(self.grid)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None
        self.clock = None

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        pygame.font.init()

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((205, 193, 179))
        pix_square_size = self.window_size // self.size

        # 画网格
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (187, 173, 160),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=10,
            )
            pygame.draw.line(
                canvas,
                (187, 173, 160),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=10,
            )

        font_color = [(119, 110, 101), (249, 246, 242)]
        tile_color = [
            (238, 228, 218),
            (237, 224, 200),
            (242, 177, 121),
            (245, 149, 99),
            (246, 124, 95),
            (246, 94, 59),
            (237, 207, 114),
            (237, 204, 98),
            (237, 200, 80),
            (237, 197, 63),
            (237, 194, 46),
        ]
        # 画数字
        for x in range(self.size):
            for y in range(self.size):
                tile = self.grid[x, y]
                if tile > 0:
                    if tile <= 2048:
                        bcolor = tile_color[int(np.log2(tile) - 1)]
                    else:
                        bcolor = (0, 0, 0)

                    pygame.draw.rect(
                        canvas,
                        bcolor,
                        (
                            pix_square_size * y + 5,
                            pix_square_size * x + 5,
                            pix_square_size - 10,
                            pix_square_size - 10,
                        ),
                    )

                    if tile >= 100:
                        fsize = 48
                        if tile >= 1000:
                            fsize = 36
                            if tile >= 10000:
                                fsize = 24
                    else:
                        fsize = 64

                    text = str(tile)
                    font = pygame.font.SysFont("microsoftyahei", fsize, bold=True)
                    if tile <= 4:
                        fcolor = font_color[0]
                    else:
                        fcolor = font_color[1]
                    text_surface = font.render(text, True, fcolor)
                    text_rect = text_surface.get_rect()
                    text_rect.center = (
                        int(pix_square_size * (y + 0.5)),
                        int(pix_square_size * (x + 0.5)),
                    )
                    canvas.blit(text_surface, text_rect)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            assert self.window is not None and self.clock is not None
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def _move_left(self):
        new_grid = np.zeros_like(self.grid)
        for i in range(self.size):
            row = self.grid[i]
            non_zero = row[row != 0]
            merged = self._merge(non_zero)

            new_grid[i, : len(merged)] = merged
        return new_grid

    def _move_right(self):
        new_grid = np.zeros_like(self.grid)
        for i in range(self.size):
            row = self.grid[i]
            non_zero = row[::-1][row[::-1] != 0]
            merged = self._merge(non_zero)

            new_grid[i, self.size - len(merged) :] = merged[::-1]
        return new_grid

    def _move_up(self):
        new_grid = np.zeros_like(self.grid)
        for i in range(self.size):
            col = self.grid[:, i]
            non_zero = col[col != 0]
            merged = self._merge(non_zero)

            new_grid[: len(merged), i] = merged
        return new_grid

    def _move_down(self):
        new_grid = np.zeros_like(self.grid)
        for i in range(self.size):
            col = self.grid[:, i]
            non_zero = col[::-1][col[::-1] != 0]
            merged = self._merge(non_zero)

            new_grid[self.size - len(merged) :, i] = merged[::-1]
        return new_grid

    def _merge(self, row):
        merged = []
        skip = False
        for j in range(len(row)):
            if skip:
                skip = False
                continue
            if j + 1 < len(row) and row[j] == row[j + 1]:
                merged.append(row[j] * 2)
                skip = True
            else:
                merged.append(row[j])
        return np.asarray(merged, dtype=self.grid.dtype)

    def _check_game_over(self):
        if np.any(self.grid == 0):
            return False
        for i in range(self.size):
            for j in range(self.size):
                if (j < self.size - 1 and self.grid[i][j] == self.grid[i][j + 1]) or (
                    i < self.size - 1 and self.grid[i][j] == self.grid[i + 1][j]
                ):
                    return False
        return True

    def _add_random_tile(self):
        empty_pos = np.argwhere(self.grid == 0)
        if empty_pos.size > 0:
            r, c = empty_pos[self.np_random.integers(0, len(empty_pos))]
            self.grid[r, c] = 2 if self.np_random.random() < 0.9 else 4
