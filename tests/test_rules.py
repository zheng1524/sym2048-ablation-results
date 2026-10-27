from itertools import product

import jax
import jax.numpy as jnp
import numpy as np

from game2048 import jax_env, kernels, rules
from game2048.envs import Game2048Env
from tests.reference import exponents, gym_afterstates


def test_all_three_cell_lines():
    rows = np.asarray(list(product(range(16), repeat=3)), dtype=np.int32)
    env = Game2048Env(size=3)
    expected = []
    for row in rows:
        env.grid = np.zeros((3, 3), np.int32)
        env.grid[0] = np.where(row > 0, 1 << row, 0)
        expected.append(exponents(env._move_left()[0]))
    expected = np.stack(expected)
    np.testing.assert_array_equal(np.asarray(jax.jit(jax_env.merge_lines)(rows)), expected)
    lines, _ = kernels.make_tables(3)
    for code, merged in enumerate(expected):
        # rows are lexicographic; table codes store the first cell in the low nibble.
        index = int(rules.encode(rows[code]))
        assert lines[index] == (rules.encode(merged) if merged.max() <= 15 else 0)


def test_gym_numba_jax_and_d4():
    rng = np.random.default_rng(2)
    boards = rng.integers(0, 10, (100, 3, 3), dtype=np.int32)
    lines, sym = kernels.make_tables(3)
    accelerated, mask = jax.jit(jax.vmap(jax_env.afterstates))(jnp.asarray(boards))
    directions = rules.action_permutations()
    for i, board in enumerate(boards):
        board = np.asarray(board)
        expected, legal = gym_afterstates(board)
        np.testing.assert_array_equal(accelerated[i], expected)
        np.testing.assert_array_equal(mask[i], legal)
        code = rules.encode(board)
        for a in range(4):
            result = kernels.move_code(code, a, 3, lines)
            np.testing.assert_array_equal(rules.decode(result), expected[a])
        canon, _ = kernels.canonical_code(code, 3, sym)
        assert canon == rules.canonical(board)[0]
        for g in range(8):
            transformed = gym_afterstates(rules.transform(board, g))[0]
            for a in range(4):
                np.testing.assert_array_equal(
                    transformed[int(directions[g, a])],
                    rules.transform(expected[a], g),
                )


def test_spawn_probabilities_and_empty_positions():
    board = np.array([[1, 0, 0], [2, 3, 0], [0, 4, 0]], np.int32)
    keys = jax.random.split(jax.random.PRNGKey(1), 20_000)
    generated = np.asarray(jax.jit(jax.vmap(jax_env.spawn, in_axes=(None, 0)))(board, keys))
    assert np.all(generated[:, board != 0] == board[board != 0])
    additions = generated[:, board == 0]
    assert np.all((additions != 0).sum(axis=1) == 1)
    assert abs(np.mean(additions.max(axis=1) == 2) - 0.1) < 0.01
    np.testing.assert_allclose((additions != 0).mean(axis=0), 0.2, atol=0.015)


def test_gym_seed_rewards_and_reference_moves():
    a, b = Game2048Env(size=3, target=256), Game2048Env(size=3, target=256)
    np.testing.assert_array_equal(a.reset(seed=1)[0], b.reset(seed=1)[0])
    for action in [0, 1, 3, 2] * 10:
        np.testing.assert_array_equal(a.step(action)[0], b.step(action)[0])
    a.grid = np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]], np.int32)
    old = a.grid.copy()
    observed, reward, _, _, _ = a.step(0)
    np.testing.assert_array_equal(observed, old)
    assert reward == 0
    a.grid = np.array([[256, 0, 0], [0, 2, 0], [0, 0, 0]], np.int32)
    assert not a._check_game_over()
    for action, method in enumerate((a._move_left, a._move_right, a._move_up, a._move_down)):
        exp = np.where(a.grid > 0, np.log2(np.maximum(a.grid, 1)), 0).astype(np.int32)
        expected = np.asarray(jax_env.afterstates(jnp.asarray(exp))[0][action])
        np.testing.assert_array_equal(method(), np.where(expected > 0, 1 << expected, 0))


def test_gym_render_contract(capsys, monkeypatch):
    ansi = Game2048Env(render_mode="ansi")
    ansi.reset(seed=2)
    rendered = ansi.render()
    assert isinstance(rendered, str) and "[" in rendered
    assert capsys.readouterr().out == ""
    rgb = Game2048Env(render_mode="rgb_array")
    rgb.reset(seed=2)
    frame = rgb.render()
    assert frame.shape == (512, 512, 3) and frame.dtype == np.uint8
    human = Game2048Env(render_mode="human")
    calls = []
    monkeypatch.setattr(human, "_render_frame", lambda: calls.append(True))
    human.render()
    assert calls == [True]
    for env in (ansi, rgb, human):
        env.close()
