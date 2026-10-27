"""Reference checks use Gym moves, explicit chance tapes, and a simple value oracle."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from game2048 import jax_env, rules
from game2048.config import PlannerConfig
from game2048.planner import (
    centered,
    choose_feedback,
    make_actor,
    make_rollout,
    rollout_randomness,
    weighted_mean,
)
from tests.reference import gym_afterstates


class SimpleValue:
    def apply(self, variables, boards):
        # D4 invariant, intentionally nonconstant over the four action afterstates.
        return (
            jnp.max(boards, axis=(1, 2)).astype(jnp.float32)
            + 0.07 * (boards == 0).sum(axis=(1, 2))
            + 0.013 * boards[:, 1, 1]
        )


def reference_value(board):
    return float(board.max() + 0.07 * (board == 0).sum() + 0.013 * board[1, 1])


def reference_spawn(board, key):
    draws = np.asarray(jax.random.uniform(key, (2,)))
    board = board.copy().reshape(-1)
    empty = np.flatnonzero(board == 0)
    if len(empty):
        position = empty[min(int(draws[0] * len(empty)), len(empty) - 1)]
        board[position] = 1 if draws[1] < 0.9 else 2
    return board.reshape(3, 3)


def reference_rollout(board, plans, spawn_keys, ties):
    roots, _ = gym_afterstates(board)
    result = np.zeros((4, plans.shape[1], spawn_keys.shape[1]))
    for action, root in enumerate(roots):
        for k in range(plans.shape[1]):
            for r in range(spawn_keys.shape[1]):
                after = root.copy()
                for t in range(plans.shape[2]):
                    current = reference_spawn(after, spawn_keys[t, r])
                    options, legal = gym_afterstates(current)
                    if not legal.any():
                        result[action, k, r] = current.max()
                        break
                    values = np.array([reference_value(option) for option in options], np.float32)
                    scores = np.where(legal, values + plans[action, k, t], -np.inf)
                    tied = legal & (scores == scores.max())
                    chosen = np.argmax(np.where(tied, ties[t, r], -1))
                    after = options[chosen]
                    result[action, k, r] = values[chosen]
    return result


@pytest.mark.parametrize("horizon", [2, 4])
def test_feedback_rollout_matches_gym_reference(horizon):
    model = SimpleValue()
    prepare, evaluate = make_rollout(model, horizon, scenarios=3)
    spawn, ties = rollout_randomness(jax.random.PRNGKey(10), horizon - 1, 3)
    rng = np.random.default_rng(19)
    plans = rng.normal(0, 0.1, (4, 2, horizon - 1, 4)).astype(np.float32)
    plans[:, 0] = 0  # Includes the exact zero-bias greedy continuation.
    for board in (
        np.array([[1, 1, 0], [2, 0, 3], [0, 0, 1]], np.int32),
        np.array([[1, 2, 3], [3, 4, 5], [5, 6, 0]], np.int32),
    ):
        first = prepare({}, jnp.asarray(board), spawn)
        actual = jax.jit(evaluate)({}, jnp.asarray(plans), first, spawn, ties)
        expected = reference_rollout(board, plans, np.asarray(spawn), np.asarray(ties))
        np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_terminal_scores_are_absorbing_and_bias_never_rewards_itself():
    model = SimpleValue()
    board = jnp.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
    prepare, evaluate = make_rollout(model, 4, 2)
    spawn, ties = rollout_randomness(jax.random.PRNGKey(4), 3, 2)
    first = prepare({}, board, spawn)
    values = evaluate({}, jnp.full((4, 2, 3, 4), 100.0), first, spawn, ties)
    np.testing.assert_array_equal(values, 7.0)
    assert (
        int(
            make_actor(model, "mppi", PlannerConfig(candidates=2, scenarios=2))(
                {}, board, jax.random.PRNGKey(2)
            )
        )
        == 0
    )

    options = jnp.zeros((1, 4, 3, 3), jnp.int32)
    branches = (
        options,
        jnp.array([[True, True, False, False]]),
        jnp.array([[2.0, 3.0, 99.0, 99.0]]),
        jnp.array([1]),
    )
    _, dead, value = choose_feedback(
        branches, jnp.array([[5.0, 0.0, 100.0, 100.0]]), jnp.ones((1, 4))
    )
    assert not bool(dead[0]) and float(value[0]) == 2.0


def test_importance_correction_targets_fixed_prior_after_proposal_moves():
    # For exp(c·delta) times N(0, P), the exact zero-sum Gaussian mean is c.
    target = jnp.array([0.2, -0.1, 0.1, -0.2])
    mean = jnp.array([[[-0.3, 0.4, -0.2, 0.1]]])
    noise = centered(jax.random.normal(jax.random.PRNGKey(17), (1, 40_000, 1, 4)))
    samples = mean[:, None] + noise
    returns = jnp.sum(samples * target, axis=(-1, -2))
    fitted = weighted_mean(samples, mean, returns, 1.0, 1.0)
    np.testing.assert_allclose(fitted[0, 0], target, atol=0.025)
    np.testing.assert_allclose(samples.sum(axis=-1), 0, atol=1e-6)


def test_horizon_one_and_random_stream_reproducibility():
    model = SimpleValue()
    board = jax_env.reset(jax.random.PRNGKey(3))
    key = jax.random.PRNGKey(19)
    greedy = make_actor(model)({}, board, key)
    for mode in ("rollout", "shooting", "mppi"):
        actual = make_actor(model, mode, PlannerConfig(horizon=1))({}, board, key)
        assert int(actual) == int(greedy)
    config = PlannerConfig(candidates=2, scenarios=2, iterations=2, verification_scenarios=3)
    for mode in ("rollout", "shooting", "mppi"):
        actor = make_actor(model, mode, config)
        assert int(actor({}, board, key)) == int(actor({}, board, key))
        # Actor calls cannot consume or mutate the externally supplied environment key.
        env_key = jax.random.PRNGKey(123)
        before = jax_env.spawn(board, env_key)
        actor({}, board, key).block_until_ready()
        np.testing.assert_array_equal(jax_env.spawn(board, env_key), before)


def test_feedback_action_equivariance_with_coupled_biases_and_ties():
    model = SimpleValue()
    board = np.array([[1, 1, 0], [3, 2, 0], [1, 0, 4]], np.int32)
    bias = np.array([0.1, -0.2, 0.3, -0.2], np.float32)
    noise = np.array([0.9, 0.1, 0.3, 0.2], np.float32)

    def decide(b, delta, tie):
        states, legal = jax_env.afterstates(jnp.asarray(b))
        values = model.apply({}, states)
        return choose_feedback(
            (states[None], legal[None], values[None], jnp.array([b.max()])),
            jnp.asarray(delta)[None],
            jnp.asarray(tie)[None],
        )

    selected, _, value = decide(board, bias, noise)
    for group, permutation in enumerate(rules.action_permutations()):
        transformed_bias, transformed_noise = np.empty(4), np.empty(4)
        transformed_bias[permutation], transformed_noise[permutation] = bias, noise
        transformed, _, score = decide(
            rules.transform(board, group), transformed_bias, transformed_noise
        )
        np.testing.assert_array_equal(
            transformed[0], rules.transform(np.asarray(selected[0]), group)
        )
        np.testing.assert_allclose(score, value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 0},
        {"noise_std": 0},
        {"temperature": float("nan")},
        {"update_rate": float("inf")},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        replace(PlannerConfig(), **kwargs)


def test_one_empty_cell_exact_spawn_expectation():
    """Enumerate the two chance outcomes, weighted 0.9/0.1, using Gym as oracle."""
    model = SimpleValue()
    board = np.array([[1, 0, 2], [3, 4, 5], [6, 7, 8]], np.int32)
    keys = jax.random.split(jax.random.PRNGKey(8), 100)
    draws = np.asarray(jax.vmap(lambda key: jax.random.uniform(key, (2,)))(keys))
    spawn = keys[
        jnp.array([np.flatnonzero(draws[:, 1] < 0.9)[0], np.flatnonzero(draws[:, 1] >= 0.9)[0]])
    ][None]
    ties = jnp.array([[[0.1, 0.3, 0.7, 0.9], [0.1, 0.3, 0.7, 0.9]]])
    prepare, evaluate = make_rollout(model, 2, 2)
    first = prepare({}, jnp.asarray(board), spawn)
    plans = jnp.zeros((4, 1, 1, 4))
    values = np.asarray(evaluate({}, plans, first, spawn, ties))[:, 0]
    roots, legal = gym_afterstates(board)
    for action in np.flatnonzero(legal):
        expected = []
        for tile in (1, 2):
            current = roots[action].copy()
            assert (current == 0).sum() == 1
            current[current == 0] = tile
            options, possible = gym_afterstates(current)
            expected.append(
                max(reference_value(b) for b in options[possible])
                if possible.any()
                else current.max()
            )
        np.testing.assert_allclose(
            values[action] @ np.array([0.9, 0.1]), np.dot(expected, [0.9, 0.1]), atol=1e-6
        )


def test_first_chance_deduplication_preserves_scenarios_and_feedback():
    model = SimpleValue()
    board = jnp.array([[1, 1, 0], [2, 0, 3], [0, 0, 1]])
    spawn, ties = rollout_randomness(jax.random.PRNGKey(26), 2, 32)
    prepare, evaluate = make_rollout(model, 3, 32)
    first = prepare({}, board, spawn)
    roots, _ = jax_env.afterstates(board)
    afters = jnp.broadcast_to(roots[:, None], (4, 32, 3, 3)).reshape(-1, 3, 3)
    current = jax.vmap(jax_env.spawn)(afters, jnp.tile(spawn[0], (4, 1)))
    states, legal = jax.vmap(jax_env.afterstates)(current)
    raw = (
        states,
        legal,
        model.apply({}, states.reshape(-1, 3, 3)).reshape(-1, 4),
        current.max(axis=(1, 2)),
    )
    for actual, expected in zip(first, raw, strict=True):
        np.testing.assert_array_equal(actual, expected)
    plans = jnp.zeros((4, 1, 2, 4))
    np.testing.assert_array_equal(
        evaluate({}, plans, first, spawn, ties), evaluate({}, plans, raw, spawn, ties)
    )
