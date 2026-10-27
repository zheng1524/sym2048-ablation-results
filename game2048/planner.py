"""Neural feedback rollout and importance-weighted Gaussian policy search.

The model values afterstates. Biases affect simulated actions, never their score.
Environment chance is averaged before weighting control candidates. This is a
finite-sample, MPPI-inspired policy search, not an optimality guarantee.
"""

from typing import cast

import jax
import jax.numpy as jnp

from game2048 import jax_env
from game2048.config import INFERENCE_VERSIONS
from game2048.config import PlannerConfig as PlannerConfig
from game2048.network import ValueModel, greedy_action


def centered(x):
    """Project four action biases onto the three-dimensional zero-sum subspace."""
    return x - x.mean(axis=-1, keepdims=True)


def weighted_mean(plans, mean, returns, noise_std, temperature):
    """Fit the fixed-prior Gibbs target from *random proposal samples only*.

    Prior and proposal have covariance sigma² I on the zero-sum subspace.
    Their density normalizers cancel, including the subspace determinant.
    """
    log_ratio = jnp.sum((plans - mean[:, None]) ** 2 - plans**2, axis=(-2, -1)) / (2 * noise_std**2)
    logits = returns / temperature + log_ratio
    weights = jax.nn.softmax(logits - logits.max(axis=1, keepdims=True), axis=1)
    return centered(jnp.einsum("ak,aktd->atd", weights, plans))


def rollout_randomness(key, steps, scenarios):
    """Common scenarios, independent spawn and tie streams; no environment keys."""
    spawn = jax.random.split(jax.random.fold_in(key, 0), steps * scenarios).reshape(
        steps, scenarios, 2
    )
    ties = jax.random.uniform(jax.random.fold_in(key, 1), (steps, scenarios, 4))
    return spawn, ties


def _branches(model, params, afters, spawn_keys):
    current = jax.vmap(jax_env.spawn)(afters, spawn_keys)
    return _current_branches(model, params, current)


def _current_branches(model, params, current):
    options, legal = jax.vmap(jax_env.afterstates)(current)
    values = cast(jax.Array, model.apply({"params": params}, options.reshape(-1, 3, 3)))
    return options, legal, values.reshape(-1, 4), jnp.max(current, axis=(1, 2))


def choose_feedback(branches, bias, tie_noise):
    """Select with biased values, return the unmodified bootstrap/terminal value."""
    options, legal, values, terminal = branches
    scores = jnp.where(legal, values + bias, -jnp.inf)
    ties = legal & (scores == scores.max(axis=-1, keepdims=True))
    chosen = jnp.argmax(jnp.where(ties, tie_noise, -1.0), axis=-1)
    rows = jnp.arange(len(options))
    dead = ~legal.any(axis=-1)
    return options[rows, chosen], dead, jnp.where(dead, terminal, values[rows, chosen])


def make_rollout(model, horizon, scenarios):
    """Build preparation and scoring functions; output shape is root × plan × chance.

    Preparation evaluates the first spawned states once per root/scenario. All
    candidates and optimizer iterations reuse these network evaluations.
    """
    steps = horizon - 1

    def prepare(params, board, spawn_keys):
        roots, _ = jax_env.afterstates(board)
        afters = jnp.broadcast_to(roots[:, None], (4, scenarios, 3, 3)).reshape(-1, 3, 3)
        keys = jnp.tile(spawn_keys[0], (4, 1))
        if scenarios <= 18:
            return _branches(model, params, afters, keys)
        # A root has at most 9 empty cells × 2 tile values, even when many
        # hundreds of scenarios were sampled. Score each distinct first state
        # once, then restore the original scenario order (and multiplicity).
        current = jax.vmap(jax_env.spawn)(afters, keys).reshape(4, scenarios, 9)
        unique, inverse = jax.vmap(
            lambda x: jnp.unique(x, axis=0, size=18, return_inverse=True, fill_value=0)
        )(current)
        branches = _current_branches(model, params, unique.reshape(-1, 3, 3))
        indices = (inverse + 18 * jnp.arange(4)[:, None]).reshape(-1)
        return jax.tree.map(lambda x: x[indices], branches)

    def evaluate(params, plans, first, spawn_keys, tie_noise):
        n = plans.shape[1]
        count = 4 * n * scenarios

        def expand(x):
            tail = x.shape[1:]
            x = x.reshape(4, scenarios, *tail)
            return jnp.broadcast_to(x[:, None], (4, n, scenarios, *tail)).reshape(count, *tail)

        def bias_at(t):
            return jnp.broadcast_to(plans[:, :, None, t], (4, n, scenarios, 4)).reshape(count, 4)

        def ties_at(t):
            return jnp.tile(tie_noise[t], (4 * n, 1))

        initial = choose_feedback(jax.tree.map(expand, first), bias_at(0), ties_at(0))

        def advance(carry, t):
            after, finished, outcome = carry
            keys = jnp.tile(spawn_keys[t], (4 * n, 1))
            branches = _branches(model, params, after, keys)
            nxt, dead, value = choose_feedback(branches, bias_at(t), ties_at(t))
            return (
                jnp.where(finished[:, None, None], after, nxt),
                finished | dead,
                jnp.where(finished, outcome, value),
            ), None

        (_, _, values), _ = jax.lax.scan(advance, initial, jnp.arange(1, steps))
        return values.reshape(4, n, scenarios)

    return prepare, evaluate


def make_actor(model: ValueModel, mode: str = "greedy", config: PlannerConfig = PlannerConfig()):
    if mode not in INFERENCE_VERSIONS:
        raise ValueError(f"Unknown inference mode: {mode}")
    if mode == "greedy" or config.horizon == 1:
        return jax.jit(lambda params, board, key: greedy_action(model, params, board, key))

    prepare, evaluate = make_rollout(model, config.horizon, config.scenarios)
    prepare_verify, evaluate_verify = make_rollout(
        model, config.horizon, config.verification_scenarios
    )
    iterations = config.iterations if mode == "mppi" else 1
    candidates = config.candidates if mode == "mppi" else config.candidates * config.iterations

    def actor(params, board, key):
        zero = jnp.zeros((4, config.horizon - 1, 4), jnp.float32)
        # Separate controls, optimization chance, verification chance, and root ties.
        sample_key, chance_key, verify_key, tie_key = jax.random.split(key, 4)
        spawn_keys, ties = rollout_randomness(chance_key, config.horizon - 1, config.scenarios)
        first = prepare(params, board, spawn_keys)
        if mode == "rollout":
            scores = evaluate(params, zero[:, None], first, spawn_keys, ties).mean(axis=2)[:, 0]
        else:

            def iteration(carry, t):
                mean, best, best_score = carry
                noise = centered(
                    jax.random.normal(
                        jax.random.fold_in(sample_key, t),
                        (4, candidates, config.horizon - 1, 4),
                    )
                )
                samples = mean[:, None] + config.noise_std * noise
                plans = jnp.concatenate([samples, zero[:, None], mean[:, None]], axis=1)
                # All rounds use the same chance tape; incumbent scores are comparable.
                returns = evaluate(params, plans, first, spawn_keys, ties).mean(axis=2)
                fitted = weighted_mean(
                    samples, mean, returns[:, :candidates], config.noise_std, config.temperature
                )
                updated = centered((1 - config.update_rate) * mean + config.update_rate * fitted)
                winner = jnp.argmax(returns, axis=1)
                score = returns[jnp.arange(4), winner]
                proposal = plans[jnp.arange(4), winner]
                best = jnp.where((score > best_score)[:, None, None], proposal, best)
                return (updated, best, jnp.maximum(score, best_score)), None

            (mean, best, _), _ = jax.lax.scan(
                iteration, (zero, zero, jnp.full(4, -jnp.inf)), jnp.arange(iterations)
            )
            finalists = jnp.stack([zero, best, mean], axis=1)
            verify_spawn, verify_ties = rollout_randomness(
                verify_key, config.horizon - 1, config.verification_scenarios
            )
            verify_first = prepare_verify(params, board, verify_spawn)
            scores = (
                evaluate_verify(params, finalists, verify_first, verify_spawn, verify_ties)
                .mean(axis=2)
                .max(axis=1)
            )
        _, legal = jax_env.afterstates(board)
        scores = jnp.where(legal, scores, -jnp.inf)
        tied = legal & (scores == scores.max())
        # An already terminal board returns the unused placeholder action 0.
        return jnp.argmax(jnp.where(tied, jax.random.uniform(tie_key, (4,)), -1.0)).astype(
            jnp.int32
        )

    return jax.jit(actor)
