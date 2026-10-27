"""Historical MPPI-inspired priority-plan baseline, retained for result reproduction.

Intermediate actions use legality-only priority lists; the NN scores leaves only.
This baseline underperformed greedy control in the completed three-seed experiment.
"""

from typing import cast

import jax
import jax.numpy as jnp

from game2048 import jax_env
from game2048.config import PlannerConfig as PlannerConfig
from game2048.network import ValueNetwork, greedy_action
from game2048.rules import PRIORITIES


def make_actor(model: ValueNetwork, mode: str = "greedy", config: PlannerConfig = PlannerConfig()):
    if mode == "greedy":
        return jax.jit(lambda params, board, key: greedy_action(model, params, board, key))
    if mode not in ("mppi", "shooting"):
        raise ValueError(f"Unknown inference mode: {mode}")
    priorities = jnp.asarray(PRIORITIES)
    horizon = config.horizon
    iterations = config.iterations if mode == "mppi" else 1
    candidates = config.candidates if mode == "mppi" else config.candidates * config.iterations

    def evaluate(params, board, plans, random_key, scenarios):
        n = plans.shape[1]
        roots, root_legal = jax_env.afterstates(board)
        boards = jnp.broadcast_to(roots[:, None, None], (4, n, scenarios, 3, 3)).reshape(-1, 3, 3)
        sequence = jnp.repeat(plans[:, :, None], scenarios, axis=2).reshape(-1, horizon - 1)
        done = jnp.zeros(len(boards), dtype=jnp.bool_)
        terminal = jnp.zeros(len(boards), dtype=jnp.float32)

        def advance(carry, t):
            after, finished, outcome = carry
            # Shared random scenarios across candidates (common random numbers).
            keys = jax.random.split(jax.random.fold_in(random_key, t), scenarios)
            keys = jnp.tile(keys, (4 * n, 1))
            current = jax.vmap(jax_env.spawn)(after, keys)
            options, legal = jax.vmap(jax_env.afterstates)(current)
            dead = ~jnp.any(legal, axis=1)
            outcome = jnp.where(~finished & dead, jnp.max(current, axis=(1, 2)), outcome)
            finished = finished | dead
            order = priorities[sequence[:, t]]
            possible = jnp.take_along_axis(legal, order, axis=1)
            chosen = jnp.take_along_axis(order, jnp.argmax(possible, axis=1)[:, None], axis=1)[:, 0]
            nxt = options[jnp.arange(len(options)), chosen]
            return (jnp.where(finished[:, None, None], after, nxt), finished, outcome), None

        (boards, done, terminal), _ = jax.lax.scan(
            advance, (boards, done, terminal), jnp.arange(horizon - 1)
        )
        values = cast(jax.Array, model.apply({"params": params}, boards))
        values = jnp.where(done, terminal, values).reshape(4, n, scenarios).mean(axis=2)
        return jnp.where(root_legal[:, None], values, -1e9)

    def actor(params, board, key):
        probs = jnp.full((4, horizon - 1, 24), 1.0 / 24, dtype=jnp.float32)
        best = jnp.zeros((4, horizon - 1), dtype=jnp.int32)
        scores = jnp.full(4, -jnp.inf)

        def iteration(carry, t):
            distribution, best_plan, best_score = carry
            sample_key, eval_key = jax.random.split(jax.random.fold_in(key, t))
            logits = jnp.broadcast_to(
                jnp.log(distribution)[:, None], (4, candidates, horizon - 1, 24)
            )
            plans = jax.random.categorical(sample_key, logits, axis=-1).astype(jnp.int32)
            returns = evaluate(params, board, plans, eval_key, config.scenarios)
            weights = jax.nn.softmax(
                (returns - returns.max(axis=1, keepdims=True)) / config.temperature, axis=1
            )
            frequencies = jnp.einsum("ak,akth->ath", weights, jax.nn.one_hot(plans, 24))
            updated = (1 - config.update_rate) * distribution + config.update_rate * frequencies
            updated = (1 - config.exploration) * updated + config.exploration / 24
            winner = jnp.argmax(returns, axis=1)
            score = returns[jnp.arange(4), winner]
            proposal = plans[jnp.arange(4), winner]
            best_plan = jnp.where((score > best_score)[:, None], proposal, best_plan)
            return (updated, best_plan, jnp.maximum(score, best_score)), None

        (probs, best, _), _ = jax.lax.scan(iteration, (probs, best, scores), jnp.arange(iterations))
        finalists = jnp.stack([best, jnp.argmax(probs, axis=-1)], axis=1)
        results = evaluate(
            params,
            board,
            finalists,
            jax.random.fold_in(key, iterations),
            config.verification_scenarios,
        )
        root_scores = results.max(axis=1)
        _, legal = jax_env.afterstates(board)
        ties = legal & (root_scores == jnp.max(jnp.where(legal, root_scores, -jnp.inf)))
        noise = jax.random.uniform(jax.random.fold_in(key, iterations + 1), (4,))
        return jnp.argmax(jnp.where(ties, noise, -1.0)).astype(jnp.int32)

    return jax.jit(actor)
