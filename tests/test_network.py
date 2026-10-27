import jax
import jax.numpy as jnp
import numpy as np

from game2048 import jax_env, rules
from game2048.evaluation import evaluate
from game2048.network import ModelConfig, ValueNetwork, load_model, save_model
from game2048.planner import PlannerConfig, make_actor
from game2048.training import _decode_words


def test_d4_network_gradient_checkpoint_and_independent_inference(tmp_path):
    model = ValueNetwork(ModelConfig(width=4, blocks=1, head_width=8))
    boards = np.random.default_rng(3).integers(0, 10, (5, 3, 3), dtype=np.int32)
    params = model.init(jax.random.PRNGKey(1), jnp.asarray(boards))["params"]
    predict = jax.jit(lambda p, x: model.apply({"params": p}, x))
    for _ in range(2):
        expected = predict(params, boards)
        for g in range(8):
            transformed = np.stack([rules.transform(b, g) for b in boards])
            np.testing.assert_allclose(predict(params, transformed), expected, atol=2e-5)
        grad = jax.grad(lambda p: jnp.mean((predict(p, boards) - 8) ** 2))(params)
        params = jax.tree.map(lambda p, d: p - 1e-3 * d, params, grad)
    save_model(tmp_path, params, model.config, {"test": True})
    loaded, weights, _ = load_model(tmp_path)
    np.testing.assert_allclose(
        loaded.apply({"params": weights}, boards), predict(params, boards), atol=2e-6
    )
    start = jax_env.reset(jax.random.PRNGKey(7))
    _, valid = jax_env.afterstates(start)
    config = PlannerConfig(
        horizon=3, candidates=2, scenarios=2, iterations=2, verification_scenarios=2
    )
    for mode in ("greedy", "shooting", "mppi"):
        action = make_actor(loaded, mode, config)(weights, start, jax.random.PRNGKey(1))
        assert bool(valid[action])
    result = evaluate(loaded, weights, games=2, batch_size=2)
    assert result["mean_moves"] > 0


def test_word_encoding_and_holdout_orbits():
    boards = np.random.default_rng(9).integers(0, 11, (128, 3, 3), dtype=np.int32)
    codes = np.array([rules.encode(b) for b in boards], np.uint64)
    words = np.stack(
        [(codes & 0xFFFFFFFF).astype(np.uint32), (codes >> 32).astype(np.uint32)], axis=1
    )
    np.testing.assert_array_equal(_decode_words(words), boards)
    mask = jax_env.validation_mask(jnp.asarray(boards))
    for g in range(8):
        transformed = np.stack([rules.transform(b, g) for b in boards])
        np.testing.assert_array_equal(jax_env.validation_mask(jnp.asarray(transformed)), mask)
