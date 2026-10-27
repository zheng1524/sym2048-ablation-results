import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from game2048.network import ModelConfig, ValueNetwork, save_model
from game2048.training import finetune, lambda_returns


def test_lambda_return_terminal_boundaries():
    rewards = jnp.array([[0.0], [8.0], [0.0]])
    done = jnp.array([[False], [True], [False]])
    next_values = jnp.array([[6.0], [100.0], [4.0]])
    bootstrap = jnp.array([4.0])
    for trace, expected in ((0.0, [6, 8, 4]), (0.5, [7, 8, 4]), (1.0, [8, 8, 4])):
        result = lambda_returns(rewards, done, next_values, bootstrap, trace)
        np.testing.assert_allclose(result[:, 0], expected)
    gradient = jax.grad(lambda v: lambda_returns(rewards, done, v, bootstrap, 0.5).sum())(
        next_values
    )
    np.testing.assert_array_equal(gradient, 0)


def test_teacher_free_td_and_resume(tmp_path, monkeypatch):
    from game2048 import training

    def reject_teacher(*args, **kwargs):
        raise AssertionError("Online TD must not load a teacher")

    monkeypatch.setattr(training, "Teacher", reject_teacher)
    model = ValueNetwork(ModelConfig(width=2, blocks=0, head_width=4))
    params = model.init(jax.random.PRNGKey(1), jnp.zeros((1, 3, 3), jnp.int32))["params"]
    save_model(tmp_path / "initial", params, model.config, {})
    options = dict(updates=2, envs=2, horizon=2, validation_every=1, validation_games=2)
    result = finetune(tmp_path / "initial", tmp_path / "td", resume=False, **options)
    resumed = finetune(tmp_path / "initial", tmp_path / "td", resume=True, **options)
    assert result == resumed
    assert (tmp_path / "td" / "best" / "params.msgpack").exists()
    info_path = tmp_path / "td" / "training.json"
    info = json.loads(info_path.read_text())
    info["step"] -= 1
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="steps disagree"):
        finetune(tmp_path / "initial", tmp_path / "td", resume=True, **options)
