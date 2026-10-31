import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization

from game2048 import jax_env, rules
from game2048.ablations.config import ModelSpec, protocol
from game2048.ablations.evaluation import evaluate, make_game_runner, verify_evaluation
from game2048.ablations.models import (
    PlainValueNetwork,
    initialize,
    load_model,
    parameter_count,
    parameter_hash,
    save_model,
)
from game2048.ablations.provenance import claim
from game2048.ablations.state import random_streams, take
from game2048.ablations.supervised import augment, masked_loss, supervised_batch
from game2048.ablations.td import train
from game2048.network import ValueNetwork, action_values


def tiny_protocol():
    p = protocol((0,), True)
    return replace(
        p,
        validation_games=2,
        test_games=2,
        evaluation_batch=2,
        td=replace(p.td, updates=3, validation_every=2),
    )


def test_factory_counts_initialization_and_structural_audit(tmp_path):
    _, d4 = initialize(ModelSpec(), 0)
    plain_model, plain = initialize(ModelSpec("plain", 83), 0)
    assert parameter_count(d4) == 304_833
    assert parameter_count(plain) == 305_403
    assert abs(parameter_count(plain) / parameter_count(d4) - 1) < 0.01
    for seed in (0, 1, 2):
        old, a = initialize(ModelSpec(), seed)
        _, b = initialize(ModelSpec(), seed)
        assert parameter_hash(a) == parameter_hash(b)
        boards = jax.random.randint(jax.random.PRNGKey(seed), (6, 3, 3), 0, 10)
        np.testing.assert_array_equal(
            old.apply({"params": a}, boards), ValueNetwork().apply({"params": a}, boards)
        )
    assert isinstance(plain_model, PlainValueNetwork)
    keys = set(plain)
    assert not any("Group" in name for name in keys)
    assert plain["Dense_1"]["kernel"].shape == (9 * 83, 64)
    boards = np.random.default_rng(17).integers(0, 10, (8, 3, 3), dtype=np.int32)
    variants = np.stack(
        [
            np.asarray(
                plain_model.apply(
                    {"params": plain}, np.stack([rules.transform(b, g) for b in boards])
                )
            )
            for g in range(8)
        ]
    )
    assert np.count_nonzero(np.ptp(variants, axis=0) > 1e-4) >= 4
    save_model(tmp_path / "legacy", d4, ModelSpec(), {})
    path = tmp_path / "legacy" / "model.json"
    data = json.loads(path.read_text())
    del data["architecture"]
    path.write_text(json.dumps(data))
    assert isinstance(load_model(tmp_path / "legacy")[0], ValueNetwork)


@pytest.mark.parametrize("architecture", ["d4", "plain"])
def test_checkpoint_jit_boundary_and_actual_evaluation(tmp_path, architecture):
    spec = ModelSpec(architecture, 2, 0, 4)
    model, params = initialize(spec, 0)
    boards = jax.random.randint(jax.random.PRNGKey(6), (7, 3, 3), 0, 10)
    predict = jax.jit(lambda p, x: model.apply({"params": p}, x))
    np.testing.assert_allclose(
        model.apply({"params": params}, boards), predict(params, boards), rtol=1e-5, atol=2e-5
    )
    save_model(tmp_path / "model", params, spec, {"seed": 0})
    loaded, restored, _, restored_spec = load_model(tmp_path / "model")
    assert type(loaded) is type(model) and restored_spec == spec
    np.testing.assert_array_equal(predict(restored, boards), predict(params, boards))
    boundary = jnp.zeros((2, 3, 3), jnp.int32).at[:, 0, 0].set(10)
    np.testing.assert_array_equal(predict(params, boundary), 10.0)
    board = jax_env.reset(jax.random.PRNGKey(1))
    q, legal = action_values(loaded, restored, board)
    assert np.isneginf(np.asarray(q)[~np.asarray(legal)]).all()
    result = evaluate(
        spec, restored, games=2, seed=500_000, batch_size=2, output=tmp_path / "eval", smoke=True
    )
    assert verify_evaluation(tmp_path / "eval") == result
    if architecture == "plain":
        assert make_game_runner(spec) is not make_game_runner(replace(spec, architecture="d4"))
    csv = tmp_path / "eval" / "games.csv"
    csv.write_text(csv.read_text().replace("500000,", "500001,"))
    with pytest.raises(ValueError, match="hash"):
        verify_evaluation(tmp_path / "eval")


def test_augmented_labels_masks_and_independent_streams():
    boards = np.random.default_rng(19).integers(0, 9, (128, 3, 3), dtype=np.int32)
    codes = np.array([rules.encode(b) for b in boards], np.uint64)
    words = jnp.asarray(
        np.stack([codes.astype(np.uint32), (codes >> 32).astype(np.uint32)], axis=1)
    )
    labels = jnp.arange(len(boards) * 4, dtype=jnp.float32).reshape(-1, 4)
    masks = jax.vmap(jax_env.afterstates)(jnp.asarray(boards))[1]
    rows = jnp.arange(len(boards))
    x, targets, valid = supervised_batch(words, labels, masks, rows, jax.random.PRNGKey(8))
    afters, _ = jax.vmap(jax_env.afterstates)(jnp.asarray(boards))
    original = afters.reshape(-1, 3, 3)
    np.testing.assert_array_equal(targets, labels.reshape(-1))
    np.testing.assert_array_equal(valid, masks.reshape(-1) & ~jax_env.validation_mask(original))
    np.testing.assert_array_equal(jax_env.validation_mask(x), jax_env.validation_mask(original))
    for g in range(8):
        transformed = np.stack([rules.transform(b, g) for b in np.asarray(original)])
        np.testing.assert_array_equal(
            jax_env.validation_mask(jnp.asarray(transformed)), jax_env.validation_mask(x)
        )
    # Every augmented scalar afterstate stays in its original orbit, without parent/action mixing.
    np.testing.assert_array_equal(jax_env.canonical_boards(x), jax_env.canonical_boards(original))
    n = 16_000
    unique = jnp.arange(9).reshape(1, 3, 3).repeat(n, axis=0)
    versions = np.asarray(augment(unique, jax.random.PRNGKey(1))).reshape(n, 9)
    for p in rules.permutations_d4():
        assert abs(np.mean(np.all(versions == p, axis=1)) - 1 / 8) < 0.015
    one, two = random_streams(0), random_streams(0)
    for _ in range(5):
        take(one, "augment")
    np.testing.assert_array_equal(take(one, "sample"), take(two, "sample"))
    assert (
        float(
            masked_loss(jnp.array([2.0, 4.0]), jnp.array([1.0, jnp.nan]), jnp.array([True, False]))
        )
        == 1.0
    )


def test_scratch_td_no_teacher_and_exact_resume(tmp_path, monkeypatch):
    import game2048.teacher as teacher_module

    def forbidden(*args, **kwargs):
        raise AssertionError("Scratch TD must not construct or query teacher")

    monkeypatch.setattr(teacher_module.Teacher, "__init__", forbidden)
    monkeypatch.setattr(teacher_module.Teacher, "lookup", forbidden)
    spec, p = ModelSpec("d4", 2, 0, 4), tiny_protocol()
    options = dict(group="d4_scratch_td", seed=0, spec=spec, protocol=p)
    train(tmp_path / "continuous", **options)
    train(tmp_path / "split", stop_after=1, **options)
    assert not (tmp_path / "split" / "complete.json").exists()
    train(tmp_path / "split", resume=True, **options)

    def read(name):
        return serialization.msgpack_restore((tmp_path / name / "recovery.msgpack").read_bytes())

    a, b = read("continuous"), read("split")
    for field in ("state", "streams", "boards", "best_params"):
        for x, y in zip(jax.tree.leaves(a[field]), jax.tree.leaves(b[field]), strict=True):
            np.testing.assert_array_equal(x, y)
    ia, ib = json.loads(a["info"]), json.loads(b["info"])
    assert ia["step"] == ib["step"] == 3
    assert [x["step"] for x in ia["history"]] == [0, 2, 3]
    assert [x["validation"]["mean_max_exponent"] for x in ia["history"]] == [
        x["validation"]["mean_max_exponent"] for x in ib["history"]
    ]
    with pytest.raises(ValueError, match="mismatch"):
        train(tmp_path / "split", resume=True, **{**options, "spec": replace(spec, width=3)})
    with pytest.raises(ValueError, match="mismatch"):
        train(
            tmp_path / "split",
            resume=True,
            **{**options, "protocol": replace(p, td=replace(p.td, learning_rate=1e-4))},
        )
    with pytest.raises(ValueError, match="cannot load"):
        train(tmp_path / "new", upstream=tmp_path / "continuous" / "best", **options)
    # Inference import and real games work without any teacher asset directory.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import game2048.ablations.evaluation; "
            "assert 'game2048.teacher' not in sys.modules",
        ],
        check=True,
    )


def test_refuse_unrecognized_output_and_incompatible_protocol(tmp_path):
    (tmp_path / "user-file").write_text("preserve")
    with pytest.raises(FileExistsError, match="unrecognized"):
        claim(tmp_path, "request.json", {})
    with pytest.raises(ValueError, match="overlap"):
        replace(protocol(), test_seed=400_001)


def test_supervised_pool_resume_and_upstream_hash_guard(tmp_path, monkeypatch):
    from game2048.ablations import supervised

    # A small deterministic label fixture tests recovery mechanics only. Real smoke
    # supervision separately uses the complete verified 3x3 teacher.
    boards = np.asarray(jax.vmap(jax_env.reset)(jax.random.split(jax.random.PRNGKey(4), 8)))
    codes = np.array([rules.encode(b) for b in boards], np.uint64)
    words = jnp.asarray(
        np.stack([codes.astype(np.uint32), (codes >> 32).astype(np.uint32)], axis=1)
    )
    masks = jax.vmap(jax_env.afterstates)(jnp.asarray(boards))[1]
    labels = jnp.full((len(boards), 4), 8.0, jnp.float32)
    arrays = (words, labels, masks, jnp.arange(8), jnp.array([0]), jnp.array([8]))
    teacher = SimpleNamespace(lookup=lambda x: (np.arange(len(x)) % 8, None, None))
    monkeypatch.setattr(supervised, "checked_teacher", lambda _: teacher)
    monkeypatch.setattr(supervised, "_teacher_arrays", lambda _: arrays)
    p = tiny_protocol()
    p = replace(
        p,
        supervised=replace(
            p.supervised, max_updates=3, validation_every=2, warmup_updates=0, collect_every=1
        ),
    )
    spec = ModelSpec("d4", 2, 0, 4)
    options = dict(group="d4_distill_td", seed=0, spec=spec, protocol=p)
    supervised.train(tmp_path / "unavailable", tmp_path / "whole", **options)
    supervised.train(tmp_path / "unavailable", tmp_path / "split", stop_after=1, **options)
    supervised.train(tmp_path / "unavailable", tmp_path / "split", resume=True, **options)
    whole, split = [
        serialization.msgpack_restore((tmp_path / name / "recovery.msgpack").read_bytes())
        for name in ("whole", "split")
    ]
    for field in ("state", "streams", "pool", "best_params"):
        for a, b in zip(jax.tree.leaves(whole[field]), jax.tree.leaves(split[field]), strict=True):
            np.testing.assert_array_equal(a, b)
    assert json.loads(split["info"])["environment_transitions"] == 12
    options["upstream"] = tmp_path / "whole" / "best"
    train(tmp_path / "td", **options)
    with pytest.raises(ValueError, match="protocol"):
        train(
            tmp_path / "wrong-protocol",
            **{**options, "protocol": replace(p, td=replace(p.td, learning_rate=1e-4))},
        )
    _, params, metadata, _ = load_model(options["upstream"])
    changed = jax.tree.map(lambda x: x + 0.01, params)
    save_model(options["upstream"], changed, spec, metadata)
    with pytest.raises(ValueError, match="mismatch"):
        train(tmp_path / "td", resume=True, **options)
