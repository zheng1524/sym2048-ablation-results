import hashlib
import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from game2048 import evaluation
from game2048.cli import make_parser
from game2048.config import PlannerConfig
from game2048.evaluation_jobs import cached_evaluation
from game2048.network import ValueNetwork


def test_cli_and_inference_imports_are_solver_free():
    subprocess.run(
        [sys.executable, "-c", "import sys; import game2048.cli; assert 'jax' not in sys.modules"],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import game2048.evaluation; import game2048.comparison; "
            "assert 'game2048.teacher' not in sys.modules",
        ],
        check=True,
    )
    parser = make_parser()
    args = parser.parse_args(["replay", "model", "replay.json", "--noise-std", "0.2"])
    assert not hasattr(args, "games") and args.noise_std == 0.2
    with pytest.raises(SystemExit):
        parser.parse_args(["replay", "model", "replay.json", "--games", "2"])


def test_evaluation_completion_and_cache_validation(tmp_path, monkeypatch):
    # Isolate artifact publication from expensive simulation; no training or model inference.
    class Runner:
        def lower(self, *_):
            return self

        def compile(self):
            return self

        def __call__(self, params, seeds):
            progress = json.loads((directory / "progress.json").read_text())
            assert progress["status"] == "running"
            n = len(seeds)
            return np.full(n, 8), np.full(n, 400), np.full(n, 180), np.zeros(n, bool)

    monkeypatch.setattr(evaluation, "make_game_runner", lambda *args: Runner())
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "params.msgpack").write_bytes(b"artifact-test-placeholder")
    info = {
        "checkpoint_sha256": hashlib.sha256(b"artifact-test-placeholder").hexdigest(),
        "checkpoint_metadata": {"step": 0},
    }
    directory = tmp_path / "eval"
    planner = PlannerConfig()
    metrics = evaluation.evaluate(
        ValueNetwork(),
        {},
        games=2,
        batch_size=2,
        seed=50,
        mode="mppi",
        output=directory,
        checkpoint_info=info,
    )
    assert cached_evaluation(directory, checkpoint, "mppi", 2, 50, planner) == metrics
    old = {key: value for key, value in metrics.items() if key != "algorithm_version"}
    (directory / "metrics.json").write_text(json.dumps(old))
    assert cached_evaluation(directory, checkpoint, "mppi", 2, 50, planner) is None
    (directory / "metrics.json").write_text(json.dumps(metrics))
    assert cached_evaluation(directory, checkpoint, "mppi", 3, 50, planner) is None
    assert cached_evaluation(directory, checkpoint, "mppi", 2, 51, planner) is None
    assert (
        cached_evaluation(directory, checkpoint, "mppi", 2, 50, replace(planner, horizon=5)) is None
    )
    (checkpoint / "params.msgpack").write_bytes(b"changed-weights")
    assert cached_evaluation(directory, checkpoint, "mppi", 2, 50, planner) is None
    (checkpoint / "params.msgpack").write_bytes(b"artifact-test-placeholder")
    (directory / "games.csv").write_text(
        "seed,max_exponent,max_tile,tile_sum,moves\n50,8,256,400,180\n"
    )
    assert cached_evaluation(directory, checkpoint, "mppi", 2, 50, planner) is None

    def interrupted_write(*args):
        raise OSError("Interrupted CSV write")

    monkeypatch.setattr(evaluation.csv.DictWriter, "writerows", interrupted_write)
    with pytest.raises(OSError, match="Interrupted"):
        evaluation.evaluate(ValueNetwork(), {}, games=2, batch_size=2, output=directory)
    assert not (directory / "metrics.json").exists()


@pytest.mark.parametrize("games,batch_size", [(0, 2), (2, 0), (2, -1)])
def test_invalid_evaluation_counts(games, batch_size):
    with pytest.raises(ValueError, match="positive"):
        evaluation.evaluate(ValueNetwork(), {}, games=games, batch_size=batch_size)
