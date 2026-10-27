from functools import lru_cache

import numpy as np
import pytest

from game2048 import rules
from game2048.teacher import Teacher, build, verify
from tests.reference import gym_afterstates


@lru_cache(None)
def reference_value(flat):
    board = np.asarray(flat, dtype=np.int32).reshape(2, 2)
    afters, legal = gym_afterstates(board)
    if not legal.any():
        return float(board.max())
    values = []
    for after in afters[legal]:
        empty = np.argwhere(after == 0)
        score = 0.0
        for r, c in empty:
            for exponent, probability in ((1, 0.9), (2, 0.1)):
                nxt = after.copy()
                nxt[r, c] = exponent
                score += probability * reference_value(tuple(nxt.ravel())) / len(empty)
        values.append(score)
    return max(values)


def test_complete_table_against_independent_solver(tmp_path):
    build(tmp_path, size=2, threads=2)
    with pytest.raises(ValueError, match="verification"):
        Teacher(tmp_path)
    result = verify(tmp_path, threads=2)
    teacher = Teacher(tmp_path)
    assert result["verified"] and result["bellman_residual"] < 1e-12
    assert any(rules.decode(code, 2).max() == 5 for code in teacher.keys)
    for key, q, mask in zip(teacher.keys, teacher.q, teacher.legal, strict=True):
        board = rules.decode(key, 2)
        assert q[mask].max() == pytest.approx(reference_value(tuple(board.ravel())), abs=1e-12)
        boards = np.stack([rules.transform(board, g) for g in range(8)])
        _, transformed_q, transformed_mask = teacher.lookup(boards)
        for g in range(8):
            # Compare via inverse action permutation: rotate the original action labels.
            inverse = np.argsort(rules.action_permutations()[g])
            np.testing.assert_allclose(transformed_q[g], q[inverse], atol=1e-12)
            np.testing.assert_array_equal(transformed_mask[g], mask[inverse])
    with pytest.raises(KeyError):
        teacher.lookup(np.zeros((1, 2, 2), np.int32))
    assert build(tmp_path, size=2, threads=2)["verified"]
    # Non-finite stored labels must fail verification even in the first layer.
    corrupt = np.load(tmp_path / "q.npy", mmap_mode="r+")
    corrupt[0, 0] = np.nan
    corrupt.flush()
    with pytest.raises(RuntimeError, match="Invalid teacher labels"):
        verify(tmp_path, threads=2)
