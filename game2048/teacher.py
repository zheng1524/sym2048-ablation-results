"""Complete reachable-state teacher. Training never invokes enumeration or search."""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from numba import set_num_threads

from game2048 import kernels
from game2048.artifacts import atomic_json
from game2048.rules import action_permutations, canonical, decode

RULES_VERSION = 1


def atomic_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npy.tmp")
    with tmp.open("wb") as file:
        np.save(file, value, allow_pickle=False)
    os.replace(tmp, path)


def read_array(path: Path, dtype=np.uint64) -> np.ndarray:
    return np.load(path, mmap_mode="r") if path.exists() else np.empty(0, dtype=dtype)


def _initial(size: int) -> dict[int, np.ndarray]:
    states: dict[int, set[int]] = {4: set(), 6: set(), 8: set()}
    for i in range(size * size):
        for j in range(i + 1, size * size):
            for a in (1, 2):
                for b in (1, 2):
                    board = np.zeros((size, size), dtype=np.int32)
                    board.flat[i], board.flat[j] = a, b
                    code, _ = canonical(board)
                    states[(1 << a) + (1 << b)].add(int(code))
    return {s: np.array(sorted(values), dtype=np.uint64) for s, values in states.items()}


def build(directory: Path, size: int = 3, threads: int = 16, chunk: int = 8192) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    layers = directory / "layers"
    layers.mkdir(exist_ok=True)
    set_num_threads(min(threads, os.cpu_count() or 1))
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["size"] != size or manifest["rules_version"] != RULES_VERSION:
            raise ValueError("Existing teacher has incompatible rules or board size")
    else:
        manifest: dict[str, Any] = {
            "rules_version": RULES_VERSION,
            "size": size,
            "objective": "expected_terminal_max_exponent",
            "spawn_probabilities": [0.9, 0.1],
            "includes_post_maximum_states": True,
            "enumerated_through": 2,
            "enumeration_complete": False,
            "solve_complete": False,
            "verified": False,
        }
        for total, codes in _initial(size).items():
            atomic_array(layers / f"{total}.npy", codes)
        atomic_json(manifest_path, manifest)
    if manifest.get("solve_complete"):
        return manifest
    lines, sym = kernels.make_tables(size)
    total = int(manifest["enumerated_through"]) + 2
    while not manifest["enumeration_complete"]:
        started = time.monotonic()
        codes = read_array(layers / f"{total}.npy")
        parts: list[list[np.ndarray]] = [[], []]
        for offset in range(0, len(codes), chunk):
            generated = kernels.successors(codes[offset : offset + chunk], size, lines, sym)
            for tile in range(2):
                values = generated[:, tile].ravel()
                parts[tile].append(np.unique(values[values != 0]))
        for tile in range(2):
            target = layers / f"{total + 2 + 2 * tile}.npy"
            previous = read_array(target)
            if parts[tile]:
                merged = np.unique(np.concatenate([previous, *parts[tile]]))
                atomic_array(target, merged)
        manifest["enumerated_through"] = total
        if total >= 8 and not any(
            len(read_array(layers / f"{total + delta}.npy")) for delta in (2, 4)
        ):
            manifest["enumeration_complete"] = True
        atomic_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "phase": "enumerate",
                    "sum": total,
                    "states": len(codes),
                    "seconds": round(time.monotonic() - started, 3),
                }
            ),
            flush=True,
        )
        total += 2

    sums = sorted(int(path.stem) for path in layers.glob("*.npy"))
    solved = directory / "solved"
    solved.mkdir(exist_ok=True)
    for total in reversed(sums):
        if (solved / f"{total}.done").exists():
            continue
        started = time.monotonic()
        codes = read_array(layers / f"{total}.npy")
        next2 = read_array(layers / f"{total + 2}.npy")
        next4 = read_array(layers / f"{total + 4}.npy")
        v2 = read_array(solved / f"{total + 2}.v.npy", np.float64)
        v4 = read_array(solved / f"{total + 4}.v.npy", np.float64)
        q, legal, missing = kernels.solve_layer(codes, next2, v2, next4, v4, size, lines, sym)
        if missing.sum():
            raise RuntimeError(
                f"Teacher closure failure at sum {total}: {missing.sum()} successors"
            )
        if len(codes) and not legal.any(axis=1).all():
            raise RuntimeError("A terminal state was unexpectedly included in the table")
        atomic_array(solved / f"{total}.q.npy", q)
        atomic_array(solved / f"{total}.legal.npy", legal)
        atomic_array(solved / f"{total}.v.npy", q.max(axis=1) if len(q) else np.empty(0))
        (solved / f"{total}.done").touch()
        print(
            json.dumps(
                {
                    "phase": "solve",
                    "sum": total,
                    "states": len(codes),
                    "seconds": round(time.monotonic() - started, 3),
                }
            ),
            flush=True,
        )

    count = sum(len(read_array(layers / f"{s}.npy")) for s in sums)
    if count >= np.iinfo(np.int32).max:
        raise RuntimeError("Teacher exceeds the supported int32 row-index capacity")
    keys = np.lib.format.open_memmap(
        directory / "keys.npy", mode="w+", dtype=np.uint64, shape=(count,)
    )
    q_all = np.lib.format.open_memmap(
        directory / "q.npy", mode="w+", dtype=np.float64, shape=(count, 4)
    )
    masks = np.lib.format.open_memmap(
        directory / "legal.npy", mode="w+", dtype=np.bool_, shape=(count, 4)
    )
    offset = 0
    for total in sums:
        codes = read_array(layers / f"{total}.npy")
        end = offset + len(codes)
        keys[offset:end] = codes
        q_all[offset:end] = read_array(solved / f"{total}.q.npy")
        masks[offset:end] = read_array(solved / f"{total}.legal.npy")
        offset = end
    keys.flush()
    q_all.flush()
    masks.flush()
    capacity = 1 << (max(2, count * 2) - 1).bit_length()
    slots = np.lib.format.open_memmap(
        directory / "index.npy", mode="w+", dtype=np.int32, shape=(capacity,)
    )
    slots[:] = -1
    kernels.build_index(keys, slots)
    slots.flush()
    manifest.update(
        {"solve_complete": True, "states": count, "index_capacity": capacity, "verified": False}
    )
    atomic_json(manifest_path, manifest)
    return manifest


class Teacher:
    def __init__(self, directory: Path, *, require_verified: bool = True):
        self.directory = directory
        self.manifest = json.loads((directory / "manifest.json").read_text())
        if self.manifest["rules_version"] != RULES_VERSION:
            raise ValueError("Incompatible teacher rules")
        if not self.manifest["solve_complete"]:
            raise ValueError("Complete teacher construction is required before lookup")
        if require_verified and not self.manifest.get("verified"):
            raise ValueError("Teacher must pass full verification before training")
        self.size = int(self.manifest["size"])
        self.keys = np.load(directory / "keys.npy", mmap_mode="r")
        self.q = np.load(directory / "q.npy", mmap_mode="r")
        self.legal = np.load(directory / "legal.npy", mmap_mode="r")
        self.slots = np.load(directory / "index.npy", mmap_mode="r")
        self.lines, self.sym = kernels.make_tables(self.size)

    def lookup(self, boards: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boards = np.asarray(boards, dtype=np.int32)
        if boards.ndim != 3 or boards.shape[1:] != (self.size, self.size):
            raise ValueError("Expected a batch of decision boards")
        if np.any(boards < 0) or np.any(boards > self.size * self.size + 1):
            raise ValueError("Out-of-range board exponent")
        rows, groups = kernels.lookup_boards(boards, self.keys, self.slots, self.size, self.sym)
        if np.any(rows < 0):
            raise KeyError(f"Missing/non-decision teacher states: {boards[rows < 0][:3].tolist()}")
        actions = action_permutations()[groups]
        return (
            rows,
            np.take_along_axis(self.q[rows], actions, axis=1),
            np.take_along_axis(self.legal[rows], actions, axis=1),
        )


def verify(directory: Path, *, threads: int = 16) -> dict:
    """Full pass over every transition; no sampled completeness assertion."""
    set_num_threads(min(threads, os.cpu_count() or 1))
    teacher = Teacher(directory, require_verified=False)
    manifest = teacher.manifest
    manifest["verified"] = False
    atomic_json(directory / "manifest.json", manifest)
    layers, solved = directory / "layers", directory / "solved"
    residual = 0.0
    count = 0
    digest = hashlib.sha256()
    for total in sorted(int(p.stem) for p in layers.glob("*.npy")):
        keys = read_array(layers / f"{total}.npy")
        q, legal, missing = kernels.solve_layer(
            keys,
            read_array(layers / f"{total + 2}.npy"),
            read_array(solved / f"{total + 2}.v.npy", np.float64),
            read_array(layers / f"{total + 4}.npy"),
            read_array(solved / f"{total + 4}.v.npy", np.float64),
            teacher.size,
            teacher.lines,
            teacher.sym,
        )
        end = count + len(keys)
        if missing.sum() or not np.array_equal(keys, teacher.keys[count:end]):
            raise RuntimeError(f"Closure or consolidation failure at sum {total}")
        if not np.array_equal(legal, teacher.legal[count:end]):
            raise RuntimeError(f"Legal masks disagree at sum {total}")
        if len(keys):
            stored_q = teacher.q[count:end]
            stored_v = read_array(solved / f"{total}.v.npy")
            if (
                not all(np.isfinite(a).all() for a in (q, stored_q, stored_v))
                or not legal.any(axis=1).all()
            ):
                raise RuntimeError("Invalid teacher labels")
            residual = max(residual, float(np.max(np.abs(q - stored_q))))
            residual = max(residual, float(np.max(np.abs(stored_v - stored_q.max(axis=1)))))
        digest.update(keys.tobytes())
        digest.update(np.asarray(teacher.q[count:end]).tobytes())
        count = end
    if count != len(teacher.keys) or residual > 1e-10:
        raise RuntimeError(f"Teacher verification failed: {count=}, {residual=}")
    if kernels.verify_index(teacher.keys, teacher.slots):
        raise RuntimeError("Teacher lookup index is incomplete or inconsistent")
    initial = np.concatenate(list(_initial(teacher.size).values()))
    teacher.lookup(np.stack([decode(k, teacher.size) for k in initial]))
    manifest.update(
        {
            "verified": True,
            "bellman_residual": residual,
            "content_sha256": digest.hexdigest(),
            "verified_states": count,
        }
    )
    atomic_json(directory / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)
    return manifest
