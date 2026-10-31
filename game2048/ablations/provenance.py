"""Source fingerprints and strict ownership of new experiment directories."""

import hashlib
import json
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from game2048.artifacts import atomic_json

ROOT = Path(__file__).resolve().parents[2]


def source_identity():
    files = (
        sorted((ROOT / "game2048").rglob("*.py"))
        + sorted((ROOT / "scripts").rglob("*.py"))
        + [
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
        ]
    )
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    packages = {}
    for name in (
        "jax",
        "jaxlib",
        "flax",
        "optax",
        "numpy",
        "jax-cuda12-plugin",
        "jax-cuda12-pjrt",
        "nvidia-cudnn-cu12",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        # Delivery-only commits must not invalidate an otherwise identical resume.
        "commit": subprocess.check_output(
            [
                "git",
                "log",
                "-1",
                "--format=%H",
                "--",
                "game2048",
                "scripts",
                "pyproject.toml",
                "uv.lock",
            ],
            cwd=ROOT,
            text=True,
        ).strip(),
        "source_sha256": digest,
        "files": hashes,
        "runtime": {
            "python": platform.python_version(),
            "packages": packages,
            "flags": {
                name: os.environ.get(name)
                for name in (
                    "JAX_PLATFORMS",
                    "XLA_FLAGS",
                    "JAX_DEFAULT_MATMUL_PRECISION",
                    "XLA_PYTHON_CLIENT_PREALLOCATE",
                    "LD_LIBRARY_PATH",
                )
            },
        },
    }


def claim(directory: Path, filename: str, specification: dict):
    path = directory / filename
    if path.exists():
        if json.loads(path.read_text()) != specification:
            raise ValueError(f"Configuration/source mismatch: {directory}")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing unrecognized nonempty directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(path, specification)
