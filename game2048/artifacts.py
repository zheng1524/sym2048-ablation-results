"""Small artifact writers, independent of the solver and numerical backends."""

import json
from pathlib import Path


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def atomic_json(path: Path, value: dict) -> None:
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    atomic_bytes(path, content.encode("utf-8"))
