# Project conventions

- Use Python for project code. NumPy, Numba, JAX and their compiled backends are allowed; do not add Rust, C++ or CUDA sources.
- Use uv for Python environments and dependencies. Maintain pyproject.toml and uv.lock; run project commands with uv run.
- Use JAX for neural networks, GPU simulation and training.
- Build and verify the complete reachable-state teacher before supervised training. Teacher lookups are read-only; missing states are errors, never triggers for search or approximate labels.
- Inference must work without teacher files. Only game rules and neural network parameters may guide inference.
- Preserve exact D4 value invariance, legal-action masking and independent environment/planner random streams.
- Keep hot paths compiled and batched. Profile only to remove meaningful end-to-end bottlenecks; preserve a readable reference implementation.
- Keep Numba and JAX as the two accelerated rule implementations. Use the retained Gym environment as the reference; do not add a separate NumPy move/merge implementation. NumPy board encoding and symmetry helpers are allowed.
- After code changes, run and resolve failures from:
  - uv run ruff check .
  - uv run ruff format --check .
  - uv run ty check
- Run relevant correctness tests and report their results. Never describe unrun training or evaluation as completed.
