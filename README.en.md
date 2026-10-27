# 3×3 2048 neural agent

See [README.md](README.md) for installation, commands, architecture and experiment settings.

A fully precomputed exact teacher supervises a D4-invariant JAX afterstate value network.
Teacher-free TD(lambda) and categorical MPPI-inspired planning are implemented separately.
Inference only requires network parameters and game rules. Environments use uv; checks use
Ruff, ty and pytest.

The completed three-seed experiment improved greedy play with TD(lambda). The current
shooting/MPPI priority-plan baselines performed substantially worse than greedy and remain
available for reproducing those results. See [RUN_STATUS.md](RUN_STATUS.md) for the evidence.
Numba and JAX implement accelerated game rules; the retained Gym environment provides the
reference. `rules.py` contains encoding and symmetry helpers only.
