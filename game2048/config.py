"""Planner configuration shared by CLI and orchestration without importing JAX."""

from dataclasses import dataclass
from math import isfinite

INFERENCE_VERSIONS = {
    "greedy": "greedy-v1",
    "rollout": "feedback-rollout-v1",
    "shooting": "feedback-shooting-v1",
    "mppi": "feedback-mppi-v1",
}


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int = 3
    candidates: int = 16
    scenarios: int = 8
    iterations: int = 2
    temperature: float = 0.1
    update_rate: float = 0.5
    noise_std: float = 0.1
    verification_scenarios: int = 32

    def __post_init__(self):
        if (
            min(
                self.horizon,
                self.candidates,
                self.scenarios,
                self.iterations,
                self.verification_scenarios,
            )
            < 1
        ):
            raise ValueError("Planner counts must be positive")
        if not all(isfinite(x) for x in (self.temperature, self.update_rate, self.noise_std)):
            raise ValueError("Planner parameters must be finite")
        if self.temperature <= 0 or not 0 < self.update_rate <= 1:
            raise ValueError("Invalid planner temperature/update rate")
        if self.noise_std <= 0:
            raise ValueError("Noise standard deviation must be positive; use rollout for no bias")
