"""Planner configuration shared by CLI and orchestration without importing JAX."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int = 8
    candidates: int = 64
    scenarios: int = 4
    iterations: int = 3
    temperature: float = 0.1
    update_rate: float = 0.5
    exploration: float = 0.05
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
            or self.horizon < 2
        ):
            raise ValueError("Planner counts must be positive and horizon >= 2")
        if self.temperature <= 0 or not 0 < self.update_rate <= 1:
            raise ValueError("Invalid planner temperature/update rate")
        if not 0 < self.exploration <= 1:
            raise ValueError("Exploration must be in (0, 1]")
