"""Frozen protocol; smoke runs occupy a separate namespace and use full-size models."""

from dataclasses import asdict, dataclass, field, replace

from game2048.network import ModelConfig

VERSION = "distillation-d4-20261031-v1"
GROUPS = ("d4_distill_td", "d4_scratch_td", "plain_distill_td")


@dataclass(frozen=True)
class ModelSpec:
    architecture: str = "d4"
    width: int = 32
    blocks: int = 2
    head_width: int = 64

    def __post_init__(self):
        if self.architecture not in ("d4", "plain"):
            raise ValueError("Unknown architecture")
        if min(self.width, self.head_width) < 1 or self.blocks < 0:
            raise ValueError("Invalid model dimensions")

    def dimensions(self):
        return ModelConfig(self.width, self.blocks, self.head_width)


@dataclass(frozen=True)
class SupervisedConfig:
    batch_size: int = 1024
    learning_rate: float = 3e-4
    max_updates: int = 100_000
    warmup_updates: int = 5000
    validation_every: int = 5000
    collect_every: int = 1000
    collection_envs: int = 128
    collection_steps: int = 256
    pool_capacity: int = 262_144


@dataclass(frozen=True)
class TDConfig:
    updates: int = 10_000
    learning_rate: float = 3e-5
    trace_lambda: float = 0.8
    envs: int = 128
    horizon: int = 32
    validation_every: int = 1000


@dataclass(frozen=True)
class Protocol:
    seeds: tuple[int, ...] = (0, 1, 2)
    smoke: bool = False
    validation_seed: int = 400_000
    validation_games: int = 512
    test_seed: int = 500_000
    test_games: int = 2000
    evaluation_batch: int = 256
    supervised: SupervisedConfig = field(default_factory=SupervisedConfig)
    td: TDConfig = field(default_factory=TDConfig)
    version: str = VERSION

    def __post_init__(self):
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or min(self.seeds) < 0:
            raise ValueError("Training seeds must be distinct nonnegative integers")
        for cfg in (self.supervised, self.td):
            for name, value in asdict(cfg).items():
                if name in ("warmup_updates", "trace_lambda"):
                    continue
                if value <= 0:
                    raise ValueError(f"{name} must be positive")
        if not 0 <= self.td.trace_lambda <= 1 or self.supervised.warmup_updates < 0:
            raise ValueError("Invalid lambda or warmup")
        if min(self.validation_games, self.test_games, self.evaluation_batch) < 1:
            raise ValueError("Evaluation counts must be positive")
        if set(range(self.validation_seed, self.validation_seed + self.validation_games)) & set(
            range(self.test_seed, self.test_seed + self.test_games)
        ):
            raise ValueError("Validation and test seeds overlap")

    def json(self):
        result = asdict(self)
        result["seeds"] = list(self.seeds)
        return result


def protocol(seeds=(0, 1, 2), smoke=False):
    base = Protocol(seeds=tuple(seeds), smoke=smoke)
    if not smoke:
        return base
    return replace(
        base,
        validation_games=4,
        test_games=4,
        evaluation_batch=4,
        supervised=SupervisedConfig(
            batch_size=8,
            max_updates=3,
            warmup_updates=1,
            validation_every=1,
            collect_every=1,
            collection_envs=2,
            collection_steps=2,
            pool_capacity=16,
        ),
        td=TDConfig(updates=2, envs=2, horizon=2, validation_every=1),
    )


def model_spec(group):
    if group not in (*GROUPS, "d4_scratch_practical", "lr_pilot"):
        raise ValueError("Unknown training group")
    return ModelSpec(architecture="plain", width=83) if group == GROUPS[2] else ModelSpec()


def from_json(data):
    data = dict(data)
    data["seeds"] = tuple(data["seeds"])
    data["supervised"] = SupervisedConfig(**data["supervised"])
    data["td"] = TDConfig(**data["td"])
    return Protocol(**data)
