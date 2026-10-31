"""Initialization, symmetry, parameter counts and fixed-batch forward/backward timing."""

import platform
import time
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np

from game2048.ablations.config import ModelSpec
from game2048.ablations.models import initialize, parameter_count, parameter_hash
from game2048.ablations.optimization import batch_value_and_grad
from game2048.ablations.provenance import source_identity
from game2048.artifacts import atomic_json
from game2048.rules import permutations_d4


def profile_functions(model):
    @jax.jit
    def forward(p, x):
        return model.apply({"params": p}, x)

    @jax.jit
    def backward(p, x):
        return batch_value_and_grad(
            model, p, x, jnp.full(len(x), 8.0, jnp.float32), jnp.ones(len(x), jnp.bool_), 512
        )

    return forward, backward


def audit(output, batch_size=4096):
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    boards = jnp.asarray(
        np.random.default_rng(71).integers(0, 10, (batch_size, 3, 3), dtype=np.int32)
    )
    result = {
        "source": source_identity(),
        "platform": platform.platform(),
        "devices": [str(d) for d in jax.devices()],
        "jax_version": jax.__version__,
        "batch_size": batch_size,
        "gradient_microbatch_size": 512,
        "models": {},
        "initializations": {},
    }
    for seed in (0, 1, 2):
        _, a = initialize(ModelSpec(), seed)
        _, b = initialize(ModelSpec(), seed)
        assert parameter_hash(a) == parameter_hash(b)
        result["initializations"][str(seed)] = {"A": parameter_hash(a), "B": parameter_hash(b)}
    for architecture, width in (("d4", 32), ("plain", 83)):
        spec = ModelSpec(architecture=architecture, width=width)
        model, params = initialize(spec, 0)

        forward, backward = profile_functions(model)

        times = {}
        for name, fn in (("forward", forward), ("forward_backward", backward)):
            started = time.monotonic()
            compiled = fn.lower(params, boards).compile()
            compilation = time.monotonic() - started
            jax.block_until_ready(compiled(params, boards))
            samples = []
            for _ in range(3):
                started = time.monotonic()
                jax.block_until_ready(compiled(params, boards))
                samples.append(time.monotonic() - started)
            times[name] = {
                "compile_seconds": compilation,
                "execution_seconds": samples,
                "afterstates_per_second": batch_size / float(np.median(samples)),
            }
        probes = boards[: min(16, batch_size)]
        predictions = [
            np.asarray(forward(params, probes.reshape(-1, 9)[:, p].reshape(-1, 3, 3)))
            for p in permutations_d4()
        ]
        result["models"][architecture] = {
            "spec": asdict(spec),
            "parameter_count": parameter_count(params),
            "timing": times,
            "initial_orbit_max_range": float(np.ptp(np.stack(predictions), axis=0).max()),
        }
    d4 = result["models"]["d4"]["parameter_count"]
    plain = result["models"]["plain"]["parameter_count"]
    result["parameter_relative_difference"] = abs(plain - d4) / d4
    if result["parameter_relative_difference"] > 0.01:
        raise ValueError("Parameter mismatch exceeds 1%")
    atomic_json(output, result)
    return result
