"""Memory-bounded exact-batch gradients: one clipping/AdamW step after accumulation."""

import jax
import jax.numpy as jnp


def batch_value_and_grad(model, params, boards, targets, valid, microbatch_size):
    if microbatch_size < 1:
        raise ValueError("Gradient microbatch size must be positive")
    count = len(boards)
    chunk = min(count, microbatch_size)
    padding = (-count) % chunk
    denominator = jnp.maximum(valid.sum(), 1)

    def loss(p, x, y, mask):
        error = jnp.where(mask, model.apply({"params": p}, x) - y, 0.0)
        return jnp.sum(error**2) / denominator

    gradient = jax.value_and_grad(loss)
    if count <= chunk:
        return gradient(params, boards, targets, valid)
    x = jnp.pad(boards, ((0, padding), (0, 0), (0, 0))).reshape(-1, chunk, 3, 3)
    y = jnp.pad(targets, ((0, padding),)).reshape(-1, chunk)
    mask = jnp.pad(valid, ((0, padding),), constant_values=False).reshape(-1, chunk)

    def accumulate(carry, batch):
        total, grads = carry
        value, current_grads = gradient(params, *batch)
        return (total + value, jax.tree.map(jnp.add, grads, current_grads)), None

    initial = (jnp.float32(0), jax.tree.map(jnp.zeros_like, params))
    (value, grads), _ = jax.lax.scan(accumulate, initial, (x, y, mask))
    return value, grads
