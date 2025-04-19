import jax.numpy as jnp


def get_sinusoidal_embeddings(timesteps, d):
    max_period = 10_000
    half = d // 2

    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half) / half)
    args = timesteps[:, None] * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    return embedding
