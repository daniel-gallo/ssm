import jax.numpy as jnp
import flax.linen as nn

import hps


class VSSM(nn.Module):
    H: hps.Hyperparams

    @nn.compact
    def __call__(self, x, rng):
        loss = jnp.sum(nn.linear.Dense(8)(x) ** 2)
        metrics = {'loss': loss}
        return loss, metrics
