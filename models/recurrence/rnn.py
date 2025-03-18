import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.special import expit, logit

from models.efficient_scan import pallas, scan
from models.recurrence.common import (
    get_scan_implementation,
    get_sinusoidal_embeddings,
    sqrt_bound_derivative,
)
from models.recurrence.hps import RNNHyperparams

_mesh = jax.make_mesh((jax.device_count(),), ("batch",))
SHARDING_SPEC = pallas.ShardingSpec(mesh=_mesh)


class RNN(nn.Module):
    H: RNNHyperparams
    d_hidden: int
    d_out: int
    reverse: bool = False

    @nn.compact
    def __call__(self, x, h_prev=None):
        batch_size, seq_len, _ = x.shape

        def stable_init(rng, shape):
            r_min, r_max = self.H.init_minval_real, self.H.init_maxval_real
            u = jax.random.uniform(
                key=rng, shape=shape, minval=r_min, maxval=r_max
            )
            return logit(u)

        a_logit = self.param("a_logit", stable_init, (self.d_hidden,))
        a = expit(a_logit)

        if self.H.pos_embedding:
            x = jnp.concatenate(
                [x, get_sinusoidal_embeddings(batch_size, seq_len, 16)], -1
            )
        dx = nn.Dense(self.d_out)(x)
        x = nn.Dense(self.d_hidden)(x)
        if self.H.input_norm:
            x = sqrt_bound_derivative(1 - a**2, 200) * x
        a = jnp.broadcast_to(a, x.shape)
        h, h_last = scan.linear_scan(
            x=x,
            a=a,
            h0=h_prev,
            reverse=self.reverse,
            scan_type=get_scan_implementation(self.H),
            sharding_spec=SHARDING_SPEC,
            unroll=128,
        )
        return (dx + nn.Dense(self.d_out)(h)) / 2, h_last

    def default_state(self, batch_size):
        return jnp.zeros((batch_size, self.d_hidden))
