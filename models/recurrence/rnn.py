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
from hps import Hyperparams


class RNN(nn.Module):
    H: Hyperparams
    d_hidden: int
    d_out: int
    reverse: bool = False

    @nn.compact
    def __call__(self, x, h_prev=None):
        H_rnn = self.H.rnn
        batch_size, seq_len, _ = x.shape

        def stable_init(rng, shape):
            r_min, r_max = H_rnn.init_minval_real, H_rnn.init_maxval_real
            u = jax.random.uniform(
                key=rng, shape=shape, minval=r_min, maxval=r_max
            )
            return logit(u)

        a_logit = self.param("a_logit", stable_init, (self.d_hidden,))
        a = expit(a_logit)

        if H_rnn.pos_embedding:
            x = jnp.concatenate(
                [x, get_sinusoidal_embeddings(batch_size, seq_len, 16)], -1
            )
        dx = nn.Dense(self.d_out)(x)
        x = nn.Dense(self.d_hidden)(x)
        if H_rnn.input_norm:
            x = sqrt_bound_derivative(1 - a**2, 200) * x
        a = jnp.broadcast_to(a, x.shape)
        sharding_spec = pallas.ShardingSpec(
            self.H._mesh, batch_axis_name="batch", sequence_axis_name="seq"
        )
        h, h_last = scan.linear_scan(
            x=x,
            a=a,
            h0=h_prev,
            reverse=self.reverse,
            scan_type=get_scan_implementation(H_rnn),
            sharding_spec=sharding_spec,
            unroll=128,
        )
        return (dx + nn.Dense(self.d_out)(h)) / 2, h_last

    def default_state(self, batch_size):
        return jnp.zeros((batch_size, self.d_hidden))
