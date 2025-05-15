import flax.linen as nn
import jax
import jax.numpy as jnp

from hps import Hyperparams
from models.efficient_scan import complex_lib, pallas, scan
from models.recurrence.common import (
    BlockDiagonalLinear,
    complex_to_merged,
    get_scan_implementation,
    get_sinusoidal_embeddings,
    merged_to_complex,
    real_imag_complex,
    sqrt_bound_derivative,
)


class RGLRU(nn.Module):
    H: Hyperparams
    d_hidden: int
    d_out: int
    reverse: bool = False

    @nn.compact
    def __call__(self, x, h_prev=None, pos_emb=None):
        H_rnn = self.H.rnn
        # TODO: implement BlockDiagonalLinear from RecurrentGemma
        batch_size, seq_len, d_in = x.shape
        d_hidden = self.d_hidden if H_rnn.only_real else self.d_hidden // 2

        def stable_init_real(rng, shape, eps=1e-8):
            r_min, r_max = H_rnn.init_minval_real, H_rnn.init_maxval_real
            u = jax.random.uniform(rng, shape=shape)
            a_real = 0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2 + eps)
            return jnp.log(jnp.exp(-a_real) - 1.0)

        def stable_init_imag(rng, shape):
            u = jax.random.uniform(rng, shape=shape)
            return jnp.pi * H_rnn.init_maxval_imag * u

        a_real_param = self.param("a_real_param", stable_init_real, (d_hidden,))
        if not H_rnn.only_real:
            a_imag_param = self.param(
                "a_imag_param", stable_init_imag, (d_hidden,)
            )

        if H_rnn.pos_embedding:
            if pos_emb is None:
                pos_emb = get_sinusoidal_embeddings(batch_size, seq_len, 16)
            x = jnp.concatenate([x, pos_emb], -1)
        x = nn.Dense(self.d_hidden)(x)

        gate_x = complex_lib.sigmoid(
            BlockDiagonalLinear(
                n_blocks=H_rnn.n_diag_blocks,
                d_input=self.d_hidden,
                d_output=d_hidden,
            )(x)
        )
        gate_a = complex_lib.sigmoid(
            BlockDiagonalLinear(
                n_blocks=H_rnn.n_diag_blocks,
                d_input=self.d_hidden,
                d_output=d_hidden,
            )(x)
        )

        log_a = -8.0 * gate_a * complex_lib.softplus(a_real_param)
        if H_rnn.only_real:
            a, a_squared = complex_lib.exp(log_a), complex_lib.exp(2 * log_a)
        else:
            log_a_imag = a_imag_param * gate_a
            log_a_complex = real_imag_complex(H_rnn, log_a, log_a_imag)
            a = complex_lib.exp(log_a_complex)
            a_squared = complex_lib.abs_squared(a)

        x = merged_to_complex(H_rnn, x)
        h_prev = (
            merged_to_complex(H_rnn, h_prev) if h_prev is not None else None
        )

        x = gate_x * x
        # TODO: placement of norm corresponding to RGLRU
        # reconsider doing it before gating
        if H_rnn.input_norm:
            x = sqrt_bound_derivative(1 - a_squared, 200) * x

        sharding_spec = pallas.ShardingSpec(
            self.H._mesh(batch_size),
            batch_axis_name="batch",
            sequence_axis_name="seq",
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
        h = complex_to_merged(H_rnn, h)
        h_last = complex_to_merged(H_rnn, h_last)
        return nn.Dense(self.d_out)(h), h_last

    def default_state(self, batch_size):
        return jnp.zeros((batch_size, self.d_hidden))
