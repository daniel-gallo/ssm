import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

from hps import Hyperparams
from models.efficient_scan import complex_lib, pallas, scan
from models.efficient_scan.common import ScanType
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
    reverse: bool = False
    temporal_scale: int = 1  # rename
    feature_scale: int = 1  # rename

    @nn.compact
    def __call__(self, x, h_prev=None, pos_emb=None, sampling=False):
        H_rnn = self.H.rnn
        # TODO: implement BlockDiagonalLinear from RecurrentGemma
        batch_size, seq_len, d_in = x.shape
        d_hidden = (
            H_rnn.d_hidden * self.feature_scale
            if H_rnn.adaptive_d
            else H_rnn.d_hidden
        )
        assert d_hidden % H_rnn.dtype_scaling == 0
        d_inner = d_hidden // H_rnn.dtype_scaling

        def stable_init_real(rng, shape, eps=1e-8):
            r_min, r_max = H_rnn.init_minval_real, H_rnn.init_maxval_real
            u = jax.random.uniform(rng, shape=shape)
            a_real = 0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2 + eps)
            return jnp.log(jnp.exp(-a_real) - 1.0)

        def stable_init_imag(rng, shape):
            u = jax.random.uniform(rng, shape=shape)
            scale = (
                H_rnn.init_maxval_imag * self.temporal_scale
                if H_rnn.adaptive_phase
                else H_rnn.init_maxval_imag
            )
            return jnp.pi * scale * u

        a_real_param = self.param("a_real_param", stable_init_real, (d_inner,))
        match H_rnn.dtype_hidden:
            case "complex":
                a_imag_param = self.param(
                    "a_imag_param", stable_init_imag, (d_inner,)
                )
            case "quaternion":
                a_imag_param = self.param(
                    "a_imag_param", stable_init_imag, (d_inner, 3)
                )

        if H_rnn.pos_embedding:
            if pos_emb is None:
                pos_emb = get_sinusoidal_embeddings(batch_size, seq_len, 16)
            x = jnp.concatenate([x, pos_emb], -1)
        x = nn.Dense(d_hidden)(x)

        match H_rnn.gate_x:
            case "sigmoid":
                gate_x = complex_lib.sigmoid(
                    BlockDiagonalLinear(
                        n_blocks=H_rnn.n_diag_blocks,
                        d_input=d_hidden,
                        d_output=d_inner,
                    )(x)
                )
            case "tanh":
                gate_x = nn.tanh(
                    BlockDiagonalLinear(
                        n_blocks=H_rnn.n_diag_blocks,
                        d_input=d_hidden,
                        d_output=d_inner,
                    )(x)
                )
            case "mlp":
                gate_x = BlockDiagonalLinear(
                    n_blocks=H_rnn.n_diag_blocks,
                    d_input=d_hidden,
                    d_output=d_inner,
                )(x)
            case "none":
                gate_x = jnp.ones((batch_size, seq_len, d_inner))
            case _:
                raise ValueError(f"Unknown gate_x: {H_rnn.gate_x}")

        match H_rnn.gate_a_real:
            case "sigmoid":
                gate_a = complex_lib.sigmoid(
                    BlockDiagonalLinear(
                        n_blocks=H_rnn.n_diag_blocks,
                        d_input=d_hidden,
                        d_output=d_inner,
                    )(x)
                )
                gate_a_real = gate_a
            case "mlp":
                gate_a = BlockDiagonalLinear(
                    n_blocks=H_rnn.n_diag_blocks,
                    d_input=d_hidden,
                    d_output=d_inner,
                )(x)
                gate_a_real = complex_lib.softplus(gate_a)
            case "none":
                gate_a = jnp.ones((batch_size, seq_len, d_inner))
                gate_a_real = gate_a
            case _:
                raise ValueError(f"Unknown gate_a_real: {H_rnn.gate_a_real}")

        if H_rnn.dtype_hidden in ["complex", "quaternion"]:
            d_gate = d_inner if H_rnn.dtype_hidden == "complex" else d_inner * 3
            match H_rnn.gate_a_imag:
                case "same":
                    gate_a_imag = gate_a
                    if H_rnn.dtype_hidden == "quaternion":
                        gate_a_imag = jnp.expand_dims(gate_a_imag, axis=-1)

                case "sigmoid":
                    gate_a_imag = complex_lib.sigmoid(
                        BlockDiagonalLinear(
                            n_blocks=H_rnn.n_diag_blocks,
                            d_input=d_hidden,
                            d_output=d_gate,
                        )(x)
                    )
                    if H_rnn.dtype_hidden == "quaternion":
                        gate_a_imag = rearrange(
                            gate_a_imag, "... (d i)->... d i", i=3
                        )

                case "tanh":
                    gate_a_imag = nn.tanh(
                        BlockDiagonalLinear(
                            n_blocks=H_rnn.n_diag_blocks,
                            d_input=d_hidden,
                            d_output=d_gate,
                        )(x)
                    )
                    if H_rnn.dtype_hidden == "quaternion":
                        gate_a_imag = rearrange(
                            gate_a_imag, "... (d i)->... d i", i=3
                        )
                case "mlp":
                    gate_a_imag = BlockDiagonalLinear(
                        n_blocks=H_rnn.n_diag_blocks,
                        d_input=d_hidden,
                        d_output=d_gate,
                    )(x)
                    if H_rnn.dtype_hidden == "quaternion":
                        gate_a_imag = rearrange(
                            gate_a_imag, "... (d i)->... d i", i=3
                        )
                case "none":
                    gate_a_imag = jnp.ones((batch_size, seq_len, d_inner))
                    if H_rnn.dtype_hidden == "quaternion":
                        gate_a_imag = jnp.expand_dims(gate_a_imag, axis=-1)
                case _:
                    raise ValueError(
                        f"Unknown gate_a_imag: {H_rnn.gate_a_imag}"
                    )

        if H_rnn.gate_a_real == "mlp":
            log_a = H_rnn.log_a_scale * complex_lib.softplus(
                a_real_param * gate_a_real
            )
        else:
            log_a = (
                H_rnn.log_a_scale
                * gate_a_real
                * complex_lib.softplus(a_real_param)
            )

        match H_rnn.dtype_hidden:
            case "real":
                a = complex_lib.exp(log_a)
            case "complex":
                log_a_imag = a_imag_param * gate_a_imag
                log_a_complex = real_imag_complex(H_rnn, log_a, log_a_imag)
                a = complex_lib.exp(log_a_complex)
            case "quaternion":
                a_real = complex_lib.exp(log_a)
                log_a_imag = a_imag_param * gate_a_imag
                alpha, beta, gamma = (
                    log_a_imag[..., 0],
                    log_a_imag[..., 1],
                    log_a_imag[..., 2],
                )
                imag_w = jnp.cos(alpha / 2) * jnp.cos(beta / 2) * jnp.cos(
                    gamma / 2
                ) + jnp.sin(alpha / 2) * jnp.sin(beta / 2) * jnp.sin(gamma / 2)
                imag_x = jnp.sin(alpha / 2) * jnp.cos(beta / 2) * jnp.cos(
                    gamma / 2
                ) - jnp.cos(alpha / 2) * jnp.sin(beta / 2) * jnp.sin(gamma / 2)
                imag_y = jnp.cos(alpha / 2) * jnp.sin(beta / 2) * jnp.cos(
                    gamma / 2
                ) + jnp.sin(alpha / 2) * jnp.cos(beta / 2) * jnp.sin(gamma / 2)
                imag_z = jnp.cos(alpha / 2) * jnp.cos(beta / 2) * jnp.sin(
                    gamma / 2
                ) - jnp.sin(alpha / 2) * jnp.sin(beta / 2) * jnp.cos(gamma / 2)
                a_imag = jnp.stack([imag_x, imag_y, imag_z], axis=0)
                a = real_imag_complex(H_rnn, imag_w, a_imag) * a_real
            case _:
                raise ValueError(f"Unknown dtype_hidden: {H_rnn.dtype_hidden}")

        x = merged_to_complex(H_rnn, x)
        h_prev = (
            merged_to_complex(H_rnn, h_prev) if h_prev is not None else None
        )

        x = gate_x * x
        # TODO: placement of norm corresponding to RGLRU
        # reconsider doing it before gating
        match H_rnn.input_norm:
            case "fixed":
                a_squared = complex_lib.exp(2 * log_a)
                norm_factor = sqrt_bound_derivative(1 - a_squared, 200)
            case "none":
                norm_factor = 1.0
            case _:
                raise ValueError(f"Unknown gate_a_imag: {H_rnn.gate_a_imag}")
        x = norm_factor * x

        sharding_spec = (
            None
            if sampling
            else pallas.ShardingSpec(
                self.H._mesh(batch_size),
                batch_axis_name="batch",
                sequence_axis_name="seq",
            )
        )
        if True:
            h, h_last = scan.linear_scan(
                x=x,
                a=a,
                h0=h_prev,
                reverse=self.reverse,
                scan_type=(
                    ScanType.LINEAR_NATIVE
                    if sampling
                    else get_scan_implementation(H_rnn)
                ),
                sharding_spec=sharding_spec,
                unroll=128,
            )
        else:

            def scan_func(h, x):
                x, a = x
                h = h + x * a
                return h, h

            x = complex_lib.moveaxis(x, 1, 0)
            a = complex_lib.moveaxis(a, 1, 0)
            h_last, h = jax.lax.scan(
                scan_func,
                h_prev,
                (x, a),
            )
            h = complex_lib.moveaxis(h, 0, 1)
        h = complex_to_merged(H_rnn, h)
        h_last = complex_to_merged(H_rnn, h_last)
        return nn.Dense(d_in)(h), h_last

    def default_state(self, batch_size):
        H_rnn = self.H.rnn
        d_hidden = (
            H_rnn.d_hidden * self.feature_scale
            if H_rnn.adaptive_d
            else H_rnn.d_hidden
        )
        return jnp.zeros((batch_size, d_hidden))
