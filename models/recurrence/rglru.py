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

        # INITIALISERS
        def stable_init_real(rng, shape, eps=1e-8):
            r_min, r_max = H_rnn.init_minval_real, H_rnn.init_maxval_real
            u = jax.random.uniform(rng, shape=shape)
            match H_rnn.param_real:
                case "softplus":
                    a_real = jnp.sqrt(u * (r_max**2 - r_min**2) + r_min**2 + eps)
                case "exponential":
                    constant = -0.5
                    a_real = constant * jnp.log(u * (r_max**2 - r_min**2) + r_min**2 + eps)
                case _:
                    raise ValueError(f"Unknown param_real: {H_rnn.param_real}")
            return jnp.log(jnp.exp(a_real) - 1.0)

        def stable_init_imag(rng, shape):
            u = 2 * jax.random.uniform(rng, shape=shape) - 1
            scale = (
                H_rnn.init_maxval_imag * self.temporal_scale
                if H_rnn.adaptive_phase
                else H_rnn.init_maxval_imag
            )
            match H_rnn.param_imag:
                case "linear":
                    theta = scale * u
                case "tanh":
                    theta = jnp.arctanh(scale * u)
                case "exponential":
                    theta = jnp.log(scale * u)
                case _:
                    raise ValueError(f"Unknown param_imag: {H_rnn.param_imag}")
            return theta

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
            case "log_sigmoid":
                gate_a = complex_lib.sigmoid(
                    BlockDiagonalLinear(
                        n_blocks=H_rnn.n_diag_blocks,
                        d_input=d_hidden,
                        d_output=d_inner,
                    )(x)
                )
                log_gate_a_real = gate_a
                gate_a_real = jnp.ones_like(gate_a)
            case "mlp":
                gate_a = BlockDiagonalLinear(
                    n_blocks=H_rnn.n_diag_blocks,
                    d_input=d_hidden,
                    d_output=d_inner,
                )(x)
                log_gate_a_real = jnp.ones_like(gate_a)
                gate_a_real = gate_a
            case "tanh":
                gate_a = jnp.tanh(
                        BlockDiagonalLinear(
                        n_blocks=H_rnn.n_diag_blocks,
                        d_input=d_hidden,
                        d_output=d_inner,
                    )(x)
                )
                log_gate_a_real = jnp.ones_like(gate_a)
                gate_a_real = gate_a
            case "none":
                gate_a = jnp.ones((batch_size, seq_len, d_inner))
                log_gate_a_real = gate_a
                gate_a_real = gate_a
            case _:
                raise ValueError(f"Unknown gate_a_real: {H_rnn.gate_a_real}")

        if H_rnn.dtype_hidden != "real":
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

        match H_rnn.param_real:
            case "softplus":
                magn_a = jnp.power(complex_lib.softplus(a_real_param), log_gate_a_real)
                magn_a = magn_a * gate_a_real
            case "exponential":
                log_a = (
                    H_rnn.log_a_scale
                    * log_gate_a_real
                    * complex_lib.softplus(a_real_param)
                )
                magn_a = complex_lib.exp(log_a) * gate_a_real
            case _:
                raise ValueError(f"Unknown parameterization for real part: {H_rnn.param_real}")

        if H_rnn.dtype_hidden != "real":
            match H_rnn.param_imag:
                case "linear":
                    log_a_imag = a_imag_param * gate_a_imag * jnp.pi
                case "tanh":
                    log_a_imag = jnp.tanh(a_imag_param) * gate_a_imag * jnp.pi
                case "exponential":
                    log_a_imag = jnp.exp(a_imag_param) * gate_a_imag * jnp.pi
                case _:
                    raise ValueError(f"Unknown parameterization for imaginary part: {H_rnn.param_imag}")

        match H_rnn.dtype_hidden:
            case "real":
                a = magn_a
            case "complex":
                log_a_complex = real_imag_complex(H_rnn, jnp.zeros_like(magn_a), log_a_imag)
                a = magn_a * complex_lib.exp(log_a_complex)
            case "quaternion":
                alpha, beta, gamma = (
                    log_a_imag[..., 0],
                    log_a_imag[..., 1],
                    log_a_imag[..., 2],
                )
                a_sin, b_sin, c_sin = jnp.sin(alpha / 2), jnp.sin(beta / 2), jnp.sin(gamma / 2)
                a_cos, b_cos, c_cos = jnp.cos(alpha / 2), jnp.cos(beta / 2), jnp.cos(gamma / 2)
                imag_w = a_cos * b_cos * c_cos + a_sin * b_sin * c_sin
                imag_x = a_sin * b_cos * c_cos - a_cos * b_sin * c_sin
                imag_y = a_cos * b_sin * c_cos + a_sin * b_cos * c_sin
                imag_z = a_cos * b_cos * c_sin - a_sin * b_sin * c_cos
                a_imag = jnp.stack([imag_x, imag_y, imag_z], axis=0)
                a = real_imag_complex(H_rnn, imag_w, a_imag) * magn_a
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
                a_squared = jnp.square(magn_a)
                norm_factor = sqrt_bound_derivative(
                    jnp.maximum(1 - a_squared, jnp.full_like(a_squared, 1e-6)),
                    200
                )
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
