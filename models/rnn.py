import functools
from typing import Union

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.special import expit, logit

from hps import Hyperparams
from models.efficient_scan import common, complex_lib, pallas, scan

# TODO: probably should be passed in train.py
_mesh = jax.make_mesh((jax.device_count(),), ("batch",))
SHARDING_SPEC = pallas.ShardingSpec(mesh=_mesh)


def get_recurrent_block(H):
    match H.rnn_block.lower():
        case "rnn":
            return RNN
        case "rglru":
            return RGLRU
        case "lru":
            return LRU
        case _:
            raise ValueError(f"Unknown reccurent block type: {H.rnn_block}")


def get_scan_implementation(H):
    match H.scan_implementation.lower():
        case "linear_native":
            return common.ScanType.LINEAR_NATIVE
        case "linear_pallas":
            return common.ScanType.LINEAR_PALLAS
        case "associative_native":
            return common.ScanType.ASSOCIATIVE_NATIVE
        case _:
            raise ValueError(
                f"Unknown scan implementation: {H.scan_implementation}"
            )


# Want to be able to vary the scale of initialized parameters
def lecun_normal(scale):
    return nn.initializers.variance_scaling(scale, "fan_in", "truncated_normal")


def get_sinusoidal_embeddings(batch_size, seq_len, dim):
    positions = jnp.arange(seq_len)[:, None]
    div_term = jnp.exp(jnp.arange(0, dim, 2) * (-jnp.log(10000.0) / dim))

    pe = jnp.zeros((seq_len, dim))
    pe = pe.at[:, 0::2].set(jnp.sin(positions * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(positions * div_term))

    # TODO: find less hacky alternative to dividing by 10 here:
    return jnp.broadcast_to(pe, (batch_size, seq_len, dim)) / 10


def merged_to_complex(H: Hyperparams, x) -> complex_lib.RealOrComplex:
    """Returns a (complex) array from a merged array.

    A merged array is one where the first half over the last axis represents the
    real part of a complex array, while the second part represents the
    imaginary.

    Args:
      x: The merged array.

    Returns:
      A (complex) array represented by `x`.
    """
    if H.rnn_only_real:
        return x

    assert x.shape[-1] % 2 == 0
    return real_imag_complex(H, *jnp.split(x, 2, axis=-1))


def real_imag_complex(H: Hyperparams, real, imag) -> complex_lib.RealOrComplex:
    """Based on the settings, creates a (complex) number in the correct format.

    Args:
      real: The real part of the complex number.
      imag: The imaginary part of the complex number.

    Returns:
      The correct representation for a complex number. If `only_real=True`
      the function expects that `imag` is None and will directly return `real`.
      When using `bfloat16` or Pallas a `complex_lib.Complex` is returned,
      otherwise a native jax array with a complex type.
    """
    if H.rnn_only_real:
        assert imag is None
        return real

    return complex_lib.Complex(real, imag)


def complex_to_merged(
    H: Hyperparams,
    x: complex_lib.RealOrComplex,
):
    """Returns a merged array from a (complex) array.

    A merged array is one where the first half over the last axis represents the
    real part of a complex array, while the second part represents the
    imaginary.

    Args:
      x: The (complex) array.

    Returns:
      A merged array represented by `x`.
    """
    if H.rnn_only_real:
        assert not isinstance(x, complex_lib.Complex) and not jnp.iscomplexobj(
            x
        )
        return x

    else:
        return einops.rearrange([x.real, x.imag], "c ... d -> ... (c d)", c=2)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def sqrt_bound_derivative(
    x: complex_lib.RealOrComplex,
    max_gradient: float | jax.Array,
) -> jax.Array:
    """Computes a square root with a gradient clipped at `max_gradient`."""
    del max_gradient  # unused
    return complex_lib.sqrt(x)


def stable_sqrt_fwd(
    x: jax.Array,
    _: float | jax.Array,
) -> tuple[jax.Array, tuple[jax.Array]]:  # pylint: disable=g-one-element-tuple
    return complex_lib.sqrt(x), (x,)


def stable_sqrt_bwd(
    max_gradient: float | jax.Array,
    res: tuple[jax.Array],  # pylint: disable=g-one-element-tuple
    g: jax.Array,
) -> tuple[jax.Array]:  # pylint: disable=g-one-element-tuple
    (x,) = res
    if isinstance(x, complex_lib.Complex):
        magnitude = jnp.sqrt(x.real**2 + x.imag**2)
        # TODO: this part not working for jitted functions, not sure why
        rescale = jnp.min(
            jnp.ones(magnitude.shape), (1 / (magnitude * (4 * max_gradient**2)))
        )
        x_pre = x * rescale
    else:
        x_pre = jnp.maximum(x, 1 / (4 * max_gradient**2))
    return jax.vjp(complex_lib.sqrt, x_pre)[1](g)


sqrt_bound_derivative.defvjp(stable_sqrt_fwd, stable_sqrt_bwd)


class BlockDiagonalLinear(nn.Module):
    n_blocks: int
    d_input: int
    d_output: Union[int, None] = None

    def setup(self):
        assert self.d_input % self.n_blocks == 0, (
            "d_input must be divisible by n_blocks"
        )
        assert self.d_output is None or self.d_output % self.n_blocks == 0, (
            "d_output must be divisible by n_blocks"
        )

        d_in = self.d_input // self.n_blocks
        d_out = (self.d_output or self.d_input) // self.n_blocks
        self.W = self.param(
            "W", lecun_normal(1.0), (self.n_blocks, d_in, d_out)
        )
        self.b = self.param("b", lecun_normal(1.0), (self.n_blocks, d_out))

    def __call__(self, x):
        # Split x to blocks.
        x = einops.rearrange(x, "... (h i) -> ... h i", h=self.n_blocks)

        # Linear layer over each block + bias.
        y = jnp.einsum("... h i, h i j -> ... h j", x, self.W) + self.b

        # Flatten the output.
        return einops.rearrange(y, "... h j -> ... (h j)", h=self.n_blocks)


parallel_scan = jax.lax.associative_scan


# Parallel scan operations
@jax.vmap
def binary_operator_diag(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def matrix_init(key, shape, dtype=jnp.float32, normalization=1):
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization


def nu_init(key, shape, r_min, r_max, dtype=jnp.float32):
    u = jax.random.uniform(key=key, shape=shape, dtype=dtype)
    return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))


def theta_init(key, shape, max_phase, dtype=jnp.float32):
    u = jax.random.uniform(key, shape=shape, dtype=dtype)
    return jnp.log(max_phase * u)


def gamma_log_init(key, lamb):
    nu, theta = lamb
    diag_lambda = jnp.exp(-jnp.exp(nu) + 1j * jnp.exp(theta))
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))


class LRU(nn.Module):
    """
    LRU module in charge of the recurrent processing.
    Implementation following the one of Orvieto et al. 2023.
    """

    H: Hyperparams
    d_hidden: int  # hidden state dimension
    d_out: int  # input and output dimensions
    reverse: bool = False
    max_phase: float = 6.28  # max phase lambda

    def setup(self):
        self.theta_log = self.param(
            "theta_log",
            functools.partial(theta_init, max_phase=self.max_phase),
            (self.d_hidden,),
        )
        self.nu_log = self.param(
            "nu_log",
            functools.partial(
                nu_init,
                r_min=self.H.rnn_init_minval,
                r_max=self.H.rnn_init_maxval,
            ),
            (self.d_hidden,),
        )
        self.gamma_log = self.param(
            "gamma_log", gamma_log_init, (self.nu_log, self.theta_log)
        )

        # Glorot initialized Input/Output projection matrices
        self.B_re = self.param(
            "B_re",
            functools.partial(
                matrix_init, normalization=jnp.sqrt(2 * self.d_out)
            ),
            (self.d_hidden, self.d_out),
        )
        self.B_im = self.param(
            "B_im",
            functools.partial(
                matrix_init, normalization=jnp.sqrt(2 * self.d_out)
            ),
            (self.d_hidden, self.d_out),
        )
        self.C_re = self.param(
            "C_re",
            functools.partial(
                matrix_init, normalization=jnp.sqrt(self.d_hidden)
            ),
            (self.d_out, self.d_hidden),
        )
        self.C_im = self.param(
            "C_im",
            functools.partial(
                matrix_init, normalization=jnp.sqrt(self.d_hidden)
            ),
            (self.d_out, self.d_hidden),
        )
        self.D = self.param("D", matrix_init, (self.d_out,))

    def __call__(self, x, h_prev=None, pos_emb=None):
        """Forward pass of a LRU: h_t+1 = lambda * h_t + B x_t+1, y_t = Re[C h_t + D x_t]"""
        diag_lambda = jnp.exp(
            -jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log)
        )
        B_norm = (self.B_re + 1j * self.B_im) * jnp.expand_dims(
            jnp.exp(self.gamma_log), axis=-1
        )
        C = self.C_re + 1j * self.C_im

        Lambda_elements = jnp.repeat(diag_lambda[None, ...], x.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(x)
        # Compute hidden states
        _, hidden_states = parallel_scan(
            binary_operator_diag, (Lambda_elements, Bu_elements)
        )
        # Use them to compute the output of the module
        outputs = jax.vmap(lambda h, x: (C @ h).real + self.D * x)(
            hidden_states, x
        )

        return outputs, hidden_states[-1]


class RNN(nn.Module):
    H: Hyperparams
    d_hidden: int
    d_out: int
    reverse: bool = False

    @nn.compact
    def __call__(self, x, h_prev=None):
        batch_size, seq_len, _ = x.shape

        def stable_init(rng, shape):
            r_min, r_max = self.H.rnn_init_minval, self.H.rnn_init_maxval
            u = jax.random.uniform(
                key=rng, shape=shape, minval=r_min, maxval=r_max
            )
            return logit(u)

        a_logit = self.param("a_logit", stable_init, (self.d_hidden,))
        a = expit(a_logit)

        if self.H.rnn_pos_embedding:
            x = jnp.concatenate(
                [x, get_sinusoidal_embeddings(batch_size, seq_len, 16)], -1
            )
        dx = nn.Dense(self.d_out)(x)
        x = nn.Dense(self.d_hidden)(x)
        if self.H.rnn_norm_input:
            x = jnp.sqrt(1 - a**2) * x
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


class RGLRU(nn.Module):
    H: Hyperparams
    d_hidden: int
    d_out: int
    reverse: bool = False

    @nn.compact
    def __call__(self, x, h_prev=None, pos_emb=None):
        # TODO: implement BlockDiagonalLinear from RecurrentGemma
        batch_size, seq_len, d_in = x.shape
        d_hidden = self.d_hidden if self.H.rnn_only_real else self.d_hidden // 2

        # def stable_init_real(rng, shape):
        #     r_min, r_max = self.H.rnn_init_minval, self.H.rnn_init_maxval
        #     u = jax.random.uniform(
        #         key=rng, shape=shape, minval=r_min, maxval=r_max
        #     )
        #     return logit(u)

        def stable_init_real(rng, shape, eps=1e-8):
            r_min, r_max = self.H.rnn_init_minval, self.H.rnn_init_maxval
            u = jax.random.uniform(rng, shape=shape)
            a_real = 0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2 + eps)
            return jnp.log(jnp.exp(-a_real) - 1.0)

        def stable_init_imag(rng, shape):
            u = jax.random.uniform(rng, shape=shape)
            return jnp.pi * self.H.rnn_init_imag * u

        a_real_param = self.param("a_real_param", stable_init_real, (d_hidden,))
        if not self.H.rnn_only_real:
            a_imag_param = self.param(
                "a_imag_param", stable_init_imag, (d_hidden,)
            )

        if self.H.rnn_pos_embedding:
            if pos_emb is None:
                pos_emb = get_sinusoidal_embeddings(batch_size, seq_len, 16)
            x = jnp.concatenate([x, pos_emb], -1)
        x = nn.Dense(self.d_hidden)(x)

        gate_x = complex_lib.sigmoid(
            BlockDiagonalLinear(
                n_blocks=self.H.rnn_n_diag_blocks,
                d_input=self.d_hidden,
                d_output=d_hidden,
            )(x)
        )
        gate_a = complex_lib.sigmoid(
            BlockDiagonalLinear(
                n_blocks=self.H.rnn_n_diag_blocks,
                d_input=self.d_hidden,
                d_output=d_hidden,
            )(x)
        )

        log_a = -8.0 * gate_a * complex_lib.softplus(a_real_param)
        if self.H.rnn_only_real:
            a, a_squared = complex_lib.exp(log_a), complex_lib.exp(2 * log_a)
        else:
            log_a_imag = a_imag_param * gate_a
            log_a_complex = real_imag_complex(self.H, log_a, log_a_imag)
            a, a_squared = (
                complex_lib.exp(log_a_complex),
                complex_lib.exp(2 * log_a_complex),
            )
        x = merged_to_complex(self.H, x)
        h_prev = (
            self.merged_to_complex(self.H, h_prev)
            if h_prev is not None
            else None
        )

        x = gate_x * x
        # TODO: placement of norm corresponding to RGLRU
        # reconsider doing it before gating
        if self.H.rnn_norm_input:
            # x = sqrt_bound_derivative(1 - a_squared, 1000) * x
            x = complex_lib.sqrt(1 - a_squared) * x

        h, h_last = scan.linear_scan(
            x=x,
            a=a,
            h0=h_prev,
            reverse=self.reverse,
            scan_type=get_scan_implementation(self.H),
            sharding_spec=SHARDING_SPEC,
            unroll=128,
        )
        h = complex_to_merged(self.H, h)
        h_last = complex_to_merged(self.H, h_last)
        return nn.Dense(self.d_out)(h), h_last

    def default_state(self, batch_size):
        return jnp.zeros((batch_size, self.d_hidden))


class RNNBlock(nn.Module):
    H: Hyperparams
    d_out: int
    bidirectional: bool = False
    residual: bool = False
    last_scale: float = 1.0

    def setup(self):
        recurrent_block = get_recurrent_block(self.H)
        self.forward = recurrent_block(
            self.H,
            d_hidden=self.H.rnn_hidden_size,
            d_out=self.d_out,
        )
        if self.bidirectional:
            self.backward = recurrent_block(
                self.H,
                d_hidden=self.H.rnn_hidden_size,
                d_out=self.d_out,
                reverse=True,
            )
        self.last_dense = nn.Dense(self.d_out)
        self.norm = nn.LayerNorm()

    def __call__(self, x):
        identity = x
        x = self.norm(x)
        x_fwd, _ = self.forward(x)
        x = (x_fwd + self.backward(x)[0]) / 2 if self.bidirectional else x_fwd
        # x = x + identity if self.residual else x

        x = nn.gelu(x)
        x = self.last_dense(x) * self.last_scale
        x = x + identity if self.residual else x
        return x
