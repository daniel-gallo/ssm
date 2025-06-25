import functools

import flax.linen as nn
import jax
import jax.numpy as jnp

from models.recurrence.hps import RNNHyperparams
from hps import Hyperparams

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


class OldLRU(nn.Module):
    """
    LRU module in charge of the recurrent processing.
    Implementation following the one of Orvieto et al. 2023.
    """

    H: Hyperparams
    # d_hidden: int  # hidden state dimension
    # d_out: int  # input and output dimensions
    # reverse: bool = False
    # max_phase: float = 6.28  # max phase lambda
    reverse: bool = False
    temporal_scale: int = 1  # rename
    feature_scale: int = 1  # rename

    def setup(self):
        H_rnn = self.H.rnn
        self.d_out = self.H.base_dim * self.feature_scale
        d_hidden = (
            H_rnn.d_hidden * self.feature_scale
            if H_rnn.adaptive_d
            else H_rnn.d_hidden
        )
        self.d_hidden = d_hidden if H_rnn.only_real else d_hidden // 2
        self.input_dense = nn.Dense(self.d_out)
        self.theta_log = self.param(
            "theta_log",
            functools.partial(theta_init, max_phase=H_rnn.init_maxval_imag),
            (self.d_hidden,),
        )
        self.nu_log = self.param(
            "nu_log",
            functools.partial(
                nu_init,
                r_min=H_rnn.init_minval_real,
                r_max=H_rnn.init_maxval_real,
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

    def __call__(self, x, h_prev=None, pos_emb=None, sampling=False):
        def __inner(x):
            """Forward pass of a LRU: h_t+1 = lambda * h_t + B x_t+1, y_t = Re[C h_t + D x_t]"""
            diag_lambda = jnp.exp(
                -jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log)
            )
            B_norm = (self.B_re + 1j * self.B_im) * jnp.expand_dims(
                jnp.exp(self.gamma_log), axis=-1
            )
            C = self.C_re + 1j * self.C_im

            x = self.input_dense(x)
            Lambda_elements = jnp.repeat(
                diag_lambda[None, ...], x.shape[0], axis=0
            )
            Bu_elements = jax.vmap(lambda v: B_norm @ v)(x)

            # Compute hidden states
            _, hidden_states = parallel_scan(
                binary_operator_diag,
                (Lambda_elements, Bu_elements),
                reverse=self.reverse,
            )

            # Use them to compute the output of the module
            outputs = jax.vmap(lambda h, x: (C @ h).real + self.D * x)(
                hidden_states, x
            )

            return outputs, hidden_states[-1]

        outputs, hidden_states = jax.vmap(__inner)(x)
        return outputs, hidden_states
