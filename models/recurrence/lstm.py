from typing import Callable, Any, Tuple, Optional
from functools import partial

import numpy as np

import jax.numpy as jnp
from jax import random
from jax import vmap

import flax.linen as nn
from flax.linen.recurrent import RNNCellBase, DenseParams
from flax.linen.linear import Dense, default_kernel_init
from flax.linen import initializers
from flax.linen import sigmoid, tanh
from flax.linen.dtypes import promote_dtype
from flax.linen.module import Module, compact, nowrap
from flax.typing import (
    Array,
    PRNGKey,
    Dtype,
    InOutScanAxis,
    Initializer,
    PrecisionLike,
)

from models.recurrence.hps import RNNHyperparams


class LSTMCellScalar(RNNCellBase):
    features: int
    gate_fn: Callable[..., Any] = sigmoid
    activation_fn: Callable[..., Any] = tanh
    kernel_init: Initializer = default_kernel_init
    recurrent_kernel_init: Initializer = initializers.ones_init()
    bias_init: Initializer = initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    carry_init: Initializer = initializers.zeros_init()

    @compact
    def __call__(self, carry, x_ifgo):
        c, h = carry
        hidden_features = h.shape[-1]

        def _elemwise_split(inputs, params):
            b, c = inputs.shape
            kernel, bias = params
            four, c_ = kernel.shape
            assert four == 4
            assert c_ == c
            four, c_ = bias.shape
            assert four == 4
            assert c_ == c
            return kernel[:, jnp.newaxis, :] * inputs + bias[:, jnp.newaxis, :]

        # Create params with the same names/shapes as `LSTMCell` for
        # compatibility.
        elemwise_params_h = (
            self.param(
                'kernel_h', self.recurrent_kernel_init,
                (4, hidden_features,), self.param_dtype
            ),
            self.param(
                'bias_h', self.bias_init, (4, hidden_features),
                self.param_dtype)
        )
        h_ifgo = _elemwise_split(h, elemwise_params_h)
        i, f, g, o = h_ifgo + jnp.moveaxis(
            jnp.reshape(x_ifgo, x_ifgo.shape[:-1] + (-1, 4)),
            -1, 0
        )

        i = self.gate_fn(i)
        f = self.gate_fn(f)
        g = self.activation_fn(g)
        o = self.gate_fn(o)

        new_c = f * c + i * g
        new_h = o * self.activation_fn(new_c)
        return (new_c, new_h), new_h

    @nowrap
    def initialize_carry(
        self, rng: PRNGKey, input_shape: tuple[int, ...]
    ) -> tuple[Array, Array]:
        """Initialize the RNN cell carry.

        Args:
          rng: random number generator passed to the init_fn.
          input_shape: a tuple providing the shape of the input to the cell.

        Returns:
          An initialized carry for the given RNN cell.
        """
        batch_dims = input_shape[:-1]
        key1, key2 = random.split(rng)
        mem_shape = batch_dims + (self.features,)
        c = self.carry_init(key1, mem_shape, self.param_dtype)
        h = self.carry_init(key2, mem_shape, self.param_dtype)
        return c, h

    @property
    def num_feature_axes(self) -> int:
        return 1


class LSTMScalar(nn.Module):
    H: RNNHyperparams
    d_hidden: int
    d_out: int
    reverse: bool = False

    def setup(self):
        H = self.H
        self.in_dense = nn.Dense(features=4 * self.d_hidden)
        self.cell = LSTMCellScalar(features=self.d_hidden)
        self.rnn = nn.RNN(self.cell, return_carry=True)
        self.out_dense = nn.Dense(features=self.d_out)

    def __call__(self, x, h_prev=None):
        b, l, c = x.shape
        if h_prev is None:
            init_carry = self.cell.initialize_carry(
                random.PRNGKey(0), (b, self.H.data_num_cats)
            )
        else:
            init_carry = h_prev
        ifgo = self.in_dense(x)
        carry, out = self.rnn(ifgo, initial_carry=init_carry)
        return self.out_dense(out), carry
