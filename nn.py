from timeit import timeit

import jax.numpy as jnp
import numpy as np
from jax import nn
from jax import grad

import jax


@jax.custom_vjp
def gelu(x):
    # Based on
    # https://github.com/jax-ml/jax/blob/aed329/jax/_src/nn/functions.py#L455-L488
    sqrt_2_over_pi = np.sqrt(2 / np.pi).astype(x.dtype)
    cdf = 0.5 * (1.0 + jnp.tanh(sqrt_2_over_pi * (x + 0.044715 * (x ** 3))))
    return x * cdf

def gelu_fwd(x):
    # Primal computation
    sqrt_2_over_pi = np.sqrt(2 / np.pi).astype(x.dtype)
    c = 0.044715
    t = jnp.tanh(sqrt_2_over_pi * (x + c * (x ** 3)))
    one_plus_t = 1.0 + t
    cdf = 0.5 * one_plus_t
    primal_out = x * cdf

    derivative = 0.5 * (
        one_plus_t + x * (1 - t ** 2) * sqrt_2_over_pi * (1 + 3 * c * x ** 2)
    )
    return primal_out, derivative

def gelu_bwd(derivative, g):
    return derivative * g,

gelu.defvjp(gelu_fwd, gelu_bwd)
