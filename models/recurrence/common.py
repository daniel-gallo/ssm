import functools
from typing import Union

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

from models.efficient_scan import common, complex_lib
from models.recurrence.hps import RNNHyperparams


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


def get_scan_implementation(H: RNNHyperparams):
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


# Custom complex-values support
def merged_to_complex(H: RNNHyperparams, x) -> complex_lib.RealOrComplex:
    """Returns a (complex) array from a merged array.

    A merged array is one where the first half over the last axis represents the
    real part of a complex array, while the second part represents the
    imaginary.

    Args:
      x: The merged array.

    Returns:
      A (complex) array represented by `x`.
    """
    match H.dtype_hidden:
        case "real":
            return x
        case "complex":
            assert x.shape[-1] % 2 == 0
            return real_imag_complex(H, *jnp.split(x, 2, axis=-1))
        case "quaternion":
            assert x.shape[-1] % 4 == 0
            x = einops.rearrange(x, "... (d i) -> i ... d", i=4)
            real, imag = x[0], x[1:]
            return real_imag_complex(H, real, imag)
        case _:
            raise ValueError(f"Unknown dtype_hidden: {H.dtype_hidden}")


def real_imag_complex(
    H: RNNHyperparams, real, imag
) -> complex_lib.RealOrComplex:
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
    match H.dtype_hidden:
        case "real":
            assert imag is None
            return real
        case "complex":
            return complex_lib.Complex(real, imag)
        case "quaternion":
            assert imag.shape[0] == 3
            i, j, k = imag
            return complex_lib.Quaternion(real, i, j, k)
        case _:
            raise ValueError(f"Unknown dtype_hidden: {H.dtype_hidden}")


def complex_to_merged(
    H: RNNHyperparams,
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
    match H.dtype_hidden:
        case "real":
            assert not isinstance(
                x, complex_lib.Complex
            ) and not jnp.iscomplexobj(x)
            return x
        case "complex":
            return einops.rearrange(
                [x.real, x.imag], "i ... d -> ... (d i)", i=2
            )
        case "quaternion":
            x = jnp.stack([x.real, x.i, x.j, x.k], axis=0)
            return einops.rearrange(x, "i ... d-> ... (d i)", i=4)
        case _:
            raise ValueError(f"Unknown dtype_hidden: {H.dtype_hidden}")


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def sqrt_bound_derivative(
    x: complex_lib.RealOrComplex,
    max_gradient: float | jax.Array,
) -> jax.Array:
    """Computes a square root with a gradient clipped at `max_gradient`."""
    del max_gradient  # unused
    return jnp.sqrt(x)


def stable_sqrt_fwd(
    x: jax.Array,
    _: float | jax.Array,
) -> tuple[jax.Array, tuple[jax.Array]]:  # pylint: disable=g-one-element-tuple
    return jnp.sqrt(x), (x,)


def stable_sqrt_bwd(
    max_gradient: float | jax.Array,
    res: tuple[jax.Array],  # pylint: disable=g-one-element-tuple
    g: jax.Array,
) -> tuple[jax.Array]:  # pylint: disable=g-one-element-tuple
    (x,) = res
    x_pre = jnp.maximum(x, 1 / (4 * max_gradient**2))
    return jax.vjp(jnp.sqrt, x_pre)[1](g)


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
