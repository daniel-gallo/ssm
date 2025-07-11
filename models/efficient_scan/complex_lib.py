# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Custom complex class module."""

import functools
import types
from typing import Any, Sequence, TypeVar, Union

import einops
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
from flax import struct


def _arg_is_pytree_placeholder(arg: Any) -> bool:
    """Check if argument is consistent with being a placeholder for pytree validation."""
    if arg is None:
        return True

    # For shard_map
    if (
        isinstance(arg, tuple)
        and len(arg) == 2
        and isinstance(arg[0], tuple)
        and (
            (
                len(arg[0]) == 1
                and isinstance(arg[0][0], jax.tree_util.FlattenedIndexKey)
            )
            or (
                len(arg[0]) == 2
                and isinstance(arg[0][0], jax.tree_util.SequenceKey)
            )
        )
        and isinstance(arg[1], jax.sharding.PartitionSpec)
    ):
        return True

    # For jax tracing
    if type(arg) in (object, str, set, bool):  # pylint: disable=unidiomatic-typecheck
        return True

    # For pallas and shard_map
    if isinstance(arg, (pl.BlockSpec, jax.sharding.PartitionSpec)):
        return True

    return False


def _is_pytree_placeholder(*args: Sequence[Any]) -> bool:
    """Check if arguments are consistent with being a placeholder for pytree validation."""
    return all(_arg_is_pytree_placeholder(arg) for arg in args)


@struct.dataclass
class Complex:
    """Custom Complex class.

    The minimum representation for Jax complex dtype is 64 bits (32 bits for the
    real part and 32 bits for the imaginary part). This class allows will allow
    us to work with smaller complex types as bfloat16.

    The complex class provides a subset of the operations that are possible on a
    Jax Array.
    """

    real: jax.Array
    imag: jax.Array

    def __post_init__(self) -> None:
        if not _is_pytree_placeholder(self.real, self.imag):
            assert self.real.shape == self.imag.shape
            assert self.real.dtype == self.imag.dtype

    @property
    def dtype(self) -> jnp.dtype:
        return self.real.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.real.shape

    @property
    def size(self) -> int:
        return self.real.size

    @property
    def ndim(self) -> int:
        return self.real.ndim

    def astype(self, dtype: jnp.dtype | None) -> "Complex":
        if dtype is None:
            return self
        return Complex(
            real=self.real.astype(dtype), imag=self.imag.astype(dtype)
        )

    def reshape(self, shape: Sequence[int]) -> "Complex":
        return Complex(
            real=self.real.reshape(shape), imag=self.imag.reshape(shape)
        )

    def to_numpy(self) -> jax.Array:
        if self.dtype in (jnp.float16, jnp.bfloat16):
            raise ValueError("There does not exist a jnp.complex32 dtype.")
        return self.real + 1j * self.imag

    def _sanity_check(
        self,
        x: Union[jax.Array, "Complex"],
    ) -> None:
        """Check if the arg is not native complex and has the same dtype as this instance.

        This is required to make sure we are converting everything that is
        jax.complex to the complex wrapper.
        Args:
          x: the operand to validate
        """

        if jnp.iscomplexobj(x):
            raise ValueError("Expected argument to not be of type jax.complex")

        if not isinstance(x, (float, int)) and self.dtype != x.dtype:
            raise ValueError(
                f"Both operands should have the same type! found {self.dtype} and"
                f" {x.dtype}"
            )

    def __matmul__(self, x: Union[jax.Array, "Complex"]) -> "Complex":
        """Performs the matrix multiplication operation."""
        self._sanity_check(x)

        if isinstance(x, (jax.Array, np.ndarray)) and not jnp.iscomplexobj(x):
            return Complex(real=self.real @ x, imag=self.imag @ x)

        tmp = (self.real + self.imag) @ (x.real + x.imag)
        tmp_real = self.real @ x.real
        tmp_imag = self.imag @ x.imag

        real = tmp_real - tmp_imag
        imag = tmp - tmp_real - tmp_imag
        return Complex(real=real, imag=imag)

    def __mul__(self, x: Union[jax.Array, "Complex"]) -> "Complex":
        """Performs the multiplication operation."""
        self._sanity_check(x)

        if isinstance(x, (jax.Array, np.ndarray)) and not jnp.iscomplexobj(x):
            return Complex(real=self.real * x, imag=self.imag * x)

        real = self.real * x.real - self.imag * x.imag
        imag = self.real * x.imag + self.imag * x.real
        return Complex(real=real, imag=imag)

    __rmul__ = __mul__

    def __truediv__(self, x: Union[jax.Array, "Complex"]) -> "Complex":
        self._sanity_check(x)

        if isinstance(x, (jax.Array, np.ndarray)) and not jnp.iscomplexobj(x):
            return Complex(real=self.real / x, imag=self.imag / x)

        denominator = x.real**2 + x.imag**2
        real = (self.real * x.real + self.imag * x.imag) / denominator
        imag = (self.imag * x.real - self.real * x.imag) / denominator
        return Complex(real=real, imag=imag)

    def __neg__(self) -> "Complex":
        return Complex(real=-self.real, imag=-self.imag)

    def __sub__(self, x: Union[jax.Array, "Complex"]) -> "Complex":
        self._sanity_check(x)
        return Complex(real=self.real - x.real, imag=self.imag - x.imag)

    def __rsub__(self, x: jax.Array) -> "Complex":
        self._sanity_check(x)
        return Complex(real=x - self.real, imag=-self.imag)

    def __add__(self, x: Union[jax.Array, "Complex"]) -> "Complex":
        self._sanity_check(x)
        return Complex(real=self.real + x.real, imag=self.imag + x.imag)

    __radd__ = __add__

    def __getitem__(self, key: Any) -> "Complex":
        return Complex(real=self.real[key], imag=self.imag[key])

    def __setitem__(self, key: Any, value: "Complex"):
        if not isinstance(value, Complex):
            raise NotImplementedError()
        self.real[key] = value.real
        self.imag[key] = value.imag

    def __eq__(
        self, other: Any
    ) -> jax.Array:  # pytype: disable=signature-mismatch
        if not isinstance(other, (jax.Array, np.ndarray, Complex)):
            raise ValueError(
                "Expected argument to be of type jax.Array, np.ndarrayor Complex."
            )

        all_equal_real = jnp.equal(self.real, other.real)
        all_equal_imag = jnp.equal(self.imag, other.imag)
        return jnp.logical_and(all_equal_real, all_equal_imag)

    def __iter__(self):
        for a, b in zip(self.real, self.imag):
            yield Complex(real=a, imag=b)


@struct.dataclass
class Quaternion:
    """Custom Complex class.

    The minimum representation for Jax complex dtype is 64 bits (32 bits for the
    real part and 32 bits for the imaginary part). This class allows will allow
    us to work with smaller complex types as bfloat16.

    The complex class provides a subset of the operations that are possible on a
    Jax Array.
    """

    real: jax.Array
    i: jax.Array
    j: jax.Array
    k: jax.Array

    def __post_init__(self) -> None:
        if any(
            type(x).__name__ == "Leaf"
            for x in [self.real, self.i, self.j, self.k]
        ):
            print(
                "[Leaf detected in Quaternion]",
                self.real,
                self.i,
                self.j,
                self.k,
            )
            import traceback

            traceback.print_stack()
        if not _is_pytree_placeholder(self.real, self.i, self.j, self.k):
            assert (
                self.real.shape == self.i.shape == self.j.shape == self.k.shape
            )
            assert (
                self.real.dtype == self.i.dtype == self.j.dtype == self.k.dtype
            )

    @property
    def dtype(self) -> jnp.dtype:
        return self.real.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.real.shape

    @property
    def size(self) -> int:
        return self.real.size

    @property
    def ndim(self) -> int:
        return self.real.ndim

    def astype(self, dtype: jnp.dtype | None) -> "Quaternion":
        if dtype is None:
            return self
        return Quaternion(
            real=self.real.astype(dtype),
            i=self.i.astype(dtype),
            j=self.j.astype(dtype),
            k=self.k.astype(dtype),
        )

    def reshape(self, shape: Sequence[int]) -> "Quaternion":
        return Quaternion(
            real=self.real.reshape(shape),
            i=self.i.reshape(shape),
            j=self.j.reshape(shape),
            k=self.k.reshape(shape),
        )

    def to_numpy(self) -> jax.Array:
        # if self.dtype in (jnp.float16, jnp.bfloat16):
        #     raise ValueError("There does not exist a jnp.complex32 dtype.")
        # return self.real + 1j * self.imag
        raise ValueError("There does not exist a jnp.quaternion dtype.")

    def _sanity_check(
        self,
        x: Union[jax.Array, "Complex", "Quaternion"],
    ) -> None:
        """Check if the arg is not native complex and has the same dtype as this instance.

        This is required to make sure we are converting everything that is
        jax.complex to the complex wrapper.
        Args:
          x: the operand to validate
        """

        if jnp.iscomplexobj(x):
            raise ValueError("Expected argument to not be of type jax.complex")

        if not isinstance(x, (float, int)) and self.dtype != x.dtype:
            raise ValueError(
                f"Both operands should have the same type! found {self.dtype} and"
                f" {x.dtype}"
            )

    def __matmul__(
        self, x: Union[jax.Array, "Complex", "Quaternion"]
    ) -> "Quaternion":
        """Performs the matrix multiplication operation."""
        self._sanity_check(x)

        if isinstance(x, Complex):
            raise ValueError(
                "Expected argument to be of scalar type or Quaternion"
            )

        if not isinstance(x, Quaternion):
            return Quaternion(
                real=self.real @ x,
                i=self.i @ x,
                j=self.j @ x,
                k=self.k @ x,
            )

        real = self.real @ x.real - self.i @ x.i - self.j @ x.j - self.k @ x.k

        temp_i = self.j @ x.k - self.k @ x.j
        temp_j = self.k @ x.i - self.i @ x.k
        temp_k = self.i @ x.j - self.j @ x.i
        i = self.real @ x.i + self.i @ x.real + temp_i
        j = self.real @ x.j + self.j @ x.real + temp_j
        k = self.real @ x.k + self.k @ x.real + temp_k
        return Quaternion(real=real, i=i, j=j, k=k)

    def __mul__(
        self, x: Union[jax.Array, "Complex", "Quaternion"]
    ) -> "Quaternion":
        """Performs the multiplication operation."""
        self._sanity_check(x)

        if not isinstance(x, Quaternion):
            return Quaternion(
                real=self.real * x, i=self.i * x, j=self.j * x, k=self.k * x
            )

        real = self.real * x.real - self.i * x.i - self.j * x.j - self.k * x.k

        temp_i = self.j * x.k - self.k * x.j
        temp_j = self.k * x.i - self.i * x.k
        temp_k = self.i * x.j - self.j * x.i
        i = self.real * x.i + self.i * x.real + temp_i
        j = self.real * x.j + self.j * x.real + temp_j
        k = self.real * x.k + self.k * x.real + temp_k
        return Quaternion(real=real, i=i, j=j, k=k)

    __rmul__ = __mul__

    def __truediv__(self, x: Union[jax.Array, "Complex"]) -> "Quaternion":
        self._sanity_check(x)

        if not isinstance(x, Quaternion):
            return Quaternion(
                real=self.real / x, i=self.i / x, j=self.j / x, k=self.k / x
            )

        denominator = x.real**2 + x.i**2 + x.j**2 + x.k**2
        real = (
            self.real * x.real + self.i * x.i + self.j * x.j + self.k * x.k
        ) / denominator

        temp_i = self.j * x.k - self.k * x.j
        temp_j = self.k * x.i - self.i * x.k
        temp_k = self.i * x.j - self.j * x.i
        i = (self.i * x.real - self.real * x.i - temp_i) / denominator
        j = (self.j * x.real - self.real * x.j - temp_j) / denominator
        k = (self.k * x.real - self.real * x.k - temp_k) / denominator
        return Quaternion(real=real, i=i, j=j, k=k)

    def __neg__(self) -> "Quaternion":
        return Quaternion(real=-self.real, i=-self.i, j=-self.j, k=-self.k)

    def __sub__(
        self, x: Union[jax.Array, "Complex", "Quaternion"]
    ) -> "Quaternion":
        self._sanity_check(x)
        return Quaternion(
            real=self.real - x.real,
            i=self.i - x.i,
            j=self.j - x.j,
            k=self.k - x.k,
        )

    def __rsub__(self, x: jax.Array) -> "Quaternion":
        self._sanity_check(x)
        return Quaternion(real=x - self.real, i=-self.i, j=-self.j, k=-self.k)

    def __add__(self, x: Union[jax.Array, "Quaternion"]) -> "Quaternion":
        self._sanity_check(x)
        return Quaternion(
            real=self.real + x.real,
            i=self.i + x.i,
            j=self.j + x.j,
            k=self.k + x.k,
        )

    __radd__ = __add__

    def __getitem__(self, key: Any) -> "Quaternion":
        return Quaternion(
            real=self.real[key], i=self.i[key], j=self.j[key], k=self.k[key]
        )

    def __setitem__(self, key: Any, value: "Quaternion"):
        # TODO: Is it even used? I don't think the original version works anyway.
        if not isinstance(value, Quaternion):
            raise NotImplementedError()
        # self.real = self.real.at[key].set(value.real)
        # self.i = self.i.at[key].set(value.i)
        # self.j = self.j.at[key].set(value.j)
        # self.k = self.k.at[key].set(value.k)
        self.real[key] = value.real
        self.i[key] = value.i
        self.j[key] = value.j
        self.k[key] = value.k

    def __eq__(
        self, other: Any
    ) -> jax.Array:  # pytype: disable=signature-mismatch
        if not isinstance(other, (jax.Array, np.ndarray, Quaternion)):
            raise ValueError(
                "Expected argument to be of type jax.Array, np.ndarrayor Complex."
            )

        all_equal_real = jnp.equal(self.real, other.real)
        all_equal_i = jnp.equal(self.i, other.i)
        all_equal_j = jnp.equal(self.j, other.j)
        all_equal_k = jnp.equal(self.k, other.k)
        return all_equal_real & all_equal_i & all_equal_j & all_equal_k

    def __iter__(self):
        for real, i, j, k in zip(self.real, self.i, self.j, self.k):
            yield Quaternion(real=real, i=i, j=j, k=k)


RealOrComplex = TypeVar("RealOrComplex", jax.Array, Complex, Quaternion)


def _treat_method(
    method_name: str,
    module: types.ModuleType,
    x: list[RealOrComplex] | RealOrComplex,
    *args: Any,
    **kwargs: Any,
) -> RealOrComplex:
    """Calls the appropriate method depending on the parameters type."""
    method = getattr(module, method_name)

    if (
        isinstance(x, Complex)
        or (
            isinstance(x, Sequence) and any(isinstance(xi, Complex) for xi in x)
        )
        or any(isinstance(ai, Complex) for ai in args)
    ):
        if isinstance(x, list):
            x_real = [e.real for e in x]
            x_imag = [e.imag for e in x]
        else:
            x_real = x.real
            x_imag = x.imag

        if args and any(isinstance(ai, Complex) for ai in args):
            # For the moment, we assume all the args will be of Complex type so we
            # need to split the real and imaginary part
            args_real = [e.real for e in args]
            args_imag = [e.imag for e in args]
            real_new = method(x_real, *args_real, **kwargs)
            imag_new = method(x_imag, *args_imag, **kwargs)
            return Complex(real=real_new, imag=imag_new)

        real_new = method(x_real, *args, **kwargs)
        imag_new = method(x_imag, *args, **kwargs)
        return Complex(real=real_new, imag=imag_new)
    elif (
        isinstance(x, Quaternion)
        or (
            isinstance(x, Sequence)
            and any(isinstance(xi, Quaternion) for xi in x)
        )
        or any(isinstance(ai, Quaternion) for ai in args)
    ):
        if isinstance(x, list):
            x_real = [e.real for e in x]
            x_i = [e.i for e in x]
            x_j = [e.j for e in x]
            x_k = [e.k for e in x]
        else:
            x_real = x.real
            x_i = x.i
            x_j = x.j
            x_k = x.k

        if args and any(isinstance(ai, Quaternion) for ai in args):
            # For the moment, we assume all the args will be of Complex type so we
            # need to split the real and imaginary part
            args_real = [e.real for e in args]
            args_i = [e.i for e in args]
            args_j = [e.j for e in args]
            args_k = [e.k for e in args]

            method = functools.partial(method, **kwargs)
            real_new = method(x_real, *args_real)
            i_new = method(x_i, *args_i)
            j_new = method(x_j, *args_j)
            k_new = method(x_k, *args_k)
            return Quaternion(real=real_new, i=i_new, j=j_new, k=k_new)

        method = functools.partial(method, **kwargs)
        real_new = method(x_real, *args)
        i_new = method(x_i, *args)
        j_new = method(x_j, *args)
        k_new = method(x_k, *args)
        return Quaternion(real=real_new, i=i_new, j=j_new, k=k_new)
    else:
        return method(x, *args, **kwargs)


def to_custom_complex(x: RealOrComplex) -> Complex:
    return Complex(x.real, x.imag)


# JNP Methods
broadcast_to = functools.partial(_treat_method, "broadcast_to", jnp)
concatenate = functools.partial(_treat_method, "concatenate", jnp)
split = functools.partial(_treat_method, "split", jnp)
expand_dims = functools.partial(_treat_method, "expand_dims", jnp)
flip = functools.partial(_treat_method, "flip", jnp)
reshape = functools.partial(_treat_method, "reshape", jnp)
moveaxis = functools.partial(_treat_method, "moveaxis", jnp)
squeeze = functools.partial(_treat_method, "squeeze", jnp)
stack = functools.partial(_treat_method, "stack", jnp)
tile = functools.partial(_treat_method, "tile", jnp)
transpose = functools.partial(_treat_method, "transpose", jnp)
zeros_like = functools.partial(_treat_method, "zeros_like", jnp)

# Jax Lax
add = functools.partial(_treat_method, "add", jax.lax)
pad = functools.partial(_treat_method, "pad", jax.lax)
slice_in_dim = functools.partial(_treat_method, "slice_in_dim", jax.lax)

# EINOPS Methods
rearrange = functools.partial(_treat_method, "rearrange", einops)
repeat = functools.partial(_treat_method, "repeat", einops)


def sigmoid(x: RealOrComplex) -> RealOrComplex:
    if isinstance(x, Complex) or isinstance(x, Quaternion):
        return 1 / (1 + exp(-x))

    return jax.nn.sigmoid(x)


def softplus(x: RealOrComplex) -> RealOrComplex:
    if isinstance(x, Complex) or isinstance(x, Quaternion):
        return log(1 + exp(x))
    else:
        return jax.nn.softplus(x)


# Special methods
def ones_like(x: RealOrComplex) -> RealOrComplex:
    if isinstance(x, Complex):
        return Complex(jnp.ones_like(x.real), jnp.zeros_like(x.imag))
    elif isinstance(x, Quaternion):
        return Quaternion(
            jnp.ones_like(x.real),
            jnp.zeros_like(x.i),
            jnp.zeros_like(x.j),
            jnp.zeros_like(x.k),
        )
    else:
        return jnp.ones_like(x)


def exp(x: RealOrComplex) -> RealOrComplex:
    if isinstance(x, Complex):
        r = jnp.exp(x.real)
        theta = x.imag
        return Complex(r * jnp.cos(theta), r * jnp.sin(theta))
    elif isinstance(x, Quaternion):
        r = jnp.exp(x.real)
        theta = jnp.sqrt(x.i**2 + x.j**2 + x.k**2)
        temp_i = x.i / (theta + 1e-6)
        temp_j = x.j / (theta + 1e-6)
        temp_k = x.k / (theta + 1e-6)
        return Quaternion(
            real=r * jnp.cos(theta),
            i=r * temp_i * jnp.sin(theta),
            j=r * temp_j * jnp.sin(theta),
            k=r * temp_k * jnp.sin(theta),
        )
    else:
        return jnp.exp(x)


def log(x: RealOrComplex) -> RealOrComplex:
    if isinstance(x, Complex):
        r_squared = x.real**2 + x.imag**2
        theta = jnp.arctan2(x.imag, x.real)
        return Complex(jnp.log(r_squared) / 2, theta)
    elif isinstance(x, Quaternion):
        temp = x.i**2 + x.j**2 + x.k**2
        magnitude = jnp.sqrt(x.real**2 + temp)
        theta = jnp.sqrt(temp)
        unit_i = x.i / (theta + 1e-6)
        unit_j = x.j / (theta + 1e-6)
        unit_k = x.k / (theta + 1e-6)
        return Quaternion(
            real=jnp.log(magnitude),
            i=unit_i * jnp.sin(x.real / magnitude),
            j=unit_j * jnp.sin(x.real / magnitude),
            k=unit_k * jnp.sin(x.real / magnitude),
        )
    else:
        return jnp.log(x)


def conjugate(x: RealOrComplex) -> RealOrComplex:
    if isinstance(x, Complex):
        return Complex(x.real, -x.imag)
    elif isinstance(x, Quaternion):
        return Quaternion(x.real, -x.i, -x.j, -x.k)
    else:
        return jnp.conjugate(x)


def abs_squared(x: RealOrComplex) -> jax.Array:
    if isinstance(x, Quaternion):
        return x.real**2 + x.i**2 + x.j**2 + x.k**2
    return x.real**2 + x.imag**2


def sqrt(x: RealOrComplex) -> RealOrComplex:
    if isinstance(x, Complex):
        mag = jnp.sqrt(x.real**2 + x.imag**2)
        return Complex(
            jnp.sqrt(0.5 * (mag + x.real)),
            jnp.sign(x.imag) * jnp.sqrt(0.5 * (mag - x.real)),
        )
    elif isinstance(x, Quaternion):
        mag = jnp.sqrt(x.real**2 + x.i**2 + x.j**2 + x.k**2)
        temp = jnp.sqrt((mag + x.real) / 2)
        return Quaternion(
            real=temp,
            i=0.5 * x.i / (temp + 1e-6),
            j=0.5 * x.j / (temp + 1e-6),
            k=0.5 * x.k / (temp + 1e-6),
        )
    else:
        return jnp.sqrt(x)


def einsum(sum_str: str, *args: jax.Array | Complex) -> jax.Array | Complex:
    """Computes the equivalent of jnp.einsum."""
    # TODO: Add Quaternion support
    num_custom_complex_args = sum(isinstance(arg, Complex) for arg in args)
    num_np_complex_args = sum(jnp.iscomplexobj(arg) for arg in args)

    if num_custom_complex_args == 0:
        return jnp.einsum(sum_str, *args)

    elif num_custom_complex_args == 1:
        r_args = [arg.real if isinstance(arg, Complex) else arg for arg in args]
        i_args = [arg.imag if isinstance(arg, Complex) else arg for arg in args]
        real = jnp.einsum(sum_str, *r_args)
        imag = jnp.einsum(sum_str, *i_args)
        if num_np_complex_args == 0:
            return Complex(real, imag)
        else:
            return Complex(real.real - imag.imag, real.imag + imag.imag)

    elif num_custom_complex_args == 2 and len(args) == 2:
        r0, i0 = args[0].real, args[0].imag
        r1, i1 = args[1].real, args[1].imag
        rr = jnp.einsum(sum_str, r0, r1)
        ri = jnp.einsum(sum_str, r0, i1)
        ir = jnp.einsum(sum_str, i0, r1)
        ii = jnp.einsum(sum_str, i0, i1)
        return Complex(rr - ii, ri + ir)

    else:
        raise NotImplementedError()
