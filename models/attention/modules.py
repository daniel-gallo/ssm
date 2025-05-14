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
"""Griffin and Hawk"s model components."""

import functools
from typing import Literal, NamedTuple, overload

import einops
import jax
import jax.numpy as jnp
from flax import linen as nn

from models.attention import array_typing as at

_MIN_LOGITS_VALUE = -2.3819763e38  # Set to a large negative number.
_MAX_WAVELENGTH = 10_000
_vmap_cache_roll = jax.vmap(functools.partial(jnp.roll, axis=0))


@at.typed
class AttentionBlockCache(NamedTuple):
    """The cache for an attention block."""

    keys: at.CachedKeys
    values: at.CachedValues
    num_tokens: at.NumTokens


@at.typed
def _apply_rope(
    inputs: at.Keys | at.Queries,
    positions: at.SegmentPos,
    max_wavelength: int = _MAX_WAVELENGTH,
) -> at.Keys | at.Queries:
    """Applies RoPE to the first half of inputs.

    Args:
      inputs: Queries or keys.
      positions: Positions of each token in the sequence.
      max_wavelength: The maximum wavelength used for the sin and cos.

    Returns:
      Rotated keys or queries in first half (along with original in second half).
    """
    x_rope, x = jnp.split(inputs, 2, axis=-1)
    positions = jnp.expand_dims(
        positions, [i for i in range(x.ndim) if i not in (0, 1)]
    )

    freq_exponents = 2 * jnp.arange(x_rope.shape[-1] // 2) / x_rope.shape[-1]
    timescale = max_wavelength**freq_exponents
    inv_frequencies = 1.0 / timescale

    sinusoid_imp = positions * inv_frequencies
    sin = jnp.sin(sinusoid_imp).astype(inputs.dtype)
    cos = jnp.cos(sinusoid_imp).astype(inputs.dtype)

    first_half, second_half = jnp.split(x_rope, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin

    return jnp.concatenate([first_part, second_part, x], axis=-1)


@at.typed
def _compute_causal_mask(
    q_positions: jax.Array,
    k_positions: jax.Array,
    window_size: int,
    q_segment_ids: at.QuerySegmentIds | None,
    k_segment_ids: at.KeySegmentIds | None,
) -> at.AttentionMask:
    """Computes the causal mask for local attention.

    Args:
      q_positions: Position of each query token in the sequence.
      k_positions: Position of each key token in the sequence.
      window_size: The local attention window size.
      q_segment_ids: Optional segment id for each query token.
      k_segment_ids: Optional segment id for each key token.

    Returns:
      The mask that needs to be applied to the logits of the local attention.
    """
    # Mask for attending only to the same segment.
    if q_segment_ids is not None or k_segment_ids is not None:
        assert q_segment_ids is not None and k_segment_ids is not None
        same_segment_mask = (
            q_segment_ids[..., None] == k_segment_ids[..., None, :]
        )
    else:
        same_segment_mask = (k_positions >= 0)[..., None, :]

    # Mask for attending only to previous tokens.
    causal_mask = q_positions[..., None] >= k_positions[..., None, :]

    # Mask for attending only to things within the window size.
    window_cond = q_positions[..., None] <= (
        k_positions[..., None, :] + window_size
    )

    mask = jnp.logical_and(causal_mask, window_cond)
    mask = jnp.logical_and(same_segment_mask, mask)
    return mask


@at.typed
def _compute_forward_pass_mask(
    segment_pos: at.SegmentPos,
    window_size: int,
) -> at.AttentionMask:
    """Compute the forward pass mask.

    Args:
      segment_pos: Position of each token in the sequence.
      window_size: The local attention window size.

    Returns:
      The mask that needs to be applied to the logits when performing a forward
      pass (e.g. prompt processing) of the local attention.
    """
    segment_ids = jnp.cumsum(segment_pos == 0, axis=-1)
    positions = jnp.arange(segment_pos.shape[-1])
    positions = jnp.repeat(positions[None], segment_pos.shape[0], axis=0)
    return _compute_causal_mask(
        positions, positions, window_size, segment_ids, segment_ids
    )


@at.typed
def _compute_cache_mask(
    seq_len: int,
    cache_num_tokens: at.NumTokens,
    window_size: int,
) -> at.AttentionMask:
    """Computes the mask when there a KV-cache is present.

    Args:
      seq_len: The sequence length of the prompt.
      cache_num_tokens: The number of active tokens currently stored in the
        KV-cache.
      window_size: The local attention window size.

    Returns:
      The mask that needs to be applied to the logits when performing a single
      inference step with a KV-cache of the local attention.
    """
    q_positions = jnp.arange(seq_len)[None] + cache_num_tokens[:, None]

    k = cache_num_tokens[:, None] // window_size
    idx = jnp.arange(window_size)
    k_positions_now = idx[None] + k * window_size
    k_position_prev = idx[None] + (k - 1) * window_size
    mask = k_positions_now < cache_num_tokens[:, None]
    k_positions = mask * k_positions_now + (1 - mask) * k_position_prev
    k_positions = jnp.concatenate([k_positions, q_positions], axis=-1)
    return _compute_causal_mask(
        q_positions, k_positions, window_size, None, None
    )


@at.typed
def _update_attention_cache(
    keys: at.Keys,
    values: at.Values,
    segment_pos: at.SegmentPos,
    cache: AttentionBlockCache,
) -> AttentionBlockCache:
    """Updates the cache with the new keys and values.

    Args:
      keys: The new keys to be added to the cache.
      values: The new values to be added to the cache.
      segment_pos: Positions of each token in the sequence.
      cache: The dictionary with the cache to be updated.

    Returns:
      The updated cache dictionary.
    """
    seq_len = keys.shape[-3]
    window_size = cache.keys.shape[-3]
    n_fill = min(window_size, seq_len)

    if n_fill == 1:
        # Autogregressive sampling.
        idx0 = jnp.arange(keys.shape[0])
        idx1 = cache.num_tokens % window_size
        return AttentionBlockCache(
            keys=cache.keys.at[idx0, idx1].set(keys[:, 0]),
            values=cache.values.at[idx0, idx1].set(values[:, 0]),
            num_tokens=cache.num_tokens + 1,
        )
    else:
        # Processing a prompt in chunks.
        return _attention_cache_from_prompt(
            keys, values, segment_pos, window_size
        )


@at.typed
def _attention_cache_from_prompt(
    keys: at.Keys,
    values: at.Values,
    segment_pos: at.SegmentPos,
    window_size: int,
) -> AttentionBlockCache:
    """Creates a new cache from a prompt.

    Args:
      keys: The new keys to be added to an empty cache.
      values: The new values to be added to an empty cache.
      segment_pos: Positions of each token in the sequence.
      window_size: The local attention window size.

    Returns:
      An empty initialized KV-cache updated with the given keys and values.
    """
    w = min(window_size, keys.shape[1])
    padding = [[0, 0], [0, window_size - w], [0, 0], [0, 0]]
    num_tokens = segment_pos[:, -1] + 1

    # This ensures that the keys and values are right padded in the cache.
    right_padded_keys = _vmap_cache_roll(keys[:, -w:], num_tokens)
    right_padded_values = _vmap_cache_roll(values[:, -w:], num_tokens)

    return AttentionBlockCache(
        keys=jnp.pad(right_padded_keys, padding),
        values=jnp.pad(right_padded_values, padding),
        num_tokens=num_tokens,
    )


class LocalAttentionBlock(nn.Module):
    """Local Multi-Head Attention (MHA) block.

    Attributes:
      width: The width of the block.
      num_heads: The number of heads for the attention mechanism.
      window_size: The local attention window size.
      final_w_init_variance_scale: The scale for the initialization of the last
        layer of the block.
      dtype: dtype used for computation.
      param_dtype: dtype used for initializing parameters.
    """

    width: int
    num_heads: int
    window_size: int
    final_w_init_variance_scale: float = 1.0
    dtype: at.dtype | None = None
    param_dtype: at.dtype = jnp.float32

    @property
    def head_dim(self) -> int:
        """The dimension of each head."""
        return self.width // self.num_heads

    @property
    def kernel_init(self) -> nn.initializers.Initializer:
        """Initialization of the kernel for the queries, keys and values projections."""
        return nn.initializers.variance_scaling(
            scale=1.0,
            mode="fan_in",
            distribution="normal",
        )

    @property
    def out_kernel_init(self) -> nn.initializers.Initializer:
        """Initialization of the kernel for the final projection."""
        return nn.initializers.variance_scaling(
            scale=self.final_w_init_variance_scale,
            mode="fan_in",
            distribution="normal",
        )

    def setup(self):
        # Layers.
        self.q = nn.Dense(
            features=self.width,
            use_bias=False,
            kernel_init=self.kernel_init,
            name="proj_q",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.k = nn.Dense(
            features=self.head_dim,
            use_bias=False,
            kernel_init=self.kernel_init,
            name="proj_k",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.v = nn.Dense(
            features=self.head_dim,
            use_bias=False,
            kernel_init=self.kernel_init,
            name="proj_v",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.out = nn.Dense(
            features=self.width,
            use_bias=True,
            kernel_init=self.out_kernel_init,
            name="proj_final",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    @overload
    def __call__(
        self,
        x: at.Activations,
        segment_pos: at.SegmentPos,
        cache: AttentionBlockCache | None = None,
        return_cache: Literal[True] = True,
    ) -> tuple[at.Activations, AttentionBlockCache]: ...

    @overload
    def __call__(
        self,
        x: at.Activations,
        segment_pos: at.SegmentPos,
        cache: AttentionBlockCache | None = None,
        return_cache: Literal[False] = False,
    ) -> tuple[at.Activations, None]: ...

    @at.typed
    def __call__(
        self,
        x,
        segment_pos: at.SegmentPos,
        cache: AttentionBlockCache | None = None,
        return_cache: bool = True,
    ) -> tuple[at.Activations, AttentionBlockCache | None]:
        """Calls the local attention block.

        Args:
          x: Sequence of input activations.
          segment_pos: Positions of each token in the sequence.
          cache: Optiona KV-cache for the block, of previous keys and values.
          return_cache: Whether to compute and return the updated cache.

        Returns:
          Output of the block together with the updated cache. If `cache` is None
          than the returned updated cache is empty initialized and filled in from
          the input sequence.
        """
        b, t, _ = x.shape
        assert segment_pos.shape == (b, t), segment_pos.shape

        # Generate keys, values and queries.
        queries = self.q(x)
        keys = self.k(x)
        values = self.v(x)
        queries = einops.rearrange(
            queries, "... (n h) -> ... n h", n=self.num_heads
        )
        keys = einops.rearrange(keys, "... (n h) -> ... n h", n=1)
        values = einops.rearrange(values, "... (n h) -> ... n h", n=1)

        # Apply rotary embeddings.
        queries = _apply_rope(queries, segment_pos)
        keys = _apply_rope(keys, segment_pos)

        if cache is not None:
            no_cache_keys, no_cache_values = keys, values

            keys = jnp.concatenate([cache.keys, no_cache_keys], axis=-3)
            values = jnp.concatenate([cache.values, no_cache_values], axis=-3)
            attn_mask = _compute_cache_mask(
                t, cache.num_tokens, self.window_size
            )

            if return_cache:
                new_cache = _update_attention_cache(
                    no_cache_keys, no_cache_values, segment_pos, cache
                )
            else:
                new_cache = None

        else:
            attn_mask = _compute_forward_pass_mask(
                segment_pos, self.window_size
            )

            if return_cache:
                new_cache = _attention_cache_from_prompt(
                    keys, values, segment_pos, self.window_size
                )
            else:
                new_cache = None

        # Compute attention.
        logits = einops.einsum(queries, keys, "b t n h, b s n h -> b n t s")
        logits = logits * (self.head_dim**-0.5)
        # Expand for heads axis.
        attn_mask = jnp.expand_dims(attn_mask, axis=-3)

        masked_logits = jnp.where(attn_mask, logits, _MIN_LOGITS_VALUE)
        masked_logits = masked_logits.astype(jnp.float32)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(x.dtype)
        encoded = einops.einsum(probs, values, "b n t s, b s n h -> b t n h")
        encoded = einops.rearrange(
            encoded, "... n h -> ... (n h)", n=self.num_heads
        )
        attn_output = self.out(encoded)

        return attn_output, new_cache

    @classmethod
    def default_state(
        cls,
        batch_size: int,
        window_size: int,
        heads_dim: int,
        dtype: at.dtype,
    ) -> AttentionBlockCache:
        """Initializes an empty KV-cache for the block."""
        return AttentionBlockCache(
            keys=jnp.zeros(
                (batch_size, window_size, 1, heads_dim), dtype=dtype
            ),
            values=jnp.zeros(
                (batch_size, window_size, 1, heads_dim), dtype=dtype
            ),
            num_tokens=jnp.zeros([batch_size], dtype=jnp.int32),
        )
