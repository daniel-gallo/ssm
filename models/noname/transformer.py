import dataclasses
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random

from models.noname.utils import get_sinusoidal_embeddings


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class KVCache:
    keys: jax.Array
    values: jax.Array
    i: jax.Array

    @classmethod
    def init(cls, bs: int, seq_len: int, num_heads: int, d: int):
        assert d % num_heads == 0
        d_head = d // num_heads

        return cls(
            keys=jnp.zeros((bs, seq_len, num_heads, d_head)),
            values=jnp.zeros((bs, seq_len, num_heads, d_head)),
            i=jnp.array(0),
        )

    def update(self, key: jax.Array, value: jax.Array):
        bs, seq_len, num_heads, d_head = self.keys.shape
        assert key.shape == (bs, num_heads, d_head)
        assert value.shape == (bs, num_heads, d_head)

        return KVCache(
            keys=self.keys.at[:, self.i, :, :].set(key),
            values=self.values.at[:, self.i, :, :].set(value),
            i=self.i + 1,
        )


def dot_product_attention(query, key, value):
    bs, seq_len, num_heads, d = query.shape
    dtype = query.dtype
    scale = d**-0.5
    query = query * scale
    # Switch dtype right before the dot-product for numerical stability.
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    weights = jnp.einsum("...qhd,...khd->...hqk", query, key)
    mask = nn.make_causal_mask(jnp.zeros((bs, seq_len)))
    weights = jnp.where(mask, weights, jnp.finfo(jnp.float32).min)
    weights = nn.softmax(weights, axis=-1)
    # After softmax, switch back to the original dtype
    weights = weights.astype(dtype)
    new_vals = jnp.einsum("...hqk,...khd->...qhd", weights, value)
    new_vals = new_vals.astype(dtype)
    return new_vals


def cached_dot_product_attention(q, cache: KVCache):
    bs, num_heads, d = q.shape

    scale = d**-0.5
    q = q * scale

    weights = jnp.einsum("...hd,...khd->...hk", q, cache.keys)

    _, _, seq_len = weights.shape
    mask = (jnp.arange(seq_len) < cache.i)[jnp.newaxis, jnp.newaxis, :]
    weights = jnp.where(mask, weights, jnp.finfo(jnp.float32).min)

    weights = nn.softmax(weights, axis=-1)
    new_vals = jnp.einsum("...hk,...khd->...hd", weights, cache.values)
    return new_vals


class Attention(nn.Module):
    num_heads: int

    @nn.compact
    def __call__(self, x, cache: Optional[KVCache]):
        d = x.shape[-1]
        assert d % self.num_heads == 0

        qkv = nn.DenseGeneral(
            features=(self.num_heads, d // self.num_heads * 3)
        )(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        if cache is None:
            x = dot_product_attention(q, k, v)
        else:
            cache = cache.update(k, v)
            x = cached_dot_product_attention(q, cache)

        x = nn.DenseGeneral(d, axis=(-2, -1))(x)
        return x, cache


class TransformerBlock(nn.Module):
    num_heads: int

    @nn.compact
    def __call__(self, x, cache: Optional[KVCache] = None):
        d = x.shape[-1]

        residual, cache = Attention(self.num_heads)(nn.LayerNorm()(x), cache)
        x = x + residual

        x = x + nn.Sequential(
            [
                nn.Dense(4 * d),
                nn.gelu,
                nn.Dense(d),
            ]
        )(nn.LayerNorm()(x))

        return x, cache


class Transformer(nn.Module):
    d: int
    num_cats: int
    num_layers: int
    num_heads: int

    def setup(self):
        self.embed = nn.Embed(self.num_cats + 1, self.d)
        self.blocks = [
            TransformerBlock(self.num_heads) for _ in range(self.num_layers)
        ]
        self.cls_head = nn.Dense(self.num_cats)

    def __call__(self, x):
        bs, seq_len = x.shape

        # Embed and move right
        x = jnp.concatenate(
            arrays=[jnp.full((bs, 1), self.num_cats), x[:, :-1]], axis=-1
        )
        x = self.embed(x)

        # Add positional embeddings
        pos_embed = get_sinusoidal_embeddings(jnp.arange(seq_len), self.d)
        x = x + pos_embed[jnp.newaxis, :, :]

        for block in self.blocks:
            x, _ = block(x)

        x = self.cls_head(x)
        return x

    def sample(self, bs: int, seq_len: int, rng, temperature: float):
        caches = [
            KVCache.init(bs, seq_len, self.num_heads, self.d)
            for _ in range(self.num_layers)
        ]
        tokens = jnp.full(shape=(bs, seq_len), fill_value=self.num_cats)

        rngs = random.split(rng, seq_len)
        pos_embed = get_sinusoidal_embeddings(jnp.arange(seq_len), self.d)

        def step_fn(carry, i):
            caches, tokens = carry

            x = jax.lax.cond(
                i == 0,
                lambda _: jnp.full((bs,), self.num_cats),
                lambda i: tokens[:, i - 1],
                i,
            )

            x = self.embed(x) + pos_embed[None, i]

            for block_id, block in enumerate(self.blocks):
                x, caches[block_id] = block(x, caches[block_id])

            x = self.cls_head(x)
            token = random.categorical(rngs[i], x / temperature)
            tokens = tokens.at[:, i].set(token)

            return (caches, tokens), None

        (_, tokens), _ = jax.lax.scan(
            step_fn, (caches, tokens), jnp.arange(seq_len)
        )
        return tokens


if __name__ == "__main__":
    x = jnp.zeros((8, 10), dtype=int)
    key = random.key(0)

    transformer = Transformer(d=128, num_cats=16, num_layers=3, num_heads=4)
    params = transformer.init(key, x)
    y = transformer.apply(params, x)
    print(y.shape)

    sample = transformer.apply(params, 2, 10, key, 1.0, method="sample")
    print(sample.shape)
