import dataclasses

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import lax, random
from typing_extensions import Union

from hps import Hyperparams
from models.rnn import get_recurrent_block


def log_likelihood(logits, x):
    bat, seq, chan, cat = logits.shape
    assert x.shape == (bat, seq, chan)
    return jnp.sum(
        jnp.take_along_axis(jax.nn.log_softmax(logits), x[..., None], -1)
    )


def loss_and_metrics(logits, x):
    normalizer = x.size * jnp.log(2)
    ll = log_likelihood(logits, x) / normalizer
    loss = -ll
    return loss, {
        "loss": loss,
        "log-like": ll,
        "mean_0": jnp.mean(logits[:, 0]),
        "max_0": jnp.max(logits[:, 0]),
        "min_0": jnp.min(logits[:, 0]),
        "mean_l": jnp.mean(logits[:, -1]),
        "max_l": jnp.max(logits[:, -1]),
        "min_l": jnp.min(logits[:, -1]),
    }


@dataclasses.dataclass(frozen=True)
class ARHyperparams(Hyperparams):
    pool_temporal: tuple[int, ...] = (4, 4)
    pool_features: tuple[int, ...] = (2, 2)

    rnn_init_minval: float = 0.9
    rnn_init_maxval: float = 0.99
    rnn_init_imag: float = 0.1
    rnn_norm_input: bool = True
    rnn_hidden_size: int = 256
    rnn_out_size: int = 16
    rnn_pos_embedding: bool = True
    rnn_block: str = "rglru"
    rnn_only_real: bool = False

    rnn_n_diag_blocks: int = 64

    base_dim: int = 64
    ff_expand: int = 2
    rnn_last_scale: float = 0.25
    rnn_n_layers: int = 2

    scan_implementation: str = "linear_native"

    @property
    def model(self):
        return ARModel(self)

    @property
    def sample_prior(self):
        return ARModel.sample_prior


class DownPool(nn.Module):
    H: ARHyperparams
    input_dim: int
    pool_temporal: int
    pool_features: int

    def setup(self):
        self.linear = nn.Dense(self.input_dim * self.pool_features)

    def __call__(self, x):
        batch_size, seq_len, dim = x.shape
        x = rearrange(x, "...(l m) d -> ... l (m d)", m=self.pool_temporal)
        return self.linear(x), None


class UpPool(nn.Module):
    H: ARHyperparams
    input_dim: int
    pool_temporal: int
    pool_features: int

    def setup(self):
        assert (self.input_dim * self.pool_temporal) % self.pool_features == 0
        self.linear = nn.Dense(
            (self.input_dim * self.pool_temporal) // self.pool_features
        )

    def __call__(self, x):
        batch_size, seq_len, dim = x.shape
        x = self.linear(x)
        # ensures causal relationship
        x = jnp.pad(x[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
        x = rearrange(x, "... l (m d) -> ... (l m) d", m=self.pool_temporal)
        return x, None


class ResBlock(nn.Module):
    H: ARHyperparams
    expand: Union[int, None] = None
    last_scale: float = 1.0

    @nn.compact
    def __call__(self, x, deterministic=False):
        bs, seq_len, dim = x.shape
        expand = self.expand or self.H.ff_expand
        z = nn.LayerNorm(feature_axes=-1)(x)
        z = nn.Dense(dim * expand)(z)
        z = nn.gelu(z)
        z = nn.Dense(dim)(z)
        z = z * self.last_scale
        return x + z, None


class RNNBlock(nn.Module):
    H: ARHyperparams
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
        self.norm = nn.LayerNorm(feature_axes=-1)

    def __call__(self, x, h_prev=None):
        assert h_prev is None or not self.bidirectional
        identity = x
        x = self.norm(x)
        x_fwd, h_next = self.forward(x, h_prev)
        x = (x_fwd + self.backward(x)[0]) / 2 if self.bidirectional else x_fwd

        x = nn.gelu(x)
        x = self.last_dense(x) * self.last_scale
        x = x + identity if self.residual else x
        return x, h_next


class ARModel(nn.Module):
    H: ARHyperparams

    def setup(self):
        self.input_mlp = nn.Dense(self.H.base_dim)
        self.cls_mlp = nn.Dense(self.H.data_num_cats)
        self.norm = nn.LayerNorm(feature_axes=-1)

        d_layers = []
        model_dim = self.H.base_dim
        for p, expand in zip(self.H.pool_temporal, self.H.pool_features):
            d_layers.append(
                DownPool(
                    self.H,
                    model_dim,
                    pool_temporal=p,
                    pool_features=expand,
                )
            )
            model_dim = model_dim * expand

        c_layers = []
        for _ in range(self.H.rnn_n_layers):
            c_layers.append(
                RNNBlock(
                    self.H,
                    model_dim,
                    residual=True,
                    last_scale=self.H.rnn_last_scale,
                )
            )
            c_layers.append(ResBlock(self.H, last_scale=self.H.rnn_last_scale))

        u_layers = []
        for p, expand in zip(
            self.H.pool_temporal[::-1], self.H.pool_features[::-1]
        ):
            block = []
            block.append(
                UpPool(
                    self.H,
                    model_dim,
                    pool_temporal=p,
                    pool_features=expand,
                )
            )

            model_dim = model_dim // expand

            for _ in range(self.H.rnn_n_layers):
                block.append(
                    RNNBlock(
                        self.H,
                        model_dim,
                        residual=True,
                        last_scale=self.H.rnn_last_scale,
                    )
                )
                block.append(ResBlock(self.H, last_scale=self.H.rnn_last_scale))
            u_layers.append(block)

        self.d_layers = d_layers
        self.c_layers = c_layers
        self.u_layers = u_layers

        assert model_dim == self.H.base_dim

    def evaluate(self, x):
        batch_size, seq_len, _ = x.shape

        x = self.H.data_preprocess_fn(x)
        x = jnp.pad(x[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
        x = self.input_mlp(x)
        outputs = []
        outputs.append(x)

        for layer in self.d_layers:
            x, _ = layer(x)
            outputs.append(x)

        for layer in self.c_layers:
            x, _ = layer(x)
        x = x + outputs.pop()

        for block in self.u_layers:
            for layer in block:
                x, _ = layer(x)
                if isinstance(layer, UpPool):
                    x = x + outputs.pop()
                    outputs.append(x)
            x = x + outputs.pop()

        # x = self.norm(x)
        x = jnp.reshape(
            self.cls_mlp(x),
            (
                batch_size,
                seq_len,
                self.H.data_num_channels,
                self.H.data_num_cats,
            ),
        )

        return x

    def __call__(self, x, rng=None):
        return loss_and_metrics(self.evaluate(x), x)

    def sample_prior(self, gen_len, n_samples, rng):
        x = jnp.zeros((n_samples, gen_len, self.H.data_num_channels), "int32")

        def fix_point(i, x):
            return random.categorical(rng, self.evaluate(x), -1)

        return lax.fori_loop(0, gen_len, fix_point, x)
