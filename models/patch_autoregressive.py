import dataclasses

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import lax, random

from hps import Hyperparams
from models.recurrence import RNNHyperparams, get_recurrent_block


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
class PatchARHyperparams(Hyperparams):
    rnn: RNNHyperparams = RNNHyperparams()

    pool_temporal: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    pool_features: tuple[int, ...] = (1, 1, 1, 1, 1, 1)

    use_norm: bool = True
    use_gating: bool = True

    base_dim: int = 64
    rnn_hidden_size: int = 256
    ff_expand: int = 2
    cnn_kernel_size: int = 3
    block_last_scale: float = 0.125
    rnn_n_layers: int = 4
    cnn_n_layers: int = 2

    @property
    def model(self):
        return PatchARModel(self)

    @property
    def sample_prior(self):
        return PatchARModel.sample_prior


class DownPool(nn.Module):
    H: PatchARHyperparams
    input_dim: int
    pool_temporal: int
    pool_features: int

    def setup(self):
        self.linear = nn.Dense(self.input_dim * self.pool_features)

    def __call__(self, x):
        batch_size, seq_len, dim = x.shape
        x = rearrange(x, "...(l m) d -> ... l (m d)", m=self.pool_temporal)
        return self.linear(x)


class UpPool(nn.Module):
    H: PatchARHyperparams
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
        return x


class ResBlock(nn.Module):
    H: PatchARHyperparams
    layer: nn.Module
    last_scale: float = 1.0

    @nn.compact
    def __call__(self, x, deterministic=False):
        bs, seq_len, dim = x.shape

        z = nn.LayerNorm()(x) if self.H.use_norm else x
        z = self.layer(z)
        z = z * self.last_scale
        return x + z


class MLPBlock(nn.Module):
    H: PatchARHyperparams
    expand: int | None = None

    @nn.compact
    def __call__(self, x):
        bs, seq_len, dim = x.shape
        expand = self.expand or self.H.ff_expand
        z = nn.Dense(dim * expand)(x)
        if self.H.use_gating:
            gated_x = nn.Dense(dim * expand)(x)
            z = z * nn.gelu(gated_x)
        else:
            z = nn.gelu(z)
        return nn.Dense(dim)(z)


class RNNBlock(nn.Module):
    H: PatchARHyperparams
    d_out: int

    @nn.compact
    def __call__(self, x, h_prev=None):
        recurrent_block = get_recurrent_block(self.H.rnn)
        x_fwd, _ = recurrent_block(
            self.H.rnn,
            d_hidden=self.H.rnn_hidden_size,
            d_out=self.d_out,
        )(x)
        if self.H.use_gating:
            gated_x = nn.Dense(self.d_out)(x)
            x = x_fwd * nn.gelu(gated_x)
        else:
            x = nn.gelu(x_fwd)
        return nn.Dense(self.d_out)(x)


class ConvBlock(nn.Module):
    H: PatchARHyperparams
    expand: int | None = None

    @nn.compact
    def __call__(self, x):
        bs, seq_len, dim = x.shape
        expand = self.expand or self.H.ff_expand
        z = nn.Conv(dim * expand, self.H.cnn_kernel_size, padding="CAUSAL")(x)
        if self.H.use_gating:
            gated_x = nn.Dense(dim * expand)(x)
            z = z * nn.gelu(gated_x)
        else:
            z = nn.gelu(z)
        return nn.Dense(dim)(z)


class SkipBlock(nn.Module):
    H: PatchARHyperparams
    inner_layer: nn.Module
    pool_temporal: int
    pool_feature: int

    @nn.compact
    def __call__(self, x):
        def _conv_block(expand=None, last_scale=1.0):
            return ResBlock(
                self.H,
                layer=ConvBlock(self.H, expand=expand),
                last_scale=last_scale,
            )

        def _mlp_block(expand=None, last_scale=1.0):
            return ResBlock(
                self.H,
                layer=MLPBlock(self.H, expand=expand),
                last_scale=last_scale,
            )

        z = x
        for _ in range(self.H.cnn_n_layers):
            z = _conv_block(self.H.ff_expand, self.H.block_last_scale)(z)
            z = _mlp_block(self.H.ff_expand, self.H.block_last_scale)(z)

        z = DownPool(
            self.H,
            z.shape[-1],
            self.pool_temporal,
            self.pool_feature,
        )(z)
        z = self.inner_layer(z)
        z = UpPool(
            self.H,
            z.shape[-1],
            self.pool_temporal,
            self.pool_feature,
        )(z)

        for _ in range(self.H.cnn_n_layers):
            z = _conv_block(self.H.ff_expand, self.H.block_last_scale)(z)
            z = _mlp_block(self.H.ff_expand, self.H.block_last_scale)(z)

        return x + z


class TemporalStack(nn.Module):
    H: PatchARHyperparams

    @nn.compact
    def __call__(self, x):
        def _rnn_block(d_out, last_scale=1.0):
            return ResBlock(
                self.H,
                layer=RNNBlock(self.H, d_out),
                last_scale=last_scale,
            )

        def _mlp_block(expand=None, last_scale=1.0):
            return ResBlock(
                self.H,
                layer=MLPBlock(self.H, expand=expand),
                last_scale=last_scale,
            )

        for _ in range(self.H.rnn_n_layers):
            x = _rnn_block(x.shape[-1], self.H.block_last_scale)(x)
            x = _mlp_block(self.H.ff_expand, self.H.block_last_scale)(x)

        return x


class PatchARModel(nn.Module):
    H: PatchARHyperparams

    def setup(self):
        self.input_mlp = nn.Dense(
            self.H.base_dim,
            bias_init=jax.nn.initializers.normal(0.5),
        )
        self.cls_mlp = nn.Dense(self.H.data_num_cats)
        self.norm = nn.LayerNorm()

        block = TemporalStack(self.H)
        for p_temporal, p_features in zip(
            reversed(self.H.pool_temporal), reversed(self.H.pool_features)
        ):
            block = SkipBlock(self.H, block, p_temporal, p_features)
        self.temporal_pyramid = block

    def evaluate(self, x):
        batch_size, seq_len, _ = x.shape

        x = self.H.data_preprocess_fn(x)
        x = jnp.pad(x[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
        x = self.input_mlp(x)

        x = self.temporal_pyramid(x)

        x = self.norm(x)
        return jnp.reshape(
            self.cls_mlp(x),
            (
                batch_size,
                seq_len,
                self.H.data_num_channels,
                self.H.data_num_cats,
            ),
        )

    def __call__(self, x, rng=None):
        return loss_and_metrics(self.evaluate(x), x)

    def sample_prior(self, gen_len, n_samples, rng):
        x = jnp.zeros((n_samples, gen_len, self.H.data_num_channels), "int32")

        def fix_point(i, x):
            return random.categorical(rng, self.evaluate(x), -1)

        return lax.fori_loop(0, gen_len, fix_point, x)
