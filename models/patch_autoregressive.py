import dataclasses
from typing import Literal

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


def fourier_features(x, d):
    max_period = 10_000
    half = d // 2

    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half) / half)
    args = x[..., None] * freqs
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    return rearrange(embedding, "b t c f -> b t (c f)")


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

    # Model architecture
    pool_temporal: tuple[int, ...] = (4, 4, 4)
    pool_features: tuple[int, ...] = (1, 1, 1)
    conv_blocks: tuple[int, ...] = (4, 4, 4)
    temporal_blocks: tuple[int, ...] = (0, 0, 2, 4)

    use_norm: bool = True
    use_gating: bool = True
    use_temporal_cnn: bool = True
    skip_residual: Literal["add", "mean", "mlp"] = "add"
    input_fourier_features: bool = True

    base_dim: int = 64
    rnn_hidden_size: int = 256
    ff_expand: int = 2
    cnn_kernel_size: int = 3
    block_last_scale: float = 0.125
    dropout_rate: float = 0.0

    @property
    def model(self):
        return PatchARModel(self)

    @property
    def sample_prior(self):
        return PatchARModel.sample_prior


class DownPool(nn.Module):
    H: PatchARHyperparams
    pool_temporal: int
    out_features: int

    def setup(self):
        self.linear = nn.Dense(self.out_features)

    def __call__(self, x):
        batch_size, seq_len, dim = x.shape
        x = rearrange(x, "...(l m) d -> ... l (m d)", m=self.pool_temporal)
        return self.linear(x)


class UpPool(nn.Module):
    H: PatchARHyperparams
    pool_temporal: int
    out_features: int

    def setup(self):
        self.linear = nn.Dense(self.out_features * self.pool_temporal)

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
    def __call__(self, x, training=False):
        bs, seq_len, dim = x.shape

        z = nn.LayerNorm()(x) if self.H.use_norm else x
        z = self.layer(z)
        z = nn.Dropout(self.H.dropout_rate, deterministic=not training)(z)
        z = z * self.last_scale
        return x + z


class MLPBlock(nn.Module):
    H: PatchARHyperparams
    expand: int | None = None
    reduce: int = 1

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
        return nn.Dense(dim // self.reduce)(z)


class TemporalMixingBlock(nn.Module):
    H: PatchARHyperparams
    d_out: int

    @nn.compact
    def __call__(self, x, h_prev=None):
        recurrent_block = get_recurrent_block(self.H.rnn)
        z = (
            nn.LayerNorm()(
                nn.Conv(self.d_out, self.H.cnn_kernel_size, padding="CAUSAL")(x)
            )
            if self.H.use_temporal_cnn
            else x
        )
        z, _ = recurrent_block(
            self.H.rnn,
            d_hidden=self.H.rnn_hidden_size,
            d_out=self.d_out,
        )(z)
        if self.H.use_gating:
            gated_x = nn.Dense(self.d_out)(x)
            x = z * nn.gelu(gated_x)
        else:
            x = nn.gelu(z)
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
    conv_blocks: int = 0
    temporal_blocks: int = 0
    pool_temporal: int = 1
    pool_feature: int = 1

    @nn.compact
    def __call__(self, x, training=False):
        in_features = x.shape[-1]

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

        def _temporal_block(d_out, last_scale=1.0):
            return ResBlock(
                self.H,
                layer=TemporalMixingBlock(self.H, d_out),
                last_scale=last_scale,
            )

        z = x

        for _ in range(self.conv_blocks):
            z = _conv_block(self.H.ff_expand, self.H.block_last_scale)(
                z, training
            )
            z = _mlp_block(self.H.ff_expand, self.H.block_last_scale)(
                z, training
            )

        for _ in range(self.temporal_blocks):
            z = _temporal_block(z.shape[-1], self.H.block_last_scale)(
                z, training
            )
            z = _mlp_block(self.H.ff_expand, self.H.block_last_scale)(
                z, training
            )

        z = DownPool(
            self.H,
            self.pool_temporal,
            self.H.base_dim * self.pool_feature,
        )(z)
        z = self.inner_layer(z, training)
        z = UpPool(
            self.H,
            self.pool_temporal,
            in_features,
        )(z)

        for _ in range(self.temporal_blocks):
            z = _temporal_block(z.shape[-1], self.H.block_last_scale)(
                z, training
            )
            z = _mlp_block(self.H.ff_expand, self.H.block_last_scale)(
                z, training
            )

        for _ in range(self.conv_blocks):
            z = _conv_block(self.H.ff_expand, self.H.block_last_scale)(
                z, training
            )
            z = _mlp_block(self.H.ff_expand, self.H.block_last_scale)(
                z, training
            )

        match self.H.skip_residual:
            case "add":
                return x + z
            case "mean":
                return (x + z) / 2
            case "mlp":
                return MLPBlock(self.H, expand=1, reduce=2)(
                    jnp.concatenate([x, z], axis=-1)
                )


class TemporalStack(nn.Module):
    H: PatchARHyperparams
    temporal_blocks: int = 1

    @nn.compact
    def __call__(self, x, training=False):
        def _temporal_block(d_out, last_scale=1.0):
            return ResBlock(
                self.H,
                layer=TemporalMixingBlock(self.H, d_out),
                last_scale=last_scale,
            )

        def _mlp_block(expand=None, last_scale=1.0):
            return ResBlock(
                self.H,
                layer=MLPBlock(self.H, expand=expand),
                last_scale=last_scale,
            )

        for _ in range(self.temporal_blocks):
            x = _temporal_block(x.shape[-1], self.H.block_last_scale)(
                x, training
            )
            x = _mlp_block(self.H.ff_expand, self.H.block_last_scale)(
                x, training
            )

        return x


class PatchARModel(nn.Module):
    H: PatchARHyperparams

    def setup(self):
        if not self.H.input_fourier_features:
            self.input_mlp = nn.Dense(
                self.H.base_dim,
                bias_init=jax.nn.initializers.normal(0.5),
            )

        self.cls_mlp = nn.Sequential(
            [
                ConvBlock(self.H, expand=2),
                ConvBlock(self.H, expand=2),
                nn.Dense(self.H.data_num_cats),
            ]
        )
        self.norm = nn.LayerNorm()

        block = TemporalStack(self.H, self.H.temporal_blocks[-1])
        for p_temporal, p_features, conv_blocks, temp_blocks in zip(
            reversed(self.H.pool_temporal),
            reversed(self.H.pool_features),
            reversed(self.H.conv_blocks),
            reversed(self.H.temporal_blocks[:-1]),
        ):
            block = SkipBlock(
                self.H,
                block,
                conv_blocks=conv_blocks,
                temporal_blocks=temp_blocks,
                pool_temporal=p_temporal,
                pool_feature=p_features,
            )
        self.temporal_pyramid = block

    def evaluate(self, x, training=False):
        batch_size, seq_len, _ = x.shape
        inp = x

        if self.H.input_fourier_features:
            assert self.H.base_dim % self.H.data_num_channels == 0
            x = fourier_features(
                inp, self.H.base_dim // self.H.data_num_channels
            )
        else:
            x = self.H.data_preprocess_fn(x)
        x = jnp.pad(x[:, :-1, :], ((0, 0), (1, 0), (0, 0)))

        if not self.H.input_fourier_features:
            x = self.input_mlp(x)

        x = self.temporal_pyramid(x, training)

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

    def __call__(self, x, rng=None, training=False):
        return loss_and_metrics(self.evaluate(x, training), x)

    def sample_prior(self, gen_len, n_samples, rng):
        x = jnp.zeros((n_samples, gen_len, self.H.data_num_channels), "int32")

        def fix_point(i, x):
            return random.categorical(rng, self.evaluate(x), -1)

        return lax.fori_loop(0, gen_len, fix_point, x)
