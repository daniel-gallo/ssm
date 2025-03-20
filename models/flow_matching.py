import dataclasses
from typing import Literal

import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange

from hps import Hyperparams
from models.recurrence import RNNHyperparams, get_recurrent_block


@dataclasses.dataclass(frozen=True)
class FlowHyperparams(Hyperparams):
    rnn: RNNHyperparams = RNNHyperparams()

    # Model architecture
    pool_temporal: tuple[int, ...] = (2, 2, 2, 2, 2)
    pool_features: tuple[int, ...] = (1, 1, 1, 1, 1)
    conv_blocks: tuple[int, ...] = (4, 4, 4, 2, 2)
    temporal_blocks: tuple[int, ...] = (0, 0, 0, 2, 2, 4)

    use_norm: bool = True
    use_inner_gating: bool = True
    use_temporal_cnn: bool = True
    skip_residual: Literal["add"] = "add"

    d_base: int = 64
    d_cond: int = 64
    rnn_hidden_size: int = 256
    ff_expand: int = 2
    cnn_kernel_size: int = 3
    block_last_scale: float = 0.125

    @property
    def model(self):
        return FlowModel(self)

    @property
    def sample_prior(self):
        return FlowModel.sample_prior


def get_timestep_embedding(t, d):
    # TODO: implemennt timestep embedding
    ...


class Pool(nn.Module):
    H: FlowHyperparams
    pool_temporal: int
    pool_features: int
    down: bool = True

    @nn.compact
    def __call__(self, x):
        _, _, d = x.shape
        if self.down:
            x = rearrange(
                x,
                "... (l factor) d -> ... l (factor d)",
                factor=self.pool_temporal,
            )
            x = nn.Dense(d * self.pool_features)(x)
        else:
            x = nn.Dense((d * self.pool_temporal) // self.pool_features)
            x = rearrange(
                x,
                "... l (factor d) -> ... (l factor) d",
                factor=self.pool_temporal,
            )
        return x


class ConditionedLayerNorm(nn.Module):
    @nn.compact
    def __call__(self, x, c):
        scale = nn.Dense(
            features=1,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(c)[:, :, None]

        bias = nn.Dense(
            features=1,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(c)[:, :, None]

        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x = (1 + scale) * x + bias
        return x


class Gate(nn.Module):
    @nn.compact
    def __call__(self, c):
        return nn.Dense(
            features=1,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(c)[:, :, None]


class DenseBlock(nn.Module):
    H: FlowHyperparams
    expand: int | None = None
    reduce: int = 1

    @nn.compact
    def __call__(self, x, c):
        _, _, d = x.shape
        expand = self.expand or self.H.ff_expand
        skip = x
        x = nn.Dense(d * expand)(x)
        if self.H.use_inner_gating:
            gate_x = nn.Dense(d * expand)(skip)
            x = x * nn.gelu(gate_x)
        else:
            x = nn.gelu(x)
        return nn.Dense(d // self.reduce)(x)


class ConvBlock(nn.Module):
    H: FlowHyperparams
    kernel_size: int | None = None
    expand: int | None = None
    reduce: int = 1

    @nn.compact
    def __call__(self, x, c):
        _, _, d = x.shape
        expand = self.expand or self.H.ff_expand
        kernel_size = self.kernel_size or self.H.cnn_kernel_size

        skip = x
        x = nn.Conv(
            d * expand,
            kernel_size,
            padding="SAME",
        )(x)
        if self.H.use_inner_gating:
            gate_x = nn.Dense(d * expand)(skip)
            x = x * nn.gelu(gate_x)
        else:
            x = nn.gelu(x)
        return nn.Dense(d // self.reduce)(x)


class RNNBlock(nn.Module):
    H: RNNHyperparams
    bidirectional: bool = False

    @nn.compact
    def __call__(self, x, c):
        _, _, d = x.shape
        recurrent_block = get_recurrent_block(self.H)
        skip = x
        x, _ = recurrent_block(
            self.H,
            d_hidden=self.H.d_hidden,
            d_out=d,
        )(skip)
        if self.bidirectional:
            x_bwd, _ = recurrent_block(
                self.H,
                d_hidden=self.H.d_hidden,
                d_out=d,
                reverse=True,
            )(skip)
            x = jnp.concatenate([x, x_bwd], axis=-1)
        return nn.Dense(d)(x)


class TemporalMixingBlock(nn.Module):
    H: FlowHyperparams
    kernel_size: int | None = None

    @nn.compact
    def __call__(self, x, c):
        _, _, d = x.shape
        kernel_size = self.kernel_size or self.H.cnn_kernel_size

        skip = x
        x = (
            nn.Conv(d, kernel_size, padding="SAME")(x)
            if self.H.use_temporal_cnn
            else x
        )
        x = RNNBlock(
            self.H.rnn,
            bidirectional=True,
        )(x, c)
        if self.H.use_inner_gating:
            gate_x = nn.Dense(d)(skip)
            x = x * nn.gelu(gate_x)
        else:
            x = nn.gelu(x)
        return nn.Dense(self.d_out)(x)


class ResBlock(nn.Module):
    H: FlowHyperparams
    inner: nn.Module

    @nn.compact
    def __call__(self, x, c):
        skip = x
        x = ConditionedLayerNorm()(x, c) if self.H.use_norm else x
        x = self.inner(x, c)
        return skip + Gate()(c) * x


class TemporalStack(nn.Module):
    H: FlowHyperparams
    temporal_blocks: int = 1

    @nn.compact
    def __call__(self, x, c):
        for _ in range(self.temporal_blocks):
            x = ResBlock(self.H, inner=TemporalMixingBlock(self.H))(x, c)
            x = ResBlock(self.H, inner=DenseBlock(self.H))(x, c)
        return x


class SkipBlock(nn.Module):
    H: FlowHyperparams
    inner: nn.Module
    conv_blocks: int = 0
    temporal_blocks: int = 0
    pool_temporal: int = 1
    pool_features: int = 1

    @nn.compact
    def __call__(self, x, c):
        skip = x
        x = self.layer(x, c)

        def _conv_block(x, c):
            x = ResBlock(self.H, inner=ConvBlock(self.H))(x, c)
            x = ResBlock(self.H, inner=DenseBlock(self.H))(x, c)
            return x

        def _temporal_block(x, c):
            x = ResBlock(self.H, inner=TemporalMixingBlock(self.H))(x, c)
            x = ResBlock(self.H, inner=DenseBlock(self.H))(x, c)
            return x

        skip = x

        for _ in range(self.conv_blocks):
            x = _conv_block(x, c)
        for _ in range(self.temporal_blocks):
            x = _temporal_block(x, c)

        x = Pool(
            self.H,
            pool_temporal=self.pool_temporal,
            pool_features=self.pool_features,
            down=True,
        )(x)
        x = self.inner(x, c)
        x = Pool(
            self.H,
            pool_temporal=self.pool_temporal,
            pool_features=self.pool_features,
            down=False,
        )

        for _ in range(self.temporal_blocks):
            x = _temporal_block(x, c)
        for _ in range(self.conv_blocks):
            x = _conv_block(x, c)

        if self.H.skip_residual == "add":
            return skip + Gate()(c) * x
        else:
            raise ValueError(f"Unknown skip_residual: {self.H.skip_residual}")


class Backbone(nn.Module):
    H: FlowHyperparams

    @nn.compact
    def __call__(self, x, t):
        c = nn.Sequential(
            [
                nn.Dense(self.H.d_cond),
                nn.gelu,
                nn.Dense(self.H.d_cond),
                nn.gelu,
            ]
        )(get_timestep_embedding(t, self.H.d_cond))

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
        return block(x, c)


class FlowModel(nn.Module):
    H: FlowHyperparams

    def setup(self):
        self.backbone = Backbone(self.H)
