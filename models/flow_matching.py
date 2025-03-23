import dataclasses
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from einops import rearrange
from jax import lax

from hps import Hyperparams
from models.recurrence import RNNHyperparams, get_recurrent_block


@dataclasses.dataclass(frozen=True)
class FlowHyperparams(Hyperparams):
    rnn: RNNHyperparams = RNNHyperparams()

    # Model architecture
    pool_temporal: tuple[int, ...] = (2, 2, 2, 2, 2)
    pool_features: tuple[int, ...] = (1, 1, 1, 1, 1)
    conv_blocks: tuple[int, ...] = (8, 4, 4, 2, 2)
    temporal_blocks: tuple[int, ...] = (0, 2, 4, 4, 8, 8)

    # FOR MNIST
    # pool_temporal: tuple[int, ...] = (2, 2, 2, 2)
    # pool_features: tuple[int, ...] = (1, 1, 1, 1)
    # conv_blocks: tuple[int, ...] = (4, 4, 4, 2)
    # temporal_blocks: tuple[int, ...] = (0, 2, 2, 4)

    use_norm: bool = True
    use_inner_gating: bool = True
    use_temporal_cnn: bool = True
    skip_residual: Literal["add"] = "add"

    d_base: int = 128
    d_cond: int = 64
    rnn_hidden_size: int = 128
    ff_expand: int = 2
    cnn_kernel_size: int = 3

    sampling_steps: int = 200

    @property
    def model(self):
        return FlowModel(self)

    @property
    def sample_prior(self):
        return FlowModel.sample_prior


def get_timestep_embedding(t, d):
    half = d // 2

    freqs = 2 * jnp.pi * (jnp.arange(half) + 1)
    args = t * freqs[None, :]
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


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
            x = nn.Dense((d * self.pool_temporal) // self.pool_features)(x)
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
        )(c)

        bias = nn.Dense(
            features=1,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(c)

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
        )(c)


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
        return nn.Dense(d)(x)


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
        )(x)

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
        x = nn.Dense(self.H.d_base)(x)

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
                pool_features=p_features,
            )
        x = block(x, c)
        return nn.Dense(self.H.data_num_channels)(x)


class FlowModel(nn.Module):
    H: FlowHyperparams

    def setup(self):
        self.backbone = Backbone(self.H)

    def __call__(self, x, rng):
        time_rng, noise_rng = jax.random.split(rng, 2)
        bs, seq_len, c = x.shape
        x_1 = self.H.data_preprocess_fn(x)
        x_0 = jax.random.normal(noise_rng, shape=x_1.shape)

        t = jax.random.uniform(
            time_rng,
            shape=(bs, 1, 1),
            minval=0,
            maxval=1,
        )
        x_t = (1 - t) * x_0 + t * x_1
        dx_t = x_1 - x_0

        flow = self.backbone(x_t, t)
        loss = optax.l2_loss(flow, dx_t).mean()
        return loss, {"loss": loss}

    def sample_prior(self, seq_len, bs, rng):
        rng, noise_rng = jax.random.split(rng, 2)

        x_t = jax.random.normal(
            noise_rng,
            shape=(bs, seq_len, self.H.data_num_channels),
        )

        def flow_step(x_t, t):
            # temporarily, using midpoint ODE solver
            t_start = jnp.reshape(t, (1, 1, 1))
            t_end = t_start + 1 / self.H.sampling_steps
            dt = t_end - t_start
            return x_t + dt * self.backbone(
                x_t + self.backbone(x_t, t_start) * dt / 2, t_start + dt / 2
            ), None

        x_t, _ = lax.scan(
            flow_step,
            x_t,
            jnp.linspace(0, 1, self.H.sampling_steps, endpoint=False),
        )

        # Make x_t [-1, 1] -> [0, 1]
        x_t = x_t / 2 + 0.5
        x_t = jnp.clip(x_t, 0, 0.9999)
        # Make x_t contain ints with the class numbers (a bit ugly)
        x_t = x_t * self.H.data_num_cats
        x_t = jnp.floor(x_t).astype(jnp.int32)

        return x_t
