import dataclasses
from copy import deepcopy
from typing import Literal, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import lax, random

from data import PaddedArray
from hps import Hyperparams
from models.recurrence import RNNHyperparams, get_recurrent_block


def log_likelihood(H: Hyperparams, logits, x: PaddedArray):
    bat, seq, chan, cat = logits.shape
    assert x.raw.shape == (bat, seq, chan)
    assert x.lengths.shape == (bat,)
    mask = (
        jnp.arange(seq, dtype=jnp.int32)[jnp.newaxis, :]
        < x.lengths[:, jnp.newaxis]
    )[..., jnp.newaxis, jnp.newaxis].astype(jnp.float32)
    return jnp.sum(
        jax.nn.log_softmax(logits) * mask * nn.one_hot(x.raw, H.data_num_cats)
    )


def fourier_features(x, d):
    max_period = 10_000
    half = d // 2

    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half) / half)
    args = x[..., None] * freqs
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    return rearrange(embedding, "b t c f -> b t (c f)")


def loss_and_metrics(H: Hyperparams, logits, x: PaddedArray):
    _, _, chan = x.raw.shape
    normalizer = chan * jnp.sum(x.lengths) * jnp.log(2)
    ll = log_likelihood(H, logits, x) / normalizer
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
    pool_temporal: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    model_structure: tuple[tuple[str, ...], ...] = (
        ("conv", "conv", "conv", "conv"),
        ("conv", "conv", "conv", "conv"),
        ("conv", "conv", "conv", "conv"),
        ("conv", "conv", "rglru", "rglru"),
        ("conv", "conv", "rglru", "rglru"),
        ("conv", "conv", "rglru", "rglru"),
        ("rglru", "rglru", "rglru", "rglru"),
    )
    unet: bool = True
    cls_head: tuple[str, ...] = ("conv", "conv", "mlp")

    use_norm: bool = True
    use_gating: bool = True
    use_temporal_cnn: bool = True
    skip_residual: Literal["add", "mean", "mlp"] = "add"
    input_transform: Literal["mlp", "sine", "embed"] = "sine"

    base_dim: int = 64
    rnn_hidden_size: int = 256
    ff_expand: int = 2
    conv_kernel_size: int = 3
    conv_feature_group_count: int = 1
    block_last_scale: float = 0.125
    dropout_rate: float = 0.2

    conv_pooling: bool = False

    @property
    def model(self):
        return PatchARModel(self)

    @property
    def sample_prior(self):
        return PatchARModel.sample_prior


def get_block(type: str, H: PatchARHyperparams, last_scale=1.0, expand=1):
    match type:
        case "conv":
            return ResBlock(
                H,
                layer=ConvBlock(H),
                last_scale=last_scale,
            )
        case "rglru":
            return ResBlock(
                H,
                layer=TemporalMixingBlock(H),
                last_scale=last_scale,
            )
        case "mlp":
            return ResBlock(
                H,
                layer=MLPBlock(H, expand=expand),
                last_scale=last_scale,
            )
        case _:
            raise ValueError(f"Unknown block type: {type}")


class DownPool(nn.Module):
    H: PatchARHyperparams
    factor: int

    @nn.compact
    def __call__(self, x):
        if self.H.conv_pooling:
            return nn.Conv(
                x.shape[-1],
                self.factor,
                self.factor,
                padding="VALID",
                feature_group_count=x.shape[-1],
            )(x)
        else:
            batch_size, seq_len, dim = x.shape
            x = rearrange(x, "...(l m) d -> ... l (m d)", m=self.factor)
            return nn.Dense(dim)(x)


class UpPool(nn.Module):
    H: PatchARHyperparams
    factor: int

    @nn.compact
    def __call__(self, x):
        if self.H.conv_pooling:
            x = nn.Conv(
                x.shape[-1],
                self.factor,
                padding=self.factor - 1,
                input_dilation=self.factor,
                feature_group_count=x.shape[-1],
            )(x)
        else:
            batch_size, seq_len, dim = x.shape
            x = nn.Dense(dim * self.factor)(x)
            x = rearrange(x, "... l (m d) -> ... (l m) d", m=self.factor)
        # ensure causal
        x = lax.pad(
            x,
            0.0,
            ((0, 0, 0), (self.factor - 1, -(self.factor - 1), 0), (0, 0, 0)),
        )
        return x


class ResBlock(nn.Module):
    H: PatchARHyperparams
    layer: nn.Module
    last_scale: float = 1.0

    @nn.compact
    def __call__(self, x, state=None, training=False):
        bs, seq_len, dim = x.shape

        z = nn.LayerNorm()(x) if self.H.use_norm else x
        z, state = self.layer(z, state)
        z = nn.Dropout(self.H.dropout_rate, deterministic=not training)(z)
        z = z * self.last_scale
        return x + z, state

    def default_state(self, x):
        return self.layer.default_state(x)


class MLPBlock(nn.Module):
    H: PatchARHyperparams
    expand: int | None = None
    reduce: int = 1

    @nn.compact
    def __call__(self, x, state=None):
        bs, seq_len, dim = x.shape
        expand = self.expand or self.H.ff_expand
        z = nn.Dense(dim * expand)(x)
        if self.H.use_gating:
            gated_x = nn.Dense(dim * expand)(x)
            z = z * nn.gelu(gated_x)
        else:
            z = nn.gelu(z)
        return nn.Dense(dim // self.reduce)(z), None

    def default_state(self, x):
        return None


class TemporalMixingBlock(nn.Module):
    H: PatchARHyperparams

    @nn.compact
    def __call__(self, x, state=None):
        bs, seq_len, dim = x.shape
        recurrent_block = get_recurrent_block(self.H.rnn)
        kernel_size = self.H.conv_kernel_size
        state = state if state is not None else self.default_state(x)
        new_state = []

        if self.H.use_temporal_cnn:
            z = jnp.concatenate([state.pop(0), x], axis=1)
            new_state.append(z[:, -(kernel_size - 1) :, :])
            z = nn.LayerNorm()(
                nn.Conv(
                    dim,
                    kernel_size,
                    padding="VALID",
                    feature_group_count=self.H.conv_feature_group_count,
                )(z)
            )
        else:
            z = x

        z, h_prev = recurrent_block(
            self.H.rnn,
            d_hidden=self.H.rnn_hidden_size,
            d_out=dim,
        )(z, state.pop())
        new_state.append(h_prev)

        if self.H.use_gating:
            gated_x = nn.Dense(dim)(x)
            x = z * nn.gelu(gated_x)
        else:
            x = nn.gelu(z)
        return nn.Dense(dim)(x), new_state

    def default_state(self, x):
        bs, seq_len, dim = x.shape
        kernel_size = self.H.conv_kernel_size

        state_rnn = jnp.zeros((bs, self.H.rnn.d_hidden))
        if self.H.use_temporal_cnn:
            state_cnn = jnp.zeros((bs, kernel_size - 1, self.H.base_dim))
            return [state_cnn, state_rnn]
        return [state_rnn]


class ConvBlock(nn.Module):
    H: PatchARHyperparams

    @nn.compact
    def __call__(self, x, state=None):
        bs, seq_len, dim = x.shape
        state = state if state is not None else self.default_state(x)
        kernel_size = self.H.conv_kernel_size

        z = jnp.concatenate([state, x], axis=1)
        state = z[:, -(kernel_size - 1) :, :]
        z = nn.Conv(
            dim,
            kernel_size,
            padding="VALID",
            feature_group_count=self.H.conv_feature_group_count,
        )(z)

        if self.H.use_gating:
            gated_x = nn.Dense(dim)(x)
            z = z * nn.gelu(gated_x)
        else:
            z = nn.gelu(z)
        return nn.Dense(dim)(z), state

    def default_state(self, x):
        bs, seq_len, dim = x.shape
        kernel_size = self.H.conv_kernel_size
        return jnp.zeros((bs, kernel_size - 1, self.H.base_dim))


class SkipBlock(nn.Module):
    H: PatchARHyperparams
    inner_layer: nn.Module
    block_structure: Tuple[str] = tuple()
    pool_temporal: int = 1

    def setup(self):
        down_blocks = []
        if self.H.unet:
            for block_type in self.block_structure:
                down_blocks.append(
                    get_block(block_type, self.H, self.H.block_last_scale)
                )
                down_blocks.append(
                    get_block(
                        "mlp", self.H, self.H.block_last_scale, self.H.ff_expand
                    )
                )
        self.down_blocks = down_blocks

        self.down_pool = DownPool(
            self.H,
            self.pool_temporal,
        )
        self.up_pool = UpPool(
            self.H,
            self.pool_temporal,
        )

        up_blocks = []
        for block_type in reversed(self.block_structure):
            up_blocks.append(
                get_block(block_type, self.H, self.H.block_last_scale)
            )
            up_blocks.append(
                get_block(
                    "mlp", self.H, self.H.block_last_scale, self.H.ff_expand
                )
            )
        self.up_blocks = up_blocks

    def __call__(self, x, state=None, training=False):
        state, state_inner = (
            state if state is not None else self.default_state(x)
        )
        state.reverse()
        new_state = []

        z = x

        for block in self.down_blocks:
            z, h_prev = block(z, state.pop(), training)
            new_state.append(h_prev)

        z = self.down_pool(z)
        z, state_inner = self.inner_layer(z, state_inner, training)
        z = self.up_pool(z)

        for block in self.up_blocks:
            z, h_prev = block(z, state.pop(), training)
            new_state.append(h_prev)

        match self.H.skip_residual:
            case "add":
                return x + z, (new_state, state_inner)
            case "mean":
                return (x + z) / 2, (new_state, state_inner)
            case "mlp":
                return MLPBlock(self.H, expand=1, reduce=2)(
                    jnp.concatenate([x, z], axis=-1)
                ), (new_state, state_inner)

    def default_state(self, x):
        state = [block.default_state(x) for block in self.down_blocks]
        state += [block.default_state(x) for block in self.up_blocks]
        return state, self.inner_layer.default_state(x)


class TemporalStack(nn.Module):
    H: PatchARHyperparams
    block_structure: Tuple[str] = ("rglru",)

    def setup(self):
        blocks = []
        for block_type in self.block_structure:
            blocks.append(
                get_block(block_type, self.H, self.H.block_last_scale)
            )
            blocks.append(
                get_block(
                    "mlp", self.H, self.H.block_last_scale, self.H.ff_expand
                )
            )
        self.blocks = blocks

    @nn.compact
    def __call__(self, x, state=None, training=False):
        state = state if state is not None else self.default_state(x)
        state.reverse()
        new_state = []

        for block in self.blocks:
            x, h_prev = block(x, state.pop(), training)
            new_state.append(h_prev)

        return x, new_state

    def default_state(self, x):
        state = [block.default_state(x) for block in self.blocks]
        return state


class CLSHead(nn.Module):
    H: PatchARHyperparams

    def setup(self):
        blocks = []
        for block in self.H.cls_head:
            match self.H.cls_head:
                case "conv":
                    blocks.append(ConvBlock(self.H))
                case "mlp":
                    blocks.append(MLPBlock(self.H, expand=self.H.ff_expand))
        self.blocks = blocks
        self.final = nn.Dense(self.H.data_num_cats)

    def __call__(self, x, state=None, training=False):
        state = state if state is not None else self.default_state(x)
        state.reverse()
        new_state = []

        for block in self.blocks:
            x, h_prev = block(x, state.pop(), training)
            new_state.append(h_prev)
        return self.final(x), new_state

    def default_state(self, x):
        state = [block.default_state(x) for block in self.blocks]
        return state


class PatchARModel(nn.Module):
    H: PatchARHyperparams

    def setup(self):
        match self.H.input_transform:
            case "mlp":
                self.input_mlp = nn.Dense(
                    self.H.base_dim,
                    bias_init=jax.nn.initializers.normal(0.5),
                )
            case "embed":
                self.input_embed = nn.Embed(
                    self.H.data_num_cats, self.H.base_dim
                )

        self.cls_head = CLSHead(self.H)
        self.norm = nn.LayerNorm()

        block = TemporalStack(self.H, self.H.model_structure[-1])
        for p_temporal, block_structure in zip(
            reversed(self.H.pool_temporal),
            reversed(self.H.model_structure[:-1]),
        ):
            block = SkipBlock(
                self.H,
                block,
                block_structure=block_structure,
                pool_temporal=p_temporal,
            )
        self.temporal_pyramid = block

    def evaluate(self, x, state=None, training=False):
        state, cls_state = state if state is not None else self.default_state(x)
        batch_size, seq_len, _ = x.shape
        inp = x

        match self.H.input_transform:
            case "sine":
                assert self.H.base_dim % self.H.data_num_channels == 0
                x = fourier_features(
                    inp, self.H.base_dim // self.H.data_num_channels
                )
            case "embed":
                x = self.input_embed(x)
                x = rearrange(x, "... ch cat -> ... (ch cat)")
            case "mlp":
                x = self.H.data_preprocess_fn(x)
        x = jnp.pad(x[:, :-1, :], ((0, 0), (1, 0), (0, 0)))

        if self.H.input_transform == "mlp":
            x = self.input_mlp(x)

        x, state = self.temporal_pyramid(x, state, training)

        x = self.norm(x)

        x, cls_state = self.cls_head(x, cls_state)

        return jnp.reshape(
            x,
            (
                batch_size,
                seq_len,
                self.H.data_num_channels,
                self.H.data_num_cats,
            ),
        ), (state, cls_state)

    def __call__(self, x: PaddedArray, rng=None, training=False):
        return loss_and_metrics(
            self.H,
            self.evaluate(x.raw, self.default_state(x.raw), training)[0],
            x,
        )

    def sample_prior(self, gen_len, n_samples, rng):
        # segment_len = jnp.prod(jnp.array(self.H.pool_temporal))
        segment_len = gen_len
        assert gen_len % segment_len == 0
        num_segments = gen_len // segment_len

        result = jnp.zeros(
            (n_samples, num_segments, segment_len, self.H.data_num_channels),
            "int32",
        )

        def gen_segment(i, x):
            result, state, rng = x
            loop_rng, rng = random.split(rng, 2)

            def fix_point(j, x):
                segment, state, _ = x
                prev_state = deepcopy(state)  # Possible inefficiency?
                segment, state = self.evaluate(segment, state)
                return (
                    random.categorical(loop_rng, segment, -1),
                    prev_state,
                    state,
                )

            segment = jnp.zeros(
                (n_samples, segment_len, self.H.data_num_channels), "int32"
            )
            segment, _, state = lax.fori_loop(
                0, segment_len, fix_point, (segment, state, state)
            )
            result = result.at[:, i, ...].set(segment)
            return result, state, rng

        x = jnp.zeros(
            (n_samples, segment_len, self.H.data_num_channels), "int32"
        )
        result, _, _ = lax.fori_loop(
            0, num_segments, gen_segment, (result, self.default_state(x), rng)
        )
        result = rearrange(result, "bs seg l c -> bs (seg l) c")
        return result

    def default_state(self, x):
        return (
            self.temporal_pyramid.default_state(x),
            self.cls_head.default_state(x),
        )
