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
from models.attention.modules import LocalAttentionBlock
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
    pool_features: tuple[int, ...] = (1, 1, 1, 1, 1, 1)
    model_structure: tuple[tuple[str, ...], ...] = (
        ("conv", "conv", "rglru", "rglru"),
        ("conv", "conv", "rglru", "rglru"),
        ("conv", "conv", "rglru", "rglru"),
        ("conv", "conv", "rglru", "rglru"),
        ("conv", "conv", "rglru", "rglru"),
        ("conv", "conv", "rglru", "rglru", "rglru", "rglru"),
        ("rglru", "rglru", "rglru", "rglru", "rglru", "rglru"),
    )
    # model_structure: tuple[tuple[str, ...], ...] = (
    #     ("conv", "conv", "rglru", "rglru"),
    #     ("conv", "conv", "rglru", "rglru"),
    #     ("conv", "conv", "rglru", "rglru"),
    #     ("conv", "conv", "rglru", "rglru", "mwa"),
    #     ("conv", "conv", "rglru", "rglru", "mwa"),
    #     ("conv", "conv", "rglru", "rglru", "mwa", "rglru", "rglru", "mwa"),
    #     ("rglru", "rglru", "rglru", "rglru", "mwa", "rglru", "rglru", "mwa"),
    # )

    unet: bool = True
    cls_head: tuple[str, ...] = ()

    use_norm: bool = True
    use_gating: bool = True
    use_temporal_cnn: bool = True
    input_transform: Literal["mlp", "sine", "embed"] = "sine"

    base_dim: int = 64
    ff_expand: int = 2
    conv_kernel_size: int = 3
    conv_feature_group_count: int = 1
    dropout_rate: float = 0.2

    mwa_heads: int = 4
    mwa_window_size: int = 64

    conv_pooling: bool = False
    segmented_sampling: bool = True

    @property
    def model(self):
        return PatchARModel(self)

    @property
    def sample_fn(self):
        def _sample_fn(weights, seq_len, num_samples, rng):
            return self.model.apply(
                weights,
                seq_len,
                num_samples,
                rng,
                method=self.model.sample_prior,
            )

        return _sample_fn


def get_block(type: str, H: PatchARHyperparams, last_layer_init_scale: float):
    match type:
        case "conv":
            return ResBlock(
                H,
                layer=ConvBlock(H),
                last_layer_init_scale=last_layer_init_scale,
            )
        case "rglru":
            return ResBlock(
                H,
                layer=TemporalMixingBlock(H),
                last_layer_init_scale=last_layer_init_scale,
            )
        case "mlp":
            return ResBlock(
                H,
                layer=MLPBlock(H),
                last_layer_init_scale=last_layer_init_scale,
            )
        case "mwa":
            return ResBlock(
                H,
                layer=AttentionBlock(H),
                last_layer_init_scale=last_layer_init_scale,
            )
        case _:
            raise ValueError(f"Unknown block type: {type}")


def get_init(H: PatchARHyperparams, scale: float):
    return nn.initializers.variance_scaling(
        scale=scale, mode="fan_in", distribution="normal"
    )


class DownPool(nn.Module):
    H: PatchARHyperparams
    factor: int
    factor_feature: int = 1

    @nn.compact
    def __call__(self, x, state=None):
        if self.H.conv_pooling:
            return nn.Conv(
                x.shape[-1] * self.factor_feature,
                self.factor,
                self.factor,
                padding="VALID",
                feature_group_count=x.shape[-1],
            )(x), None
        else:
            batch_size, seq_len, dim = x.shape
            x = rearrange(x, "...(l m) d -> ... l (m d)", m=self.factor)
            return nn.Dense(dim * self.factor_feature)(x), None

    def default_state(self, bs, dim):
        return None


class UpPool(nn.Module):
    H: PatchARHyperparams
    factor: int
    factor_feature: int
    last_layer_init_scale: float

    @nn.compact
    def __call__(self, x, state=None):
        bs, _, dim = x.shape
        state = state if state is not None else self.default_state(bs, dim)
        assert dim % self.factor_feature == 0

        if self.H.conv_pooling:
            x = nn.Conv(
                dim // self.factor_feature,
                self.factor,
                padding=self.factor - 1,
                input_dilation=self.factor,
                feature_group_count=x.shape[-1],
                kernel_init=get_init(self.H, self.last_layer_init_scale),
            )(x)
        else:
            batch_size, seq_len, dim = x.shape
            x = nn.Dense(
                dim * self.factor // self.factor_feature,
                kernel_init=get_init(self.H, self.last_layer_init_scale),
            )(x)
            x = rearrange(x, "... l (m d) -> ... (l m) d", m=self.factor)
        # ensure causal
        seq_len = x.shape[1]
        x = jnp.concatenate([state, x], axis=1)
        x, state = jnp.split(x, [seq_len], axis=1)
        return x, state

    def default_state(self, bs, dim):
        return jnp.zeros((bs, self.factor - 1, dim))


class ResBlock(nn.Module):
    H: PatchARHyperparams
    layer: nn.Module
    last_layer_init_scale: float

    @nn.compact
    def __call__(
        self, x, segment_pos, state=None, training=False, sampling=False
    ):
        _, _, d = x.shape

        skip = x

        x = nn.LayerNorm()(x) if self.H.use_norm else x

        x_branch, state = self.layer(
            x, segment_pos=segment_pos, state=state, sampling=sampling
        )
        if self.H.use_gating:
            _, _, inner_d = x_branch.shape
            gating_branch = nn.Dense(inner_d)(x)
            gating_branch = nn.gelu(gating_branch)

            x = x_branch * gating_branch
        else:
            x = nn.gelu(x_branch)
        x = nn.Dense(
            d, kernel_init=get_init(self.H, self.last_layer_init_scale)
        )(x)

        x = nn.Dropout(self.H.dropout_rate, deterministic=not training)(x)

        return skip + x, state

    def default_state(self, bs, dim):
        return self.layer.default_state(bs, dim)


class MLPBlock(nn.Module):
    H: PatchARHyperparams

    @nn.compact
    def __call__(self, x, state=None, **kwargs):
        _, _, d = x.shape
        x = nn.Dense(d * self.H.ff_expand)(x)
        return x, None

    def default_state(self, bs, dim):
        return None


class TemporalMixingBlock(nn.Module):
    H: PatchARHyperparams

    @nn.compact
    def __call__(self, x, state=None, sampling=False):
        bs, _, dim = x.shape
        recurrent_block = get_recurrent_block(self.H.rnn)
        kernel_size = self.H.conv_kernel_size
        state = state if state is not None else self.default_state(bs, dim)
        new_state = []

        if self.H.use_temporal_cnn:
            x = jnp.concatenate([state.pop(0), x], axis=1)
            new_state.append(x[:, -(kernel_size - 1) :, :])
            x = nn.LayerNorm()(
                nn.Conv(
                    dim,
                    kernel_size,
                    padding="VALID",
                    feature_group_count=self.H.conv_feature_group_count,
                )(x)
            )

        x, h_prev = recurrent_block(
            self.H,
            d_hidden=self.H.rnn_hidden_size,
            d_out=dim,
        )(x, state.pop(), sampling=sampling)
        new_state.append(h_prev)

        return x, new_state

    def default_state(self, bs, dim):
        kernel_size = self.H.conv_kernel_size

        state_rnn = jnp.zeros((bs, self.H.rnn.d_hidden))
        if self.H.use_temporal_cnn:
            state_cnn = jnp.zeros((bs, kernel_size - 1, dim))
            return [state_cnn, state_rnn]
        return [state_rnn]


class ConvBlock(nn.Module):
    H: PatchARHyperparams

    @nn.compact
    def __call__(self, x, state=None, **kwargs):
        bs, _, dim = x.shape
        state = state if state is not None else self.default_state(bs, dim)
        kernel_size = self.H.conv_kernel_size

        x = jnp.concatenate([state, x], axis=1)
        state = x[:, -(kernel_size - 1) :, :]
        x = nn.Conv(
            dim,
            kernel_size,
            padding="VALID",
            feature_group_count=self.H.conv_feature_group_count,
        )(x)

        return x, state

    def default_state(self, bs, dim):
        kernel_size = self.H.conv_kernel_size
        return jnp.zeros((bs, kernel_size - 1, dim))


class AttentionBlock(nn.Module):
    H: PatchARHyperparams

    @nn.compact
    def __call__(self, x, segment_pos, state=None, **kwargs):
        bs, seq_len, dim = x.shape
        state = state if state is not None else self.default_state(bs, dim)
        z, state = LocalAttentionBlock(
            dim, self.H.mwa_heads, self.H.mwa_window_size
        )(x, segment_pos, state)
        if self.H.use_gating:
            gated_x = nn.Dense(dim)(x)
            z = z * nn.gelu(gated_x)
        else:
            z = nn.gelu(z)
        return z, state

    def default_state(self, bs, dim):
        head_dim = dim // self.H.mwa_heads
        state = LocalAttentionBlock.default_state(
            bs, self.H.mwa_window_size, head_dim, jnp.float32
        )
        return state


class SkipBlock(nn.Module):
    H: PatchARHyperparams
    inner_layer: nn.Module
    block_structure: Tuple[str] = tuple()
    pool_temporal: int = 1
    pool_feature: int = 1

    def setup(self):
        down_blocks = []

        # MLPs and token-mixing after pooling
        num_blocks = 2 * len(self.block_structure)
        # MLPs and token-mixing before pooling
        if self.H.unet:
            num_blocks += 2 * len(self.block_structure)
        # Inner layer
        num_blocks += 1
        last_layer_init_scale = 1 / num_blocks

        if self.H.unet:
            for block_type in self.block_structure:
                down_blocks.append(
                    get_block(block_type, self.H, last_layer_init_scale)
                )
                down_blocks.append(
                    get_block("mlp", self.H, last_layer_init_scale)
                )
        self.down_blocks = down_blocks

        self.down_pool = DownPool(
            self.H,
            self.pool_temporal,
            self.pool_feature,
        )
        self.up_pool = UpPool(
            self.H,
            self.pool_temporal,
            self.pool_feature,
            last_layer_init_scale,
        )

        up_blocks = []
        for block_type in reversed(self.block_structure):
            up_blocks.append(
                get_block(block_type, self.H, last_layer_init_scale)
            )
            up_blocks.append(get_block("mlp", self.H, last_layer_init_scale))
        self.up_blocks = up_blocks

    def __call__(
        self, x, segment_pos, state=None, training=False, sampling=False
    ):
        bs, _, dim = x.shape
        state, state_inner = (
            state if state is not None else self.default_state(bs, dim)
        )
        state.reverse()
        new_state = []

        for block in self.down_blocks:
            x, h_prev = block(x, segment_pos, state.pop(), training, sampling)
            new_state.append(h_prev)

        skip = x

        x, h_prev = self.down_pool(x, state.pop())
        new_state.append(h_prev)
        x, state_inner = self.inner_layer(
            x,
            segment_pos[:, :: self.pool_temporal],
            state_inner,
            training,
            sampling,
        )
        x, h_prev = self.up_pool(x, state.pop())
        new_state.append(h_prev)

        x = x + skip

        for block in self.up_blocks:
            x, h_prev = block(x, segment_pos, state.pop(), training, sampling)
            new_state.append(h_prev)

        return x, (new_state, state_inner)

    def default_state(self, bs, dim):
        state = [block.default_state(bs, dim) for block in self.down_blocks]
        state += [
            self.down_pool.default_state(bs, dim),
            self.up_pool.default_state(bs, dim),
        ]
        state += [block.default_state(bs, dim) for block in self.up_blocks]
        return state, self.inner_layer.default_state(
            bs, dim * self.pool_feature
        )


class TemporalStack(nn.Module):
    H: PatchARHyperparams
    block_structure: Tuple[str]

    def setup(self):
        num_blocks = 2 * len(self.block_structure)
        last_layer_init_scale = 1 / num_blocks

        blocks = []
        for block_type in self.block_structure:
            blocks.append(get_block(block_type, self.H, last_layer_init_scale))
            blocks.append(get_block("mlp", self.H, last_layer_init_scale))
        self.blocks = blocks

    @nn.compact
    def __call__(
        self, x, segment_pos, state=None, training=False, sampling=False
    ):
        bs, _, dim = x.shape
        state = state if state is not None else self.default_state(bs, dim)
        state.reverse()
        new_state = []

        for block in self.blocks:
            x, h_prev = block(x, segment_pos, state.pop(), training, sampling)
            new_state.append(h_prev)

        return x, new_state

    def default_state(self, bs, dim):
        state = [block.default_state(bs, dim) for block in self.blocks]
        return state


class CLSHead(nn.Module):
    H: PatchARHyperparams

    def setup(self):
        blocks = []
        for block in self.H.cls_head:
            match block:
                case "conv":
                    blocks.append(ConvBlock(self.H))
                case "mlp":
                    blocks.append(MLPBlock(self.H, expand=self.H.ff_expand))
                case _:
                    raise ValueError(f"Unknown block {block}")
        self.blocks = blocks
        self.final = nn.Dense(self.H.data_num_cats)

    def __call__(self, x, segment_pos, state=None, training=False):
        state = (
            state
            if state is not None
            else self.default_state(x.shape[0], self.H.base_dim)
        )
        state.reverse()
        new_state = []

        for block in self.blocks:
            x, h_prev = block(x, segment_pos, state.pop(), training)
            new_state.append(h_prev)
        return self.final(x), new_state

    def default_state(self, bs, dim):
        state = [block.default_state(bs, dim) for block in self.blocks]
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
        for p_temporal, p_feature, block_structure in zip(
            reversed(self.H.pool_temporal),
            reversed(self.H.pool_features),
            reversed(self.H.model_structure[:-1]),
        ):
            block = SkipBlock(
                self.H,
                block,
                block_structure=block_structure,
                pool_temporal=p_temporal,
                pool_feature=p_feature,
            )
        self.temporal_pyramid = block

    def evaluate(
        self,
        x,
        segment_pos,
        state=None,
        inp_state=None,
        training=False,
        sampling=False,
    ):
        bs, _, channels = x.shape
        state, cls_state = (
            state if state is not None else self.default_state(bs)
        )
        inp_state = (
            inp_state
            if inp_state is not None
            else self.default_inp_state(bs, channels)
        )
        batch_size, seq_len, _ = x.shape
        x = jnp.concatenate(
            [jnp.expand_dims(inp_state, axis=1), x[:, :-1]], axis=1
        )

        match self.H.input_transform:
            case "sine":
                assert self.H.base_dim % self.H.data_num_channels == 0
                x = fourier_features(
                    x, self.H.base_dim // self.H.data_num_channels
                )
            case "embed":
                x = self.input_embed(x)
                x = rearrange(x, "... ch cat -> ... (ch cat)")
            case "mlp":
                x = self.H.data_preprocess_fn(x)
        # x = jnp.pad(x[:, :-1, :], ((0, 0), (1, 0), (0, 0)))

        if self.H.input_transform == "mlp":
            x = self.input_mlp(x)

        x, state = self.temporal_pyramid(
            x, segment_pos, state, training, sampling
        )

        x = self.norm(x)

        x, cls_state = self.cls_head(x, segment_pos, cls_state)

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
        bs, seq_len, channels = x.raw.shape
        segment_pos = jnp.arange(0, seq_len)[jnp.newaxis, :]
        segment_pos = jnp.repeat(segment_pos, bs, axis=0)
        return loss_and_metrics(
            self.H,
            self.evaluate(
                x.raw,
                segment_pos,
                self.default_state(bs),
                self.default_inp_state(bs, channels),
                training,
            )[0],
            x,
        )

    def sample_prior(self, gen_len, n_samples, rng):
        if self.H.segmented_sampling:
            segment_len = jnp.prod(jnp.array(self.H.pool_temporal))
            assert gen_len % segment_len == 0
        else:
            segment_len = gen_len  # recovers original, slow sampling
        num_segments = gen_len // segment_len

        result = jnp.zeros(
            (n_samples, num_segments, segment_len, self.H.data_num_channels),
            "int32",
        )

        def gen_segment(i, x):
            result, state, inp_state, rng = x
            loop_rng, rng = random.split(rng, 2)

            def fix_point(j, x):
                segment, state, _ = x
                prev_state = deepcopy(state)  # Possible inefficiency?

                segment_pos = (
                    jnp.arange(segment_len)[jnp.newaxis, :] + i * segment_len
                )
                segment_pos = jnp.repeat(segment_pos, n_samples, axis=0)

                segment, state = self.evaluate(
                    segment, segment_pos, state, inp_state, sampling=True
                )
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
            inp_state = segment[:, -1, :]
            return result, state, inp_state, rng

        result, _, _, _ = lax.fori_loop(
            0,
            num_segments,
            gen_segment,
            (
                result,
                self.default_state(n_samples),
                self.default_inp_state(n_samples, self.H.data_num_channels),
                rng,
            ),
        )
        result = rearrange(result, "bs seg l c -> bs (seg l) c")
        return result

    def default_state(self, bs):
        return (
            self.temporal_pyramid.default_state(bs, self.H.base_dim),
            self.cls_head.default_state(bs, self.H.base_dim),
        )

    def default_inp_state(self, bs, channels):
        return jnp.full((bs, channels), self.H.data_baseline)
