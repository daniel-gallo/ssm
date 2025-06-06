import dataclasses
from copy import deepcopy
from typing import Literal, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import lax, random
import scanagram
from scanagram import test_util

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

    norm: Literal["layer", "rms", "none"] = "layer"
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


def get_normalization(H: PatchARHyperparams):
    match H.norm:
        case "layer":
            return nn.LayerNorm()
        case "rms":
            return nn.RMSNorm()
        case "none":
            return lambda x: x
        case _:
            raise ValueError


class DownPool(nn.Module):
    H: PatchARHyperparams
    factor: int
    factor_feature: int = 1

    @nn.compact
    def __call__(self, x):
        if self.H.conv_pooling:
            return nn.Conv(
                x.shape[-1] * self.factor_feature,
                self.factor,
                self.factor,
                padding=[(self.factor - 1, 0)],
                feature_group_count=x.shape[-1],
            )(x)
        else:
            # TODO: Fix causal connection here
            raise NotImplementedError
            # batch_size, seq_len, dim = x.shape
            # x = rearrange(x, "...(l m) d -> ... l (m d)", m=self.factor)
            # return nn.Dense(dim * self.factor_feature)(x)


class UpPool(nn.Module):
    H: PatchARHyperparams
    factor: int
    factor_feature: int
    last_layer_init_scale: float

    @nn.compact
    def __call__(self, x):
        _, _, dim = x.shape
        assert dim % self.factor_feature == 0

        if self.H.conv_pooling:
            x = nn.Conv(
                dim // self.factor_feature,
                self.factor,
                padding=[(self.factor - 1, self.factor - 1)],
                input_dilation=self.factor,
                feature_group_count=x.shape[-1],
                kernel_init=get_init(self.H, self.last_layer_init_scale),
            )(x)
        else:
            # TODO: Fix causal connection here
            raise NotImplementedError
            # batch_size, seq_len, dim = x.shape
            # x = nn.Dense(
            #     dim * self.factor // self.factor_feature,
            #     kernel_init=get_init(self.H, self.last_layer_init_scale),
            # )(x)
            # x = rearrange(x, "... l (m d) -> ... (l m) d", m=self.factor)
        return x


class ResBlock(nn.Module):
    H: PatchARHyperparams
    layer: nn.Module
    last_layer_init_scale: float

    @nn.compact
    def __call__(self, x, training=False, sampling=False):
        _, _, d = x.shape

        skip = x

        x = get_normalization(self.H)(x)

        x_branch = self.layer(
            x, sampling=sampling
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

        return skip + x


class MLPBlock(nn.Module):
    H: PatchARHyperparams

    @nn.compact
    def __call__(self, x, **kwargs):
        _, _, d = x.shape
        x = nn.Dense(d * self.H.ff_expand)(x)
        return x


class TemporalMixingBlock(nn.Module):
    H: PatchARHyperparams

    @nn.compact
    def __call__(self, x, sampling=False, **kwargs):
        _, _, dim = x.shape
        recurrent_block = get_recurrent_block(self.H.rnn)
        kernel_size = self.H.conv_kernel_size

        if self.H.use_temporal_cnn:
            x = nn.LayerNorm()(
                nn.Conv(
                    dim,
                    kernel_size,
                    padding=[(kernel_size - 1, 0)],
                    feature_group_count=self.H.conv_feature_group_count,
                )(x)
            )

        x = recurrent_block(
            self.H,
            d_hidden=self.H.rnn.d_hidden,
            d_out=dim,
        )(x, sampling=sampling)

        return x


class ConvBlock(nn.Module):
    H: PatchARHyperparams

    @nn.compact
    def __call__(self, x, **kwargs):
        _, _, dim = x.shape
        kernel_size = self.H.conv_kernel_size

        x = nn.Conv(
            dim,
            kernel_size,
            padding="VALID",
            feature_group_count=self.H.conv_feature_group_count,
        )(x)

        return x


class AttentionBlock(nn.Module):
    H: PatchARHyperparams

    @nn.compact
    def __call__(self, x, **kwargs):
        bs, seq_len, dim = x.shape
        z = LocalAttentionBlock(
            dim, self.H.mwa_heads, self.H.mwa_window_size
        )(x)
        if self.H.use_gating:
            gated_x = nn.Dense(dim)(x)
            z = z * nn.gelu(gated_x)
        else:
            z = nn.gelu(z)
        return z


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

    def __call__(self, x, training=False, sampling=False):
        for block in self.down_blocks:
            x = block(x, training, sampling)

        skip = x

        x = self.down_pool(x)
        x = self.inner_layer(
            x,
            training,
            sampling,
        )
        x = self.up_pool(x)

        x = x + skip

        for block in self.up_blocks:
            x = block(x, training, sampling)

        return x


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
    def __call__(self, x, training=False, sampling=False):
        for block in self.blocks:
            x = block(x, training, sampling)
        return x


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
        self.final = nn.Dense(self.H.data_num_channels * self.H.data_num_cats)

    def __call__(self, x, training=False):
        for block in self.blocks:
            x = block(x, training)
        return self.final(x)


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

    def evaluate(self, x, training=False, sampling=False):
        batch_size, seq_len, _ = x.shape

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

        if self.H.input_transform == "mlp":
            x = self.input_mlp(x)

        x = self.temporal_pyramid(x, training, sampling)

        x = self.norm(x)

        x = self.cls_head(x)

        return rearrange(
            x,
            "... (ch cat) -> ... ch cat",
            ch=self.H.data_num_channels,
            cat=self.H.data_num_cats,
        )

    def __call__(self, x: PaddedArray, rng=None, training=False):
        x_raw = lax.pad(x.raw, 0, [[0, 0, 0], [1, -1, 0], [0, 0, 0]])
        return loss_and_metrics(
            self.H,
            self.evaluate(
                x_raw,
                training,
            ),
            x,
        )

    def sample_prior(self, gen_len, n_samples, rng):
        example_result = jnp.zeros(
            (gen_len, n_samples, self.H.data_num_channels),
            "int32",
        )
        rng = random.split(rng, gen_len)

        def full_scan(x):
            return jnp.moveaxis(
                self.evaluate(jnp.moveaxis(x, 1, 0), sampling=True),
                0, 1
            )

        body_fn, carry_init = scanagram.as_scan(full_scan, example_result)
        # test_util.check_scan(
        #     full_scan,
        #     random.randint(
        #         random.PRNGKey(0),
        #         (gen_len, n_samples, self.H.data_num_channels),
        #         0, 255
        #     )
        # )

        def gen_step(x_and_carry, rng):
            x, carry = x_and_carry
            carry, logits = body_fn(carry, x)
            x_next = random.categorical(rng, logits)
            return (x_next, carry), x_next

        _, result = lax.scan(
            gen_step,
            (jnp.zeros((n_samples, self.H.data_num_channels), "int32"),
             carry_init),
            rng
        )
        return rearrange(result, "l b c -> b l c")
