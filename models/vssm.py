import dataclasses
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax import random
from typing_extensions import Union

from hps import Hyperparams
from models.recurrence import RNNBlock, RNNHyperparams
from models.recurrence.common import lecun_normal


def gaussian_kl(q, p):
    q_mean, q_logstd = q
    p_mean, p_logstd = p
    return jnp.sum(
        p_logstd
        - q_logstd
        + (jnp.exp(q_logstd) ** 2 + (q_mean - p_mean) ** 2)
        / (2 * jnp.exp(p_logstd) ** 2)
        - 0.5
    )


def gaussian_sample(p, rng):
    mean, logstd = p
    return mean + jnp.exp(logstd) * random.normal(rng, mean.shape)


def log_likelihood(logits, x):
    bat, seq, chan, cat = logits.shape
    assert x.shape == (bat, seq, chan)
    return jnp.sum(
        jnp.take_along_axis(jax.nn.log_softmax(logits), x[..., None], -1)
    )


def loss_and_metrics(logits, kls, x):
    normalizer = x.size * jnp.log(2)
    ll = log_likelihood(logits, x) / normalizer
    kls = {f"kl-{idx}": k / normalizer for idx, k in enumerate(kls)}
    kl_total = sum(kls.values())
    loss = -(ll - kl_total)
    return loss, {"loss": loss, "log-like": ll, "kl-total": kl_total, **kls}


@dataclasses.dataclass(frozen=True)
class VSSMHyperparams(Hyperparams):
    rnn: RNNHyperparams = RNNHyperparams()

    encoder_rnn_layers: tuple[int, ...] = (2, 2, 2)
    decoder_rnn_layers: tuple[int, ...] = (3, 3, 3)

    zdim: int = 32
    rnn_out_size: int = 64

    pool_scale: int = 4
    pool_features: int = 2

    use_gating: bool = False

    scan_implementation: str = "linear_pallas"

    @property
    def model(self):
        return VSSM(self)

    @property
    def sample_prior(self):
        return VSSM.sample_prior


class DownPool(nn.Module):
    # TODO: add support for padding
    H: VSSMHyperparams
    pool_scale: Union[int, None] = None
    pool_features: Union[int, None] = None

    @nn.compact
    def __call__(self, x):
        batch_size, seq_len, dim = x.shape
        pool_scale = self.pool_scale or self.H.pool_scale
        pool_features = self.pool_features or self.H.pool_features
        x = rearrange(x, "...(l m) d -> ... l (d m)", m=pool_scale)
        return nn.Dense(dim * pool_features)(x)


class UpPool(nn.Module):
    # TODO: add support for padding
    H: VSSMHyperparams
    pool_scale: Union[int, None] = None
    pool_features: Union[int, None] = None

    @nn.compact
    def __call__(self, x):
        batch_size, seq_len, dim = x.shape
        pool_scale = self.pool_scale or self.H.pool_scale
        pool_features = self.pool_features or self.H.pool_features
        assert (dim * pool_scale) % pool_features == 0
        x = nn.Dense((dim * pool_scale) // pool_features)(x)
        x = rearrange(x, "... l (d m) -> ... (l m) d", m=pool_scale)
        return x


class DecoderBlock(nn.Module):
    # TODO: also adapt the rnn hidden size according to location
    H: VSSMHyperparams
    n_layers: int
    up_pool: bool = False
    location: int = 0

    def setup(self):
        zdim = self.H.zdim * (self.H.pool_features**self.location)
        out_size = self.H.rnn_out_size * (self.H.pool_features**self.location)
        block = partial(RNNBlock, self.H.rnn)
        self.q_block = block(
            d_out=zdim * 2,
            bidirectional=True,
            residual=False,
            last_scale=0.1,
        )
        self.p_block = block(
            d_out=zdim * 2 + out_size,
            bidirectional=False,
            residual=False,
            last_scale=0.1,
        )
        self.res_block = block(
            d_out=out_size,
            bidirectional=False,
            residual=True,
            last_scale=1.0 / np.sqrt(self.n_layers),
        )
        self.z_proj = nn.Dense(
            # out_size, kernel_init=lecun_normal(1 / np.sqrt(self.n_layers))
            out_size,
            kernel_init=lecun_normal(1.0),
        )
        if self.up_pool:
            self.up_pool_ = UpPool(self.H)

    def __call__(self, x, cond_enc, rng):
        zdim = self.H.zdim * (self.H.pool_features**self.location)

        q = jnp.split(
            self.q_block(jnp.concat([x, cond_enc], axis=-1)), 2, axis=-1
        )
        *p, x_p = jnp.split(self.p_block(x), [zdim, zdim * 2], axis=-1)

        z = gaussian_sample(q, rng)
        kl = gaussian_kl(q, p)

        x = self.res_block(x + x_p + self.z_proj(z))
        if self.up_pool:
            x = self.up_pool_(x)
        return x, kl

    def sample_prior(self, x, rng):
        zdim = self.H.zdim * (self.H.pool_features**self.location)

        *p, x_p = jnp.split(self.p_block(x), [zdim, zdim * 2], axis=-1)
        z = gaussian_sample(p, rng)
        x = self.res_block(x + x_p + self.z_proj(z))
        if self.up_pool:
            x = self.up_pool_(x)
        return x


class Decoder(nn.Module):
    H: VSSMHyperparams

    def setup(self):
        H = self.H
        blocks = []
        location = len(H.decoder_rnn_layers) - 1
        total_layers = sum(H.decoder_rnn_layers)
        for d in H.decoder_rnn_layers[:-1]:
            for _ in range(d - 1):
                blocks.append(
                    DecoderBlock(
                        H=H,
                        n_layers=total_layers,
                        up_pool=False,
                        location=location,
                    )
                )
            blocks.append(
                DecoderBlock(
                    H=H,
                    n_layers=total_layers,
                    up_pool=True,
                    location=location,
                )
            )
            location = location - 1
        assert location == 0
        for _ in range(H.decoder_rnn_layers[-1]):
            blocks.append(
                DecoderBlock(
                    H=H,
                    n_layers=total_layers,
                    up_pool=False,
                    location=location,
                )
            )

        self.blocks = blocks
        self.init_dim = H.rnn_out_size * H.pool_features ** (
            len(H.decoder_rnn_layers) - 1
        )
        self.x_bias = self.param(
            "x_bias", nn.initializers.zeros, (self.init_dim,)
        )
        self.final = nn.Dense(H.data_num_channels * H.data_num_cats)

    def __call__(self, cond_enc, rng):
        H = self.H
        # TODO: consider if it is useful to store sampled latents as well
        kls = []
        x = jnp.broadcast_to(
            self.x_bias, cond_enc[0].shape[:-1] + (self.init_dim,)
        )
        cond_enc = [
            c
            for d, c in zip(self.H.decoder_rnn_layers, cond_enc)
            for _ in range(d)
        ]
        for block_id, (block, acts) in enumerate(zip(self.blocks, cond_enc)):
            rng, block_rng = random.split(rng)
            x, kl = block(x, acts, block_rng)
            kls.append(kl)
        batch_size, seq_len, _ = x.shape
        x = jnp.reshape(
            self.final(x),
            (batch_size, seq_len, H.data_num_channels, H.data_num_cats),
        )
        return x, kls

    def sample_prior(self, gen_len, n_samples, rng):
        init_len = gen_len // (
            self.H.pool_scale ** (len(self.H.decoder_rnn_layers) - 1)
        )
        x = jnp.broadcast_to(self.x_bias, (n_samples, init_len, self.init_dim))
        for block in self.blocks:
            rng, block_rng = random.split(rng)
            x = block.sample_prior(x, block_rng)
        x = jnp.reshape(
            self.final(x),
            (
                n_samples,
                gen_len,
                self.H.data_num_channels,
                self.H.data_num_cats,
            ),
        )
        return random.categorical(rng, x, -1)


class Encoder(nn.Module):
    H: VSSMHyperparams

    @nn.compact
    def __call__(self, x):
        # TODO: also expand the rnn hidden size
        H = self.H
        x = nn.Dense(
            self.H.rnn_out_size, bias_init=jax.nn.initializers.normal(0.5)
        )(H.data_preprocess_fn(x))
        acts = []
        features = H.rnn_out_size
        for d in H.encoder_rnn_layers[:-1]:
            for _ in range(d):
                x = RNNBlock(
                    H=H.rnn,
                    d_out=features,
                    bidirectional=True,
                    residual=True,
                    last_scale=1.0,
                )(x)
            acts.append(x)
            features = features * H.pool_features
            x = DownPool(H)(x)
        for _ in range(H.encoder_rnn_layers[-1]):
            x = RNNBlock(
                H=H.rnn,
                d_out=features,
                bidirectional=True,
                residual=True,
                last_scale=1.0,
            )(x)
        acts.append(x)
        return list(reversed(acts))


class VSSM(nn.Module):
    H: VSSMHyperparams

    def setup(self):
        self.encoder = Encoder(H=self.H)
        self.decoder = Decoder(H=self.H)

    def __call__(self, x, rng, **kwargs):
        logits, kls = self.decoder(self.encoder(x), rng)
        return loss_and_metrics(logits, kls, x)

    def sample_prior(self, gen_len, n_samples, rng):
        return self.decoder.sample_prior(gen_len, n_samples, rng)
