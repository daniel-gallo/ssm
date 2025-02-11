from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import random

from hps import Hyperparams
from rnn import RNNBlocks


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


class DownPool(nn.Module):
    # TODO: add support for padding
    H: Hyperparams

    @nn.compact
    def __call__(self, x):
        batch_size, seq_len, dim = x.shape
        x = rearrange(x, "...(l m) d -> ... l (d m)", m=self.H.pool_multiplier)
        return nn.Dense(dim * self.H.pool_expand)(x)


class UpPool(nn.Module):
    # TODO: add support for padding
    H: Hyperparams

    @nn.compact
    def __call__(self, x):
        batch_size, seq_len, dim = x.shape
        expand_ratio = int(self.H.pool_multiplier / self.H.pool_expand)
        x = nn.Dense(dim * expand_ratio)(x)
        x = rearrange(x, "... l (d m) -> ... (l m) d", m=self.H.pool_multiplier)
        return x


class DecoderBlock(nn.Module):
    H: Hyperparams
    n_layers: int
    expand_factor: int = 1

    def setup(self):
        zdim = self.H.zdim * self.expand_factor
        rnn_out_size = self.H.rnn_out_size * self.expand_factor

        blocks = partial(
            RNNBlocks, self.H, self.n_layers, expand_factor=self.expand_factor
        )
        self.q_block = blocks(
            d_out=zdim * 2,
            bidirectional=True,
            residual=False,
        )
        self.p_block = blocks(
            d_out=zdim * 2 + rnn_out_size,
            bidirectional=False,
            residual=False,
        )
        self.res_block = blocks(
            d_out=rnn_out_size,
            bidirectional=False,
            residual=True,
        )
        self.z_proj = nn.Dense(rnn_out_size)
        self.up_pool = UpPool(self.H)

    @nn.compact
    def __call__(self, x, cond_enc, rng):
        # TODO: consider different location of up-pool in the decoder block
        zdim = self.H.zdim * self.expand_factor

        q = jnp.split(
            self.q_block(jnp.concat([x, cond_enc], axis=-1)), 2, axis=-1
        )
        *p, x_p = jnp.split(self.p_block(x), [zdim, zdim * 2], axis=-1)

        z = gaussian_sample(q, rng)
        kl = gaussian_kl(q, p)

        x = self.res_block(x + x_p + self.z_proj(z))
        x = self.up_pool(x)
        return x, kl

    def sample_prior(self, x, rng):
        zdim = self.H.zdim * self.expand_factor

        *p, x_p = jnp.split(self.p_block(x), [zdim, zdim * 2], axis=-1)
        z = gaussian_sample(p, rng)
        return self.up_pool(self.res_block(x + x_p + self.z_proj(z)))


class Decoder(nn.Module):
    # TODO: consider models with diff. number of latent variables per block
    H: Hyperparams

    def setup(self):
        H = self.H
        self.init_dim = H.rnn_out_size * H.pool_expand ** len(
            H.decoder_rnn_layers
        )
        expand_factors = reversed(
            [H.pool_expand ** (i + 1) for i in range(len(H.decoder_rnn_layers))]
        )

        self.blocks = [
            DecoderBlock(H, n_layers=depth, expand_factor=expand_factor)
            for depth, expand_factor in zip(
                H.decoder_rnn_layers, expand_factors
            )
        ]
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
        gen_len = gen_len // (self.H.pool_multiplier ** len(self.blocks))
        x = jnp.broadcast_to(self.x_bias, (n_samples, gen_len, self.init_dim))
        for block in self.blocks:
            rng, block_rng = random.split(rng)
            x = block.sample_prior(x, block_rng)
        x = self.final(x)
        return x


class Encoder(nn.Module):
    H: Hyperparams

    def setup(self):
        expand_factors = [
            self.H.pool_expand ** (i + 1)
            for i in range(len(self.H.encoder_rnn_layers))
        ]

        self.initial = nn.Dense(self.H.rnn_out_size)
        self.blocks = [
            RNNBlocks(
                H=self.H,
                n_layers=depth,
                d_out=self.H.rnn_out_size * expand_factor,
                expand_factor=expand_factor,
                bidirectional=True,
                residual=True,
            )
            for depth, expand_factor in zip(
                self.H.encoder_rnn_layers, expand_factors
            )
        ]

    @nn.compact
    def __call__(self, x):
        x = self.initial(self.H.data_preprocess_fn(x))
        acts = []
        for block in self.blocks:
            x = block(x)
            x = DownPool(self.H)(x)
            acts.append(x)
        return list(reversed(acts))


class VSSM(nn.Module):
    H: Hyperparams

    def setup(self):
        self.encoder = Encoder(H=self.H)
        self.decoder = Decoder(H=self.H)

    def __call__(self, x, rng):
        logits, kls = self.decoder(self.encoder(x), rng)
        return loss_and_metrics(logits, kls, x)

    def sample_prior(self, gen_len, n_samples, rng):
        return self.decoder.sample_prior(gen_len, n_samples, rng)
