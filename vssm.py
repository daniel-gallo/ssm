from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from hps import Hyperparams
from rnn import RNNBlock, lecun_normal


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


class DecoderBlock(nn.Module):
    H: Hyperparams
    n_layers: int

    def setup(self):
        block = partial(RNNBlock, self.H)
        self.q_block = block(
            d_out=self.H.zdim * 2,
            bidirectional=True,
            residual=False,
            last_scale=.1,
        )
        self.p_block = block(
            d_out=self.H.zdim * 2 + self.H.rnn_out_size,
            bidirectional=False,
            residual=False,
            last_scale=0.0,
        )
        self.res_block = block(
            d_out=self.H.rnn_out_size,
            bidirectional=False,
            residual=True,
            last_scale=1.,
        )
        self.z_proj = nn.Dense(
            self.H.rnn_out_size,
        )

    def __call__(self, x, cond_enc, rng):
        q = jnp.split(
            self.q_block(jnp.concat([x, cond_enc], axis=-1)), 2, axis=-1
        )
        *p, x_p = jnp.split(
            self.p_block(x), [self.H.zdim, self.H.zdim * 2], axis=-1
        )

        z = gaussian_sample(q, rng)
        kl = gaussian_kl(q, p)

        x = self.res_block(x + x_p + self.z_proj(z))
        return x, kl

    def sample_prior(self, x, rng):
        *p, x_p = jnp.split(
            self.p_block(x), [self.H.zdim, self.H.zdim * 2], axis=-1
        )
        z = gaussian_sample(p, rng)
        return self.res_block(x + x_p + self.z_proj(z))


class Decoder(nn.Module):
    # TODO: consider models with diff. number of latent variables per block
    H: Hyperparams

    def setup(self):
        H = self.H
        self.blocks = [
            DecoderBlock(H, H.decoder_rnn_layers)
            for _ in range(H.decoder_rnn_layers)
        ]
        self.x_bias = self.param(
            "x_bias", nn.initializers.zeros, (H.rnn_out_size,)
        )
        self.final = nn.Dense(H.data_num_channels * H.data_num_cats)

    def __call__(self, cond_enc, rng):
        H = self.H
        # TODO: consider if it is useful to store sampled latents as well
        kls = []
        x = jnp.broadcast_to(
            self.x_bias, cond_enc[-1].shape[:-1] + (H.rnn_out_size,)
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
        x = jnp.broadcast_to(
            self.x_bias, (n_samples, gen_len, self.H.rnn_out_size)
        )
        for block in self.blocks:
            rng, block_rng = random.split(rng)
            x = block.sample_prior(x, block_rng)
        x = self.final(x)
        return x


class Encoder(nn.Module):
    H: Hyperparams

    def setup(self):
        self.initial = nn.Dense(self.H.rnn_out_size)
        self.blocks = [
            RNNBlock(
                H=self.H,
                d_out=self.H.rnn_out_size,
                bidirectional=True,
                residual=True,
            )
            for _ in range(self.H.encoder_rnn_layers)
        ]

    def __call__(self, x):
        x = self.initial(self.H.data_preprocess_fn(x))
        acts = []
        for block in self.blocks:
            x = block(x)
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
