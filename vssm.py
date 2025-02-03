import flax.linen as nn
import jax
import jax.numpy as jnp
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
    (bat, seq, chan, cat) = logits.shape
    assert x.shape == (bat, seq, chan)
    return jnp.sum(
        jnp.take_along_axis(jax.nn.log_softmax(logits), x[..., None], -1)
    )


def loss_and_metrics(logits, kls, x):
    size = x.size
    ll = log_likelihood(logits, x) / size
    kls = {f"kl-{idx}": k / size for idx, k in enumerate(kls)}
    kl_total = sum(kls.values())
    loss = -(ll - kl_total)
    return loss, {"loss": loss, "log-like": ll, "kl-total": kl_total, **kls}


class DecoderBlock(nn.Module):
    H: Hyperparams
    d_in: int
    n_layers: int
    d_hidden: int
    d_z: int
    d_out: int

    def setup(self):
        self.q_block = RNNBlocks(
            H=self.H,
            n_layers=self.n_layers,
            d_hidden=self.d_hidden,
            d_out=self.d_z * 2,
            bidirectional=True,
            use_residual=False,
        )
        self.p_block = RNNBlocks(
            H=self.H,
            n_layers=self.n_layers,
            d_hidden=self.d_hidden,
            d_out=self.d_z * 2 + self.d_in,
            bidirectional=False,
            use_residual=False,
        )
        self.res_block = RNNBlocks(
            H=self.H,
            n_layers=self.n_layers,
            d_hidden=self.d_hidden,
            d_out=self.d_out,
            use_residual=True,
        )
        self.z_proj = nn.Dense(self.d_in)

    def __call__(self, x, cond_enc, rng):
        q = jnp.split(
            self.q_block(jnp.concat([x, cond_enc], axis=-1)), 2, axis=-1
        )
        *p, x_p = jnp.split(self.p_block(x), [self.d_z, self.d_z * 2], axis=-1)

        z = gaussian_sample(q, rng)
        kl = gaussian_kl(q, p)

        x = self.res_block(x + x_p + self.z_proj(z))
        return x, kl

    def sample_prior(self, x, rng):
        *p, x_p = jnp.split(
            self.p_block(x), [self.d_z, self.d_z * 2], axis=-1
        )
        z = gaussian_sample(p, rng)
        return self.res_block(x + x_p + self.z_proj(z))


class Decoder(nn.Module):
    # TODO: consider models with diff. number of latent variables per block
    H: Hyperparams

    def setup(self):
        H = self.H
        self.blocks = [
            DecoderBlock(
                H=H,
                n_layers=H.decoder_rnn_layers[block],
                d_in=H.decoder_features[block],
                d_hidden=H.decoder_hidden[block],
                d_z=H.decoder_zdim[block],
                d_out=H.decoder_features[block + 1],
            )
            for block in range(len(H.decoder_rnn_layers))
        ]
        # TODO: consider stochastic prior for x initialisation (excessive stochastisity?)
        self.x_bias = self.param(
            "x_bias", nn.initializers.zeros, (H.decoder_features[0],)
        )
        self.final = nn.Dense(H.data_num_channels * H.data_num_cats)

    def __call__(self, cond_enc, rng):
        H = self.H
        # TODO: consider if it is useful to store sampled latents as well
        kl_all = []
        x = jnp.broadcast_to(
            self.x_bias, cond_enc[-1].shape[:-1] + (H.decoder_features[0],)
        )
        for block_id, block in enumerate(self.blocks):
            rng, block_rng = random.split(rng)
            x, kl = block(
                x, cond_enc[H.decoder_enc_source[block_id]], block_rng
            )
            kl_all.append(kl)
        batch_size, seq_len, _ = x.shape
        x = jnp.reshape(
            self.final(x),
            (batch_size, seq_len, H.data_num_channels, H.data_num_cats),
        )
        return x, kl_all

    def sample_prior(self, gen_len, n_samples, rng):
        x = jnp.broadcast_to(
            self.x_bias, (gen_len, n_samples, self.H.decoder_features[0])
        )
        for block in self.blocks:
            rng, block_rng = random.split(rng)
            x = block.sample_prior(x, block_rng)
        x = self.final(x)
        return x


class Encoder(nn.Module):
    H: Hyperparams

    def setup(self):
        self.initial = nn.Dense(self.H.encoder_features[0])
        self.blocks = [
            RNNBlocks(
                H=self.H,
                n_layers=self.H.encoder_rnn_layers[block],
                d_hidden=self.H.encoder_hidden[block],
                d_out=self.H.encoder_features[block + 1],
                bidirectional=True,
                use_residual=False,
            )
            for block in range(len(self.H.encoder_rnn_layers))
        ]

    def __call__(self, x):
        x = self.initial(self.H.data_preprocess_fn(x))
        cond_enc = []
        for block in self.blocks:
            x = block(x)
            cond_enc.append(x)
        return cond_enc


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
