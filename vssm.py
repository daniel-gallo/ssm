import flax.linen as nn
import jax.numpy as jnp
from jax import random

from rnn import RNNBlock
from hps import Hyperparams


def kl_gauss(mu1, mu2, logsigma1, logsigma2):
    """
    Computes KL divergence between a two sequences of one-dimentional Gaussians.
    """
    return (
        logsigma2
        - logsigma1
        + (jnp.exp(logsigma1) ** 2 + (mu1 - mu2) ** 2)
        / (2 * jnp.exp(logsigma2) ** 2)
        - 0.5
    )


def gaussian_sampling(mu, logsigma, rng):
    y_shape = mu.shape
    z = random.normal(rng, y_shape)
    return mu + z * jnp.exp(logsigma)


class DecoderBlock(nn.Module):
    H: Hyperparams
    d_in: int
    n_layers: int
    d_hidden: int
    d_z: int
    d_out: int

    def setup(self):
        self.q_block = RNNBlock(
            H=self.H,
            n_layers=self.n_layers,
            d_hidden=self.d_hidden,
            d_out=self.d_z * 2,
            bidirectional=True,
        )
        self.p_block = RNNBlock(
            H=self.H,
            n_layers=self.n_layers,
            d_hidden=self.d_hidden,
            d_out=self.d_z * 2 + self.d_in,
            bidirectional=False,
        )
        self.res_block = RNNBlock(
            H=self.H,
            n_layers=self.n_layers,
            d_hidden=self.d_hidden,
            d_out=self.d_out,
            use_residual=True,
        )
        self.z_proj = nn.Dense(self.d_in)

    def __call__(self, x, cond_enc, rng):
        q_mu, q_sig = jnp.split(
            self.q_block(jnp.concat([x, cond_enc], axis=-1)), 2, axis=-1
        )
        p_mu, p_sig, x_p = jnp.split(
            self.p_block(x), [self.d_z, self.d_z * 2], axis=-1
        )

        z = gaussian_sampling(q_mu, q_sig, rng)
        kl = kl_gauss(q_mu, p_mu, q_sig, p_sig)

        x = self.res_block(x + x_p + self.z_proj(z))
        return x, kl

    def sample_prior(self, x, rng):
        p_mu, p_sig, x_p = jnp.split(
            self.p_block(x), [self.d_z, self.d_z * 2], axis=-1
        )
        z = gaussian_sampling(p_mu, p_sig, rng)
        return self.res_block(x + x_p + self.z_proj(z))


class Decoder(nn.Module):
    # TODO: change the way cond_enc is stored/accessed
    # CURRENT PROBLEMS:
    # 1. number of enc. blocks must be the same as dec. blocks
    # 2. each enc. activation is used instead of every n-th one (or sth even more flexible)
    #
    # TODO: consider models with diff. number of latent variables per block
    # TODO: likewise, diff. d_z and d_out per block?
    H: Hyperparams

    def setup(self):
        self.blocks = [
            DecoderBlock(
                H=self.H,
                n_layers=self.H.decoder_rnn_layers[block],
                d_in=self.H.decoder_features[block],
                d_hidden=self.H.decoder_hidden[block],
                d_z=self.H.decoder_zdim[block],
                d_out=self.H.decoder_features[block + 1],
            )
            for block in range(len(self.H.decoder_rnn_layers))
        ]
        # TODO: consider stochastic prior for x initialisation (excessive stochastisity?)
        self.x_bias = self.param(
            "x_bias", nn.initializers.zeros, (self.H.decoder_features[0],)
        )
        self.final = nn.Dense(self.H.decoder_dout)

    def __call__(self, cond_enc, rng):
        # TODO: consider if it is useful to store sampled latents as well
        kl_all = []
        x = jnp.broadcast_to(
            self.x_bias, cond_enc[-1].shape[:-1] + (self.H.decoder_features[0],)
        )
        for block_id, block in enumerate(self.blocks):
            rng, block_rng = random.split(rng)
            x, kl = block(
                x, cond_enc[self.H.decoder_enc_source[block_id]], block_rng
            )
            kl_all.append(kl)
        x = self.final(x)
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
            RNNBlock(
                H=self.H,
                n_layers=self.H.encoder_rnn_layers[block],
                d_hidden=self.H.encoder_hidden[block],
                d_out=self.H.encoder_features[block + 1],
                bidirectional=True,
            )
            for block in range(len(self.H.encoder_rnn_layers))
        ]

    def __call__(self, x):
        x = self.initial(x)
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
        cond_enc = self.encoder(x)
        x, kl = self.decoder(cond_enc, rng)
        return x, kl

    def sample_prior(self, gen_len, n_samples, rng):
        return self.decoder.sample_prior(gen_len, n_samples, rng)


def get_model_and_state(seed):
    H = Hyperparams()
    model = VSSM(H=H)

    key = random.key(seed)
    inp = jnp.zeros((bs, seq_len, d_in))
    state = model.init(key, inp, key)

    return model, state


n_layers = 3
n_blocks = 2
bs = 32
seq_len = 784
d_in = 2
d_hidden = 512
d_z = 32
d_out = 2

model, state = get_model_and_state(seed=42)
key = random.key(0)
inp = random.normal(random.key(0), (bs, seq_len, d_in))
enc_cond = random.normal(random.key(0), (bs, seq_len, d_hidden))

print(inp.shape)
out = model.apply(state, inp, key)
print("x:", out[0].shape)
print("kl:", len(out[1]))

z = model.apply(state, bs, seq_len, key, method=model.sample_prior)
print("z:", z.shape)
