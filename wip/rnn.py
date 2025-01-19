import flax.linen as nn
import jax.numpy as jnp
from jax import random
from jax.nn.initializers import glorot_normal


def kl_gauss(mu1, mu2, logsigma1, logsigma2):
    """
    Computes KL divergence between a two sequences of one-dimentional Gaussians.
    """
    return logsigma2 - logsigma1 + (jnp.exp(logsigma1)**2 + (mu1 - mu2)**2) / (2 * jnp.exp(logsigma2)**2) - 0.5

def gaussian_sampling(mu, logsigma, rng):
    y_shape = mu.shape
    z = random.normal(rng, y_shape)
    return mu + z * jnp.exp(logsigma)


class RNNNode(nn.Module):
    d_in: int
    d_hidden: int
    d_out: int

    @nn.compact
    def __call__(self, h, x):
        def stable_init(rng, shape):
            return random.uniform(rng, shape, minval=0.999, maxval=1.001)

        a = self.param("a", stable_init, (self.d_hidden,))
        b = self.param("b", glorot_normal(), (self.d_in, self.d_hidden))
        c = self.param("c", glorot_normal(), (self.d_hidden, self.d_out))
        h = h * a + x @ b
        y = h @ c

        return h, y


class RNNLayer(nn.Module):
    d_in: int
    d_hidden: int
    d_out: int
    bidirectional: bool = False

    @nn.compact
    def __call__(self, x):
        seq_len, bs, d_in = x.shape
        layer = nn.scan(
            RNNNode,
            variable_broadcast="params",
            split_rngs={"params": False},
        )(self.d_in, self.d_hidden, self.d_out)

        h = jnp.zeros((bs, d_hidden))
        _, y = layer(h, x)

        if self.bidirectional:
            bw_layer = nn.scan(
                RNNNode,
                variable_broadcast="params",
                split_rngs={"params": False},
                reverse=True,
            )(self.d_in, self.d_hidden, self.d_out)
            y = y + bw_layer(h, x)[1]
        return y


class RNNBlock(nn.Module):
    n_layers: int
    d_hidden: int
    d_out: int
    bidirectional: bool = False
    use_residual: bool = False

    def setup(self):
        self.initial = nn.Dense(self.d_hidden)
        self.layers = [
            RNNLayer(d_in=self.d_hidden, d_hidden=self.d_hidden, d_out=self.d_hidden, bidirectional=self.bidirectional)
            for _ in range(self.n_layers)
        ]
        self.final = nn.Dense(self.d_out)
        if self.use_residual:
            self.res_proj = nn.Dense(self.d_out)

    def __call__(self, x):
        identity = x
        x = self.initial(x)
        x = nn.relu(x)

        for layer in self.layers:
            x = layer(x)
            x = nn.relu(x)

        x = self.final(x)
        x = x + self.res_proj(identity) if self.use_residual else x
        return x


class DecoderBlock(nn.Module):
    d_in: int
    d_hidden: int
    d_z: int
    d_out: int

    def setup(self):
        self.q_block = RNNBlock(n_layers=2, d_hidden=self.d_hidden, d_out=self.d_z * 2, bidirectional=True)
        self.p_block = RNNBlock(n_layers=2, d_hidden=self.d_hidden, d_out=self.d_z * 2 + self.d_in, bidirectional=False)
        self.res_block = RNNBlock(n_layers=2, d_hidden=self.d_hidden, d_out=self.d_out, use_residual=True)
        self.z_proj = nn.Dense(self.d_in)

    def __call__(self, x, cond_enc):
        q_mu, q_sig = jnp.split(self.q_block(jnp.concat([x, cond_enc], axis=-1)), 2, axis=-1)
        p_mu, p_sig, x_p = jnp.split(self.p_block(x), [self.d_z, self.d_z * 2], axis=-1)

        z = gaussian_sampling(q_mu, q_sig, self.make_rng("rnn_gaussian_sampling"))
        kl = kl_gauss(q_mu, p_mu, q_sig, p_sig)

        x = self.res_block(x + x_p + self.z_proj(z))
        return x, kl

    def sample_prior(self, x):
        # TODO: check if x_p should still be used here
        p_mu, p_sig, x_p = jnp.split(self.p_block(x), 3, axis=-1)
        z = gaussian_sampling(p_mu, p_sig, self.make_rng("rnn_gaussian_sampling"))
        return self.res_block(x + x_p + self.z_proj(z))


class Decoder(nn.Module):
    # TODO: change the way cond_enc is stored/accessed
    # CURRENT PROBLEMS:
    # 1. number of enc. blocks must be the same as dec. blocks
    # 2. each enc. activation is used instead of every n-th one (or sth even more flexible)
    #
    # TODO: consider models with diff. number of latent variables per block
    # TODO: likewise, diff. d_z and d_out per block?
    n_blocks: int
    d_hidden: int
    d_z: int
    d_out: int

    def setup(self):
        self.blocks = [DecoderBlock(
            d_in=self.d_z, d_hidden=self.d_hidden, d_z=self.d_z, d_out=self.d_z)
        for _ in range(self.n_blocks)]
        # TODO: consider stochastic prior for x initialisation (excessive stochastisity?)
        self.x_bias = self.param("x_bias", nn.initializers.zeros, (self.d_z,))
        self.final = nn.Dense(self.d_out)

    def __call__(self, cond_enc):
        # TODO: consider if it is useful to store sampled latents as well
        kl_all = []
        x = jnp.broadcast_to(self.x_bias, cond_enc[-1].shape[:-1] + (self.d_z,))
        for block_id, block in enumerate(self.blocks):
            x, kl = block(x, cond_enc[-1 - block_id])
            kl_all.append(kl)
        x = self.final(x)
        return x, kl_all

    def sample_prior(self, gen_len, n_samples):
        x = jnp.broadcast_to(self.x_bias, (gen_len, n_samples, self.d_z))
        for block in self.blocks:
            x = block.sample_prior(x)
        x = self.final(x)
        return x


def get_model_and_state(seed):
    model = Decoder(n_blocks=n_blocks, d_hidden=d_hidden, d_z=d_z, d_out=d_out)

    key = random.key(0)
    inp = jnp.zeros((seq_len, bs, d_in))
    enc_cond = random.normal(random.key(0), (seq_len, bs, d_hidden))
    state = model.init(key, 2 * [enc_cond])

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
inp = random.normal(random.key(0), (seq_len, bs, d_in))
enc_cond = random.normal(random.key(0), (seq_len, bs, d_hidden))

print(inp.shape)
out = model.apply(state, 2 * [enc_cond], rngs={"params": random.key(0)})
print("x:", out[0].shape)
print("kl:", len(out[1]))

z = model.apply(state, seq_len, bs, method=model.sample_prior, rngs={"params": random.key(0)})
print("z:", z.shape)
