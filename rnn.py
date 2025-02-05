import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from jax.scipy.special import expit, logit

from hps import Hyperparams


# Want to be able to vary the scale of initialized parameters
def lecun_normal(scale):
    return nn.initializers.variance_scaling(scale, "fan_in", "truncated_normal")


def get_sinusoidal_embeddings(batch_size, seq_len, dim):
    positions = jnp.arange(seq_len)[:, None]  # Shape: (seq_len, 1)
    div_term = jnp.exp(
        jnp.arange(0, dim, 2) * (-jnp.log(10000.0) / dim)
    )  # Shape: (dim/2,)

    pe = jnp.zeros((seq_len, dim))
    pe = pe.at[:, 0::2].set(jnp.sin(positions * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(positions * div_term))

    return jnp.broadcast_to(pe, (batch_size, seq_len, dim))


class RNN(nn.Module):
    H: Hyperparams
    d_hidden: int
    d_out: int
    reverse: bool = False

    @nn.compact
    def __call__(self, x):
        batch_size, seq_len, _ = x.shape

        def stable_init(rng, shape):
            r_min, r_max = self.H.rnn_init_minval, self.H.rnn_init_maxval
            u = jax.random.uniform(
                key=rng, shape=shape, minval=r_min, maxval=r_max
            )
            return logit(u)

        a_logit = self.param("a_logit", stable_init, (self.d_hidden,))
        a = expit(a_logit)

        def f(h, x):
            h = a * h + x
            return h, h

        if self.H.rnn_pos_embedding:
            x = jnp.concatenate([x, get_sinusoidal_embeddings(batch_size, seq_len, 16)], -1)
        dx = nn.Dense(self.d_out)(x)
        init = jnp.zeros((batch_size, self.d_hidden))
        x = nn.Dense(self.d_hidden)(x)
        if self.H.rnn_norm_input:
            x = jnp.sqrt(1 - a**2) * x
        # scan assumes the sequence axis is the first one
        x = rearrange(x, "batch seq chan -> seq batch chan")
        _, h = jax.lax.scan(f, init, x, reverse=self.reverse)
        h = rearrange(h, "seq batch chan -> batch seq chan")
        return (dx + nn.Dense(self.d_out)(h)) / 2


class RNNBlock(nn.Module):
    H: Hyperparams
    d_hidden: int
    d_out: int
    bidirectional: bool = False
    use_residual: bool = False

    def setup(self):
        self.forward = RNN(self.H, d_hidden=self.d_hidden, d_out=self.d_out)
        if self.bidirectional:
            self.backward = RNN(
                self.H, d_hidden=self.d_hidden, d_out=self.d_out, reverse=True
            )

    def __call__(self, x):
        identity = x
        x = nn.gelu(x)
        x_fwd = self.forward(x)
        x = (x_fwd + self.backward(x)) / 2 if self.bidirectional else x_fwd
        return (x + identity) / 2 if self.use_residual else x


class RNNBlocks(nn.Module):
    H: Hyperparams
    n_layers: int
    d_hidden: int
    d_out: int
    bidirectional: bool = False
    use_residual: bool = False

    def setup(self):
        self.initial = nn.Dense(self.d_out)
        self.blocks = [
            RNNBlock(
                self.H,
                d_hidden=self.d_hidden,
                d_out=self.d_out,
                bidirectional=self.bidirectional,
                use_residual=self.use_residual,
            )
            for _ in range(self.n_layers)
        ]
        self.final = nn.Dense(
            self.d_out, kernel_init=lecun_normal(1 / self.n_layers)
        )

    def __call__(self, x):
        x = nn.gelu(x)
        x = self.initial(x)
        for b in self.blocks:
            x = b(x)
        x = nn.gelu(x)
        return self.final(x)
