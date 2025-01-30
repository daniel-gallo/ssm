import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import random

from hps import Hyperparams


class RNN(nn.Module):
    H: Hyperparams
    d_hidden: int
    d_out: int
    reverse: bool = False

    @nn.compact
    def __call__(self, x):
        batch_size, _, _ = x.shape

        def stable_init(rng, shape):
            return random.uniform(
                rng,
                shape,
                minval=self.H.rnn_init_minval,
                maxval=self.H.rnn_init_maxval,
            )

        a = self.param("a", stable_init, (self.d_hidden,))

        def f(h, x):
            h = a * h + x
            return h, h

        init = jnp.zeros((batch_size, self.d_hidden))
        x = nn.Dense(self.d_hidden)(x)
        # scan assumes the sequence axis is the first one
        x = rearrange(x, "batch seq chan -> seq batch chan")
        _, h = jax.lax.scan(f, init, x, reverse=self.reverse)
        h = rearrange(h, "seq batch chan -> batch seq chan")
        return nn.Dense(self.d_out)(h)


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
        x = x_fwd + self.backward(x) if self.bidirectional else x_fwd
        return x + identity if self.use_residual else x


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
        self.final = nn.Dense(self.d_out)

    def __call__(self, x):
        x = nn.gelu(x)
        x = self.initial(x)
        for b in self.blocks:
            x = b(x)
        x = nn.gelu(x)
        return self.final(x)
