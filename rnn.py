import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import random
from jax.nn.initializers import glorot_normal


class RNNLayer(nn.Module):
    d_hidden: int
    d_out: int
    # Values of A should be close to one to avoid exploding / vanishing gradients
    init_minval: float = 0.999
    init_maxval: float = 1.001

    @nn.compact
    def __call__(self, x, reverse: bool = False):
        bs, seq_len, d_in = x.shape

        def stable_init(rng, shape):
            return random.uniform(
                rng, shape, minval=self.init_minval, maxval=self.init_maxval
            )

        a = self.param("a", stable_init, (self.d_hidden,))
        b = self.param("b", glorot_normal(), (d_in, self.d_hidden))
        c = self.param("c", glorot_normal(), (self.d_hidden, self.d_out))

        def f(h, x):
            h = h * a + x @ b
            y = h @ c

            return h, y

        init = jnp.zeros((bs, self.d_hidden))
        # scan assumes the sequence axis is the first one
        x = rearrange(x, "bs seq_len d_in -> seq_len bs d_in")
        _, y = jax.lax.scan(f, init, x, reverse=reverse)
        y = rearrange(y, "seq_len bs d_out -> bs seq_len d_out")
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
            RNNLayer(d_hidden=self.d_hidden, d_out=self.d_hidden)
            for _ in range(self.n_layers)
        ]
        if self.bidirectional:
            self.backward_layers = [
                RNNLayer(d_hidden=self.d_hidden, d_out=self.d_hidden)
                for _ in range(self.n_layers)
            ]
        self.final = nn.Dense(self.d_out)
        if self.use_residual:
            self.res_proj = nn.Dense(self.d_out)

    def __call__(self, x):
        identity = x
        x = self.initial(x)
        x = nn.relu(x)

        for i in range(self.n_layers):
            if self.bidirectional:
                x = self.layers[i](x) + self.backward_layers[i](x, reverse=True)
            else:
                x = self.layers[i](x)
            x = nn.relu(x)

        x = self.final(x)
        x = x + self.res_proj(identity) if self.use_residual else x
        return x
