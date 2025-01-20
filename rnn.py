import flax.linen as nn
import jax.numpy as jnp
from jax import random
from jax.nn.initializers import glorot_normal


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

        h = jnp.zeros((bs, self.d_hidden))
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
            RNNLayer(
                d_in=self.d_hidden,
                d_hidden=self.d_hidden,
                d_out=self.d_hidden,
                bidirectional=self.bidirectional,
            )
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
