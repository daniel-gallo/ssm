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
        return y


class RNN(nn.Module):
    n_layers: int
    d_hidden: int
    d_out: int

    def setup(self):
        self.initial = nn.Dense(2 * d_hidden)
        self.layers = [
            RNNLayer(d_in=d_hidden, d_hidden=d_hidden, d_out=2 * d_hidden)
            for _ in range(self.n_layers)
        ]
        self.final = nn.Dense(d_out)

    def gaussian_sampling(self, x):
        mu, sigma = jnp.split(x, 2, axis=-1)
        y_shape = mu.shape

        z = random.normal(self.make_rng("rnn_gaussian_sampling"), y_shape)
        return mu + z * sigma

    def __call__(self, x):
        x = self.initial(x)
        x = nn.relu(x)
        x = self.gaussian_sampling(x)

        for layer in self.layers:
            x = layer(x)
            x = nn.relu(x)
            x = self.gaussian_sampling(x)

        x = self.final(x)
        return x


def get_model_and_state(seed):
    model = RNN(n_layers=n_layers, d_hidden=d_hidden, d_out=d_out)

    key = random.key(0)
    inp = jnp.zeros((seq_len, bs, d_in))
    state = model.init(key, inp)

    return model, state


n_layers = 3
bs = 32
seq_len = 784
d_in = 1
d_hidden = 512
d_out = 1

model, state = get_model_and_state(seed=42)
key = random.key(0)
inp = random.normal(random.key(0), (seq_len, bs, d_in))
print(inp.shape)
print(model.apply(state, inp, rngs={"params": random.key(0)}).shape)
