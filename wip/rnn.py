# %%
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random


class RNNNode(nn.Module):
    d_in: int
    d_hidden: int
    d_out: int

    @nn.compact
    def __call__(self, x, u):
        def init_fn(rng, shape):
            # TODO: make singular values around 1
            return random.normal(rng, shape)

        a = self.param("a", init_fn, (self.d_hidden, self.d_hidden))
        b = self.param("b", init_fn, (self.d_in, self.d_hidden))
        c = self.param("c", init_fn, (self.d_hidden, self.d_out))

        x = x @ a + u @ b
        output = x @ c

        return x, output


def get_layer_output_naive(rnn_node: RNNNode, params, u: jax.Array):
    seq_len, bs, d_in = u.shape

    x = jnp.zeros((bs, rnn_node.d_hidden))
    ys = []
    for i in range(seq_len):
        x, y = rnn_node.apply(params, x, u[i])
        ys.append(y)

    return jnp.stack(ys)


def get_layer_output_scan(rnn_node: RNNNode, params, u: jax.Array) -> jax.Array:
    seq_len, bs, d_in = u.shape

    def scan_fn(x, u):
        x, y = rnn_node.apply(params, x, u)
        return x, y

    x = jnp.zeros((bs, rnn_node.d_hidden))
    _, y = jax.lax.scan(scan_fn, x, u)

    return y


def get_layer_output_parallel(rnn_node: RNNNode, params, u: jax.Array):
    seq_len, bs, d_in = u.shape

    a = params["params"]["a"]
    b = params["params"]["b"]
    c = params["params"]["c"]

    def lift(u):
        # (u_1, ..., u_t) -> (A, Bu_1), ..., (A, Bu_t)
        return (jnp.repeat(a[None, ...], repeats=seq_len, axis=0), u @ b)

    def binary_operation(e1, e2):
        M1, v1 = e1
        M2, v2 = e2

        return M1 @ M2, v1 @ M2 + v2

    _, hidden_states = jax.lax.associative_scan(binary_operation, lift(u))
    y = hidden_states @ c

    return y


def get_model_and_state(seed):
    model = RNNNode(d_in=d_in, d_hidden=d_hidden, d_out=d_out)

    key = random.key(0)
    hidden_state = jnp.zeros((bs, d_hidden))
    inp = jnp.zeros((bs, d_in))
    state = model.init(key, hidden_state, inp)

    return model, state


# %%
bs = 32
seq_len = 784
d_in = 32
d_hidden = 10
d_out = 2

model, state = get_model_and_state(seed=42)

key = random.key(42)
inp = random.normal(key, (seq_len, bs, d_in))

# %%
tic = time.time()
# y_naive = get_layer_output_naive(model, state, inp)  # 8 seconds
# y_scan = get_layer_output_scan(model, state, inp)  # 0.055 seconds
y_parallel = get_layer_output_parallel(model, state, inp)  # 4.47 seconds
tac = time.time()
print(tac - tic)
