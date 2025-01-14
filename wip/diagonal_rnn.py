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

        a = self.param("a", init_fn, (self.d_hidden, ))
        b = self.param("b", init_fn, (self.d_in, self.d_hidden))
        c = self.param("c", init_fn, (self.d_hidden, self.d_out))

        # softmax(a) - temporary solution to keep recurrent from exploding
        # taken from RG_LRU/Griffin
        x = x * nn.softmax(a)[None, :] + u @ b
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

    a = nn.softmax(params["params"]["a"])
    b = params["params"]["b"]
    c = params["params"]["c"]

    def lift(u):
        # (u_1, ..., u_t) -> (A, Bu_1), ..., (A, Bu_t)
        return (jnp.repeat(a[None, ...], repeats=seq_len, axis=0), u @ b)

    @jax.vmap
    def binary_operation(e1, e2):
        M1, v1 = e1
        M2, v2 = e2
        return M1 * M2, v1 * M2[None, :] + v2

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
d_in = 128
d_hidden = 512
d_out = 128

model, state = get_model_and_state(seed=42)

key = random.key(42)
inp = random.normal(key, (seq_len, bs, d_in))

# %%
tic = time.time()
y_naive = get_layer_output_naive(model, state, inp)  # 8 seconds
y_scan = get_layer_output_scan(model, state, inp)  # 0.055 seconds
y_parallel = get_layer_output_parallel(model, state, inp)  # 4.47 seconds
tac = time.time()
print(tac - tic)

# %%
print(jnp.max(jnp.abs(y_scan - y_parallel)))
print(jnp.allclose(y_scan, y_parallel, atol=1e-5))
print(y_scan.shape)

# %%
tic = time.time()
for _ in range(10):
    y_naive = get_layer_output_naive(model, state, inp)  # 1.356 seconds
tac = time.time()
print("Naive: ", (tac - tic) / 10)

tic = time.time()
for _ in range(10):
    y_scan = get_layer_output_scan(model, state, inp)  # 0.016 seconds
tac = time.time()
print("Scan: ", (tac - tic) / 10)

tic = time.time()
for _ in range(10):
    y_parallel = get_layer_output_parallel(model, state, inp)  # 0.042 seconds
tac = time.time()
print("Parallel scan: ", (tac - tic) / 10)
