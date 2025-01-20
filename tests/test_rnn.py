import jax.numpy as jnp
from jax import random

from rnn import RNNLayer


def test_rnn_layer():
    d_in = 3
    d_hidden = 5
    d_out = 10

    bs = 2
    seq_len = 784

    rnn_layer = RNNLayer(d_hidden=d_hidden, d_out=d_out)
    key = random.key(0)
    x = jnp.zeros((seq_len, bs, d_in))
    state = rnn_layer.init(key, x)

    y = rnn_layer.apply(state, x)
    assert y.shape == (seq_len, bs, d_out)
