import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.special import expit, logit

from hps import Hyperparams
from models.efficient_scan import common, pallas, scan

# TODO: probably should be passed in train.py
_mesh = jax.make_mesh((jax.device_count(),), ("batch",))
SHARDING_SPEC = pallas.ShardingSpec(mesh=_mesh)


def get_recurrent_block(H):
    match H.rnn_block.lower():
        case "rnn":
            return RNN
        case "rglru":
            return RGLRU
        case _:
            raise ValueError(f"Unknown reccurent block type: {H.rnn_block}")


def get_scan_implementation(H):
    match H.scan_implementation.lower():
        case "linear_native":
            return common.ScanType.LINEAR_NATIVE
        case "linear_pallas":
            return common.ScanType.LINEAR_PALLAS
        case "associative_native":
            return common.ScanType.ASSOCIATIVE_NATIVE
        case _:
            raise ValueError(
                f"Unknown scan implementation: {H.scan_implementation}"
            )


# Want to be able to vary the scale of initialized parameters
def lecun_normal(scale):
    return nn.initializers.variance_scaling(scale, "fan_in", "truncated_normal")


def get_sinusoidal_embeddings(batch_size, seq_len, dim):
    positions = jnp.arange(seq_len)[:, None]
    div_term = jnp.exp(jnp.arange(0, dim, 2) * (-jnp.log(10000.0) / dim))

    pe = jnp.zeros((seq_len, dim))
    pe = pe.at[:, 0::2].set(jnp.sin(positions * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(positions * div_term))

    # TODO: find less hacky alternative to dividing by 10 here:
    return jnp.broadcast_to(pe, (batch_size, seq_len, dim)) / 10


class RNN(nn.Module):
    H: Hyperparams
    d_hidden: int
    d_out: int
    reverse: bool = False

    @nn.compact
    def __call__(self, x, h_prev=None):
        batch_size, seq_len, _ = x.shape

        def stable_init(rng, shape):
            r_min, r_max = self.H.rnn_init_minval, self.H.rnn_init_maxval
            u = jax.random.uniform(
                key=rng, shape=shape, minval=r_min, maxval=r_max
            )
            return logit(u)

        a_logit = self.param("a_logit", stable_init, (self.d_hidden,))
        a = expit(a_logit)

        if self.H.rnn_pos_embedding:
            x = jnp.concatenate(
                [x, get_sinusoidal_embeddings(batch_size, seq_len, 16)], -1
            )
        dx = nn.Dense(self.d_out)(x)
        x = nn.Dense(self.d_hidden)(x)
        if self.H.rnn_norm_input:
            x = jnp.sqrt(1 - a**2) * x
        a = jnp.broadcast_to(a, x.shape)
        h, h_last = scan.linear_scan(
            x=x,
            a=a,
            h0=h_prev,
            reverse=self.reverse,
            scan_type=get_scan_implementation(self.H),
            sharding_spec=SHARDING_SPEC,
            unroll=128,
        )
        return (dx + nn.Dense(self.d_out)(h)) / 2, h_last

    def default_state(self, batch_size):
        return jnp.zeros((batch_size, self.d_hidden))


class RGLRU(nn.Module):
    H: Hyperparams
    d_hidden: int
    d_out: int
    reverse: bool = False

    @nn.compact
    def __call__(self, x, h_prev=None):
        # TODO: implement BlockDiagonalLinear from RecurrentGemma
        batch_size, seq_len, _ = x.shape

        def stable_init(rng, shape):
            r_min, r_max = self.H.rnn_init_minval, self.H.rnn_init_maxval
            u = jax.random.uniform(
                key=rng, shape=shape, minval=r_min, maxval=r_max
            )
            return logit(u)

        a_logit = self.param("a_logit", stable_init, (self.d_hidden,))
        a_expit = expit(a_logit)
        if self.H.rnn_pos_embedding:
            x = jnp.concatenate(
                [x, get_sinusoidal_embeddings(batch_size, seq_len, 16)], -1
            )
        x = nn.Dense(self.d_hidden)(x)

        gate_x = nn.sigmoid(nn.Dense(self.d_hidden)(x))
        gate_a = nn.sigmoid(nn.Dense(self.d_hidden)(x))

        a = gate_a * a_expit
        a_squared = a**2

        x = gate_x * x
        # TODO: placement of norm corresponding to RGLRU
        # reconsider doing it before gating
        if self.H.rnn_norm_input:
            x = jnp.sqrt(1 - a_squared) * x

        h, h_last = scan.linear_scan(
            x=x,
            a=a,
            h0=h_prev,
            reverse=self.reverse,
            scan_type=get_scan_implementation(self.H),
            sharding_spec=SHARDING_SPEC,
            unroll=128,
        )
        return nn.Dense(self.d_out)(h), h_last

    def default_state(self, batch_size):
        return jnp.zeros((batch_size, self.d_hidden))


class RNNBlock(nn.Module):
    H: Hyperparams
    d_out: int
    bidirectional: bool = False
    residual: bool = False
    last_scale: float = 1.0

    def setup(self):
        recurrent_block = get_recurrent_block(self.H)
        self.forward = recurrent_block(
            self.H,
            d_hidden=self.H.rnn_hidden_size,
            d_out=self.d_out,
        )
        if self.bidirectional:
            self.backward = recurrent_block(
                self.H,
                d_hidden=self.H.rnn_hidden_size,
                d_out=self.d_out,
                reverse=True,
            )
        self.last_dense = nn.Dense(self.d_out)

    def __call__(self, x):
        identity = x
        x = nn.gelu(x)
        x_fwd, _ = self.forward(x)
        x = (x_fwd + self.backward(x)[0]) / 2 if self.bidirectional else x_fwd
        x = x + identity if self.residual else x

        x = nn.gelu(x)
        x = self.last_dense(x)
        x = x + identity if self.residual else x
        return self.last_scale * x
