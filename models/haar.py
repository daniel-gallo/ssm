import dataclasses

import flax.linen as nn
import jax.numpy as jnp
import optax
from einops import rearrange
from jax import lax, random

from hps import Hyperparams


@dataclasses.dataclass(frozen=True)
class HaarHyperparams(Hyperparams):
    n: int = 5
    ar_d: int = 32

    @property
    def model(self):
        return Haar(H=self)

    @property
    def sample_prior(self):
        return Haar.sample_prior


def down(averages):
    pairs = rearrange(averages, "bs (d two) -> bs d two", two=2)
    new_averages = (pairs[:, :, 0] + pairs[:, :, 1]) // 2
    new_details = (pairs[:, :, 1] - pairs[:, :, 0]) // 2

    return new_averages, new_details


def up(averages, details):
    assert averages.shape == details.shape
    bs, d = averages.shape

    new_averages = jnp.zeros((bs, 2 * d), dtype=averages.dtype)
    new_averages = new_averages.at[:, 0::2].set(averages - details)
    new_averages = new_averages.at[:, 1::2].set(averages + details)

    return new_averages


def cross_entropy(logits, labels):
    return jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    )


def debug(prefix, x):
    return {
        f"debug-{prefix}-norm": jnp.linalg.norm(x),
        f"debug-{prefix}-min": jnp.min(x),
        f"debug-{prefix}-max": jnp.max(x),
    }


class AR(nn.Module):
    H: HaarHyperparams

    def setup(self):
        self.bos = self.param("bos", nn.initializers.normal(), self.H.ar_d)
        self.initial = nn.Dense(features=self.H.ar_d)
        # TODO: use a more powerful model
        self.rnn = nn.RNN(nn.OptimizedLSTMCell(self.H.ar_d))
        self.cls_head = nn.Dense(features=self.H.data_num_cats)

    def shift_right(self, x):
        x = x.at[:, 1:, :].set(x[:, :-1, :])
        x = x.at[:, 0, :].set(self.bos)
        return x

    def __call__(self, x):
        x = self.H.data_preprocess_fn(x)
        x = x[:, :, jnp.newaxis]

        x = self.initial(x)
        x = self.shift_right(x)
        x = self.rnn(x)
        x = self.cls_head(x)

        return x

    def sample(self, gen_len, n_samples, rng):
        # TODO: implement
        x = jnp.zeros((n_samples, gen_len, self.H.data_num_channels), "int32")

        def fix_point(i, x):
            return random.categorical(rng, self(x), -1)

        return lax.fori_loop(0, gen_len, fix_point, x)


class CNN(nn.Module):
    H: HaarHyperparams

    @nn.compact
    def __call__(self, x):
        x = self.H.data_preprocess_fn(x)
        x = x[:, :, jnp.newaxis]

        # TODO: use a more powerful model
        x = nn.Sequential(
            [
                nn.Conv(features=2, kernel_size=3),
                nn.relu,
                nn.Conv(features=4, kernel_size=3),
                nn.relu,
                nn.Conv(features=8, kernel_size=3),
                nn.relu,
                nn.ConvTranspose(features=16, kernel_size=4, strides=2),
                nn.relu,
                nn.Conv(features=self.H.data_num_cats, kernel_size=3),
            ]
        )(x)

        return x


class Haar(nn.Module):
    H: HaarHyperparams

    def avgs_and_diffs(self, x):
        x = x.squeeze()

        avgs = [x]
        diffs = [None]
        for _ in range(self.H.n):
            a_i, d_i = down(avgs[-1])
            avgs.append(a_i)
            diffs.append(d_i)

        avgs = avgs[::-1]
        diffs = diffs[::-1]
        return avgs, diffs

    @nn.compact
    def __call__(self, x, rng):
        avgs, diffs = self.avgs_and_diffs(x)
        metrics = {}

        logits = AR(self.H)(avgs[0])
        a_i = jnp.argmax(logits, axis=-1).astype(float)
        metrics["a_0"] = cross_entropy(logits, avgs[0])

        for i in range(self.H.n):
            logits = CNN(self.H)(a_i)
            # a_i = jnp.argmax(logits, axis=-1).astype(float)
            a_i = avgs[i + 1].astype(float)
            metrics[f"a_{i + 1}"] = cross_entropy(logits, avgs[i + 1])

        loss = sum(
            value
            for key, value in metrics.items()
            if not key.startswith("debug")
        )
        return loss, {"loss": loss} | metrics

    def sample_prior(self, gen_len, n_samples, rng):
        # TODO: implement
        return jnp.zeros((n_samples, gen_len, self.H.data_num_channels))
