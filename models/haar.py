import dataclasses

import flax.linen as nn
import jax.numpy as jnp
import optax
from einops import rearrange, repeat
from jax import random

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


def gaussian_loss(mu, sigma, ground_truth):
    sigma += 0.1
    # TODO: is it okay to ignore all the other terms?
    loss = (ground_truth - mu) ** 2 / (2 * sigma**2)
    return jnp.mean(loss)


def gaussian_sample(rng, mu, sigma):
    z = random.normal(rng, mu.shape)
    return mu + sigma * z


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

    def sample(self, x, rng):
        # TODO: implement
        raise NotImplementedError


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
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
                nn.Conv(features=2, kernel_size=1),
            ]
        )(x)
        x = nn.Dense(features=2, kernel_init=nn.initializers.zeros)(x)
        mu = x[:, :, 0]
        log_sigma = x[:, :, 1]
        sigma = jnp.exp(log_sigma)
        return mu, sigma


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
        rngs = random.split(rng, self.H.n)
        losses = {}

        logits = AR(self.H)(avgs[0])
        a_i = jnp.argmax(logits, axis=-1).astype(float)
        losses["ar_loss"] = cross_entropy(logits, avgs[0])

        for i in range(self.H.n):
            mu_i, sigma_i = CNN()(a_i)
            d_i = gaussian_sample(rngs[i], mu_i, sigma_i)
            losses[f"d_{i}"] = gaussian_loss(mu_i, sigma_i, diffs[i])
            a_i = up(a_i, d_i)
            # Note that a_i <- inteleave(a_i + d_i, a_i - d_i)
            # Thus, the means are a_i (new) and sigma_i (repeated)
            losses[f"a_{i + 1}"] = gaussian_loss(
                a_i,
                repeat(sigma_i, "bs seq -> bs (seq two)", two=2),
                avgs[i + 1],
            )

        loss = sum(
            value for key, value in losses.items() if not key.startswith("x")
        )
        return loss, {"loss": loss} | losses

    def sample_prior(self, gen_len, n_samples, rng):
        # TODO: implement
        return jnp.zeros((n_samples, gen_len, self.H.data_num_channels))
