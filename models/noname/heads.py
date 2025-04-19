import flax.linen as nn
import jax.numpy as jnp
from jax import random


def log_likelihood(logits, x):
    bat, seq, cat = logits.shape
    x = x.squeeze()
    assert x.shape == (bat, seq)
    return jnp.mean(
        jnp.take_along_axis(nn.log_softmax(logits), x[..., None], -1)
    ) / jnp.log(2)


class ContinuousHead(nn.Module):
    num_cats: int

    def setup(self):
        self.final_layer = nn.Dense(1)

    def loss(self, reconstruction, original):
        bs, seq_len, d = reconstruction.shape
        assert original.shape == (bs, seq_len)

        reconstruction = self.final_layer(reconstruction)
        reconstruction = nn.sigmoid(reconstruction)
        reconstruction = reconstruction.squeeze()

        original = original / self.num_cats

        loss = jnp.mean(jnp.abs(reconstruction - original))
        return loss

    def __call__(self, reconstruction):
        # -> (bs, seq_len) (0, 1)
        reconstruction = self.final_layer(reconstruction)
        reconstruction = nn.sigmoid(reconstruction)
        reconstruction = reconstruction.squeeze()

        # (0, 1) -> (0, 255)
        reconstruction = reconstruction * self.num_cats
        reconstruction = reconstruction.astype(int)
        reconstruction = jnp.clip(reconstruction, 0, self.num_cats - 1)

        return reconstruction[:, :, jnp.newaxis]


class DiscreteHead(nn.Module):
    num_cats: int

    def setup(self):
        self.final_layer = nn.Dense(self.num_cats)

    def loss(self, logits, original):
        bs, seq_len, d = logits.shape
        assert original.shape == (bs, seq_len)

        logits = self.final_layer(logits)

        return -log_likelihood(logits, original)

    def __call__(self, reconstruction):
        logits = self.final_layer(reconstruction)
        return jnp.argmax(logits, axis=-1, keepdims=True)


if __name__ == "__main__":
    key = random.key(0)
    original = random.randint(key, (8, 16_000), minval=0, maxval=255)
    reconstruction = random.normal(key, (8, 16_000, 32))

    loss = DiscreteHead(num_cats=256)
    loss_params = loss.init(key, reconstruction, original)
    print(loss.apply(loss_params, reconstruction, original).shape)
    print(loss.apply(loss_params, reconstruction, method=loss.__call__).shape)
