import flax.linen as nn
import jax.numpy as jnp
from jax import random


class Discriminator(nn.Module):
    layers: int

    @nn.compact
    def __call__(self, x):
        bs, mel, time = x.shape
        x = x[:, :, :, None]

        d = 2
        for _ in range(self.layers):
            print(x.shape)
            x = nn.Conv(kernel_size=(3, 3), features=d, padding="NONE")(x)
            print(x.shape)
            x = nn.elu(x)
            d = 2 * d
            x = nn.Conv(kernel_size=(3, 3), features=d, padding="NONE")(x)
            print(x.shape)
            print()


if __name__ == "__main__":
    x = jnp.zeros((8, 64, 33))
    key = random.key(0)

    disc = Discriminator(layers=4)
    params = disc.init(key, x)
    disc.apply(params, x)
