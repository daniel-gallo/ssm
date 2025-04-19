from typing import Tuple

import flax.linen as nn


# Adapted from https://arxiv.org/abs/2107.03312
# TODO: test without LayerNorm and wihout the scale thing
class ResidualUnit(nn.Module):
    features: int
    dilation: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            kernel_size=7, features=self.features, kernel_dilation=self.dilation
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Conv(kernel_size=1, features=self.features)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)

        x = x * self.param("scale", nn.initializers.constant(0.01), (1,))
        return x


class EncoderBlock(nn.Module):
    features: int
    stride: int

    @nn.compact
    def __call__(self, x):
        x = x + ResidualUnit(self.features // 2, dilation=1)(x)
        x = x + ResidualUnit(self.features // 2, dilation=3)(x)
        x = x + ResidualUnit(self.features // 2, dilation=9)(x)
        x = nn.Conv(
            kernel_size=2 * self.stride,
            features=self.features,
            strides=self.stride,
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        return x


class Encoder(nn.Module):
    strides: Tuple[int, ...]

    @nn.compact
    def __call__(self, x):
        _, _, d = x.shape

        x = nn.Conv(kernel_size=7, features=d)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)

        for stride in self.strides:
            print(x.shape, "-> ", end="")
            d = 2 * d
            x = EncoderBlock(features=d, stride=stride)(x)
            print(x.shape)

        x = nn.Conv(kernel_size=3, features=d)(x)
        return x


class DecoderBlock(nn.Module):
    features: int
    stride: int

    @nn.compact
    def __call__(self, x):
        x = nn.ConvTranspose(
            kernel_size=2 * self.stride,
            features=self.features,
            strides=self.stride,
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = x + ResidualUnit(self.features, dilation=1)(x)
        x = x + ResidualUnit(self.features, dilation=3)(x)
        x = x + ResidualUnit(self.features, dilation=9)(x)
        return x


class Decoder(nn.Module):
    strides: Tuple[int, ...]

    @nn.compact
    def __call__(self, x):
        _, _, d = x.shape
        x = nn.Conv(kernel_size=7, features=d)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)

        for stride in self.strides:
            print(x.shape, "-> ", end="")
            d = d // 2
            x = DecoderBlock(features=d, stride=stride)(x)
            print(x.shape)

        x = nn.Conv(kernel_size=3, features=d)(x)
        return x
