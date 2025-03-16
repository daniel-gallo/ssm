import dataclasses
from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange
from jax import lax, random

from hps import Hyperparams
from models.rnn import RNNBlock


@dataclasses.dataclass(frozen=True)
class DiffusionHyperparams(Hyperparams):
    d: int = 128
    pool_factors: Tuple[int, ...] = (2, 2)
    diffusion_timesteps: int = 1000

    rnn_block: str = "rglru"
    rnn_init_minval: float = 0.9
    rnn_init_maxval: float = 0.99
    rnn_norm_input: bool = True
    rnn_pos_embedding: bool = True
    rnn_hidden_size: int = 128
    rnn_n_diag_blocks: int = 1
    scan_implementation: str = "linear_pallas"

    @property
    def model(self):
        return DiffusionModel(self)

    @property
    def sample_prior(self):
        return DiffusionModel.sample_prior


def get_timestep_embedding(timesteps, d):
    max_period = 10_000
    half = d // 2

    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half) / half)
    args = timesteps[:, None] * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    return embedding


class Identity(nn.Module):
    def __call__(self, x, c):
        return x


class ConditionedLayerNorm(nn.Module):
    @nn.compact
    def __call__(self, x, c):
        scale = nn.Dense(
            features=1,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(c)[:, :, None]

        bias = nn.Dense(
            features=1,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(c)[:, :, None]

        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x = (1 + scale) * x + bias
        return x


class Gate(nn.Module):
    @nn.compact
    def __call__(self, c):
        return nn.Dense(
            features=1,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(c)[:, :, None]


class ResBlock(nn.Module):
    H: DiffusionHyperparams

    @nn.compact
    def __call__(self, x, c):
        bs, seq_len, d = x.shape
        # Temporal module
        x = x + Gate()(c) * RNNBlock(
            H=self.H, d_out=d, bidirectional=True, residual=False
        )(ConditionedLayerNorm()(x, c))

        # MLP module
        x = x + Gate()(c) * nn.Sequential(
            [
                nn.Dense(4 * d),
                nn.gelu,
                nn.Dense(d),
            ]
        )(ConditionedLayerNorm()(x, c))

        return x


class DownSample(nn.Module):
    factor: int
    d: int

    @nn.compact
    def __call__(self, x):
        x = rearrange(
            x, "... (l factor) d -> ... l (factor d) ", factor=self.factor
        )
        x = nn.Dense(self.d)(x)
        return x


class UpSample(nn.Module):
    factor: int
    d: int

    @nn.compact
    def __call__(self, x):
        x = rearrange(
            x, "... l (factor d) -> ... (l factor) d", factor=self.factor
        )
        x = nn.Dense(self.d)(x)
        return x


class SkipBlock(nn.Module):
    H: DiffusionHyperparams
    inner: nn.Module
    factor: int

    @nn.compact
    def __call__(self, x, c):
        bs, seq_len, d = x.shape

        skip = x
        x = DownSample(self.factor, self.H.d)(x)
        x = ResBlock(self.H)(x, c)

        x = self.inner(x, c)

        x = ResBlock(self.H)(x, c)
        x = UpSample(self.factor, d)(x)

        x = skip + Gate()(c) * x
        return x


class Backbone(nn.Module):
    H: DiffusionHyperparams

    @nn.compact
    def __call__(self, x, t):
        bs, seq_len, d = x.shape

        # Conditioning vector (only time at the moment)
        c = nn.Sequential(
            [nn.Dense(self.H.d), nn.gelu, nn.Dense(self.H.d), nn.gelu]
        )(get_timestep_embedding(t, self.H.d))

        # Construct SkipBlock
        block = Identity()
        for factor in reversed(self.H.pool_factors):
            block = SkipBlock(self.H, block, factor)

        x = block(x, c)
        return x


@dataclasses.dataclass
class NoiseScheduler(nn.Module):
    H: DiffusionHyperparams

    def setup(self):
        self.betas = jnp.linspace(1e-4, 0.02, self.H.diffusion_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_bar = jnp.cumprod(self.alphas)
        self.alphas_bar_previous = jnp.pad(
            self.alphas_bar[:-1], (1, 0), constant_values=1.0
        )
        self.betas_tilde = (
            self.betas * (1 - self.alphas_bar_previous) / (1 - self.alphas_bar)
        )

    def add_noise(self, x, noise, t):
        alphas_bar = jnp.take(self.alphas_bar, t)[:, None, None]
        return jnp.sqrt(alphas_bar) * x + jnp.sqrt(1 - alphas_bar) * noise

    def get_previous_sample(self, x_t, t, predicted_noise, rng):
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_bar[t]
        sigma_t = jnp.sqrt(self.betas_tilde[t])

        z = random.normal(rng, x_t.shape)  # if t > 0 else 0
        x_prev = jnp.sqrt(1 / alpha_t) * (
            x_t - (1 - alpha_t) / (jnp.sqrt(1 - alpha_bar_t)) * predicted_noise
        )

        x_prev = jnp.where(t > 0, x_prev + sigma_t * z, x_prev)
        return x_prev


class DiffusionModel(nn.Module):
    H: DiffusionHyperparams

    def setup(self):
        self.noise_scheduler = NoiseScheduler(self.H)
        self.backbone = Backbone(self.H)

    def __call__(self, x, rng):
        time_rng, noise_rng = random.split(rng, 2)

        # (bs, seq_len, num_classes), in the range [-1, 1]
        bs, seq_len, num_channels = x.shape
        x_0 = self.H.data_preprocess_fn(x)

        t = random.randint(
            time_rng, shape=bs, minval=0, maxval=self.H.diffusion_timesteps
        )
        noise = random.normal(noise_rng, shape=x_0.shape)
        x_t = self.noise_scheduler.add_noise(x_0, noise, t)

        noise_prediction = self.backbone(x_t, t)
        loss = jnp.mean(jnp.square(noise - noise_prediction))

        return loss, {"loss": loss}

    def sample_prior(self, seq_len, bs, rng):
        rng, subrng = random.split(rng, 2)

        x_t = random.normal(
            subrng, shape=(bs, seq_len, self.H.data_num_channels)
        )

        def step_fn(carry, t):
            rng, x_t = carry
            rng, subrng = random.split(rng)
            predicted_noise = self.backbone(x_t, t * jnp.ones(bs))
            x_t = self.noise_scheduler.get_previous_sample(
                x_t, t, predicted_noise, subrng
            )
            return (rng, x_t), None

        (rng, x_t), _ = lax.scan(
            step_fn,
            (rng, x_t),
            jnp.arange(self.H.diffusion_timesteps - 1, -1, -1),
        )

        # Make x_t [-1, 1] -> [0, 1]
        x_t = x_t / 2 + 0.5
        x_t = jnp.clip(x_t, 0, 0.9999)
        # Make x_t contain ints with the class numbers (a bit ugly)
        x_t = x_t * self.H.data_num_cats
        x_t = jnp.floor(x_t).astype(jnp.int32)

        return x_t
