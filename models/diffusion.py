import dataclasses
from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange, repeat
from jax import lax, random

from hps import Hyperparams
from models.rnn import RNNBlock


@dataclasses.dataclass(frozen=True)
class DiffusionHyperparams(Hyperparams):
    x_dim: int = 96
    t_dim: int = 32
    pool_factors: Tuple[int, ...] = (2, 2)
    patch_size: int = 1
    diffusion_timesteps: int = 1000

    rnn_block: str = "rglru"
    rnn_init_minval: float = 0.9
    rnn_init_maxval: float = 0.99
    rnn_norm_input: bool = True
    rnn_pos_embedding: bool = True
    rnn_hidden_size: int = 128
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


class ResBlock(nn.Module):
    H: DiffusionHyperparams

    @nn.compact
    def __call__(self, x):
        bs, seq_len, d = x.shape
        # Temporal module
        x = x + RNNBlock(H=self.H, d_out=d, bidirectional=True, residual=False)(
            nn.LayerNorm()(x)
        )

        # MLP module
        x = x + nn.Sequential(
            [
                nn.Dense(4 * d),
                nn.gelu,
                nn.Dense(d),
            ]
        )(nn.LayerNorm()(x))
        return x


class DownSample(nn.Module):
    factor: int

    @nn.compact
    def __call__(self, x):
        bs, seq_len, d = x.shape
        x = rearrange(
            x, "... (l factor) d -> ... l (factor d) ", factor=self.factor
        )
        x = nn.Dense(d)(x)
        return x


class UpSample(nn.Module):
    factor: int

    @nn.compact
    def __call__(self, x):
        bs, seq_len, d = x.shape
        x = rearrange(
            x, "... l (factor d) -> ... (l factor) d", factor=self.factor
        )
        x = nn.Dense(d)(x)
        return x


class SkipBlock(nn.Module):
    H: DiffusionHyperparams
    inner: nn.Module
    factor: int

    @nn.compact
    def __call__(self, x):
        bs, seq_len, d = x.shape
        x = ResBlock(self.H)(x)
        skip = x

        x = DownSample(self.factor)(x)
        x = self.inner(x)
        x = UpSample(self.factor)(x)

        x = nn.Dense(d)(jnp.concatenate([x, skip], axis=-1))
        x = ResBlock(self.H)(x)

        return x


class Backbone(nn.Module):
    H: DiffusionHyperparams

    def tokenize(self, x):
        return rearrange(
            x,
            "bs (new_seq_len patch) c -> bs new_seq_len (c patch)",
            patch=self.H.patch_size
        )

    def untokenize(self, x):
        return rearrange(
            x,
            "bs new_seq_len (c patch) -> bs (new_seq_len patch) c",
            patch=self.H.patch_size
        )


    @nn.compact
    def __call__(self, x_t, t):
        x_t = self.tokenize(x_t)
        bs, seq_len, d = x_t.shape

        x_t = nn.Dense(self.H.x_dim)(x_t)

        # TODO: t_embedding is being passed to every token,
        # unlike UViT for example (but they have attention, ofc)
        t_embedding = get_timestep_embedding(t, self.H.t_dim)
        t_embedding = repeat(
            t_embedding, "bs d -> bs seq_len d", seq_len=seq_len
        )

        x = jnp.concatenate([x_t, t_embedding], axis=-1)

        # Construct skipblock
        block = ResBlock(self.H)
        for factor in reversed(self.H.pool_factors):
            block = SkipBlock(self.H, block, factor)

        x = block(x)

        x = nn.Dense(d)(x)
        x = self.untokenize(x)
        x = nn.Conv(features=1, kernel_size=3, padding="SAME")(x)
        return x


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
        # Make x_t one-hot-encoded
        x_t = nn.one_hot(x_t, num_classes=self.H.data_num_cats)
        # TODO: this is a fix until the sampling process is fixed upstream
        x_t = x_t.squeeze()

        return x_t
