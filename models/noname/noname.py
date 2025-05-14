import dataclasses
from typing import Literal, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange

from hps import Hyperparams
from models.losses import log_likelihood
from models.noname.autoencoder import Decoder, Encoder
from models.noname.heads import ContinuousHead, DiscreteHead
from models.noname.quantizer import FSQ
from models.noname.transformer import Transformer
from models.noname.utils import get_sinusoidal_embeddings


@dataclasses.dataclass(frozen=True)
class NoNameHyperparameters(Hyperparams):
    # Autoencoder
    d: int = 32
    strides: Tuple[int, ...] = (2, 4, 5)

    # FSQ
    fsq_levels: Tuple[int, ...] = (8, 8, 8)

    # Autoregressive model
    ar_model: Literal["transformer"] = "transformer"
    ar_d: int = 384
    ar_num_layers: int = 6
    ar_num_heads: int = 6
    temperature: float = 1.0

    # Output head
    head: Literal["continuous", "discrete"] = "continuous"

    @property
    def model(self):
        return NoName(self)

    @property
    def sample_fn(self):
        def _sample_fn(weights, seq_len, num_samples, rng):
            return self.model.apply(
                weights,
                seq_len,
                num_samples,
                rng,
                method=self.model.sample_prior,
            )

        return _sample_fn


def get_head(name: str):
    match name:
        case "continuous":
            return ContinuousHead
        case "discrete":
            return DiscreteHead
        case _:
            raise ValueError


class NoName(nn.Module):
    H: NoNameHyperparameters

    def setup(self):
        self.encoder = Encoder(strides=self.H.strides)
        self.decoder = Decoder(strides=self.H.strides[::-1])

        self.fsq = FSQ(self.H.fsq_levels)
        self.enc_to_fsq = nn.Dense(self.fsq.num_dimensions)
        self.fsq_to_dec = nn.Dense(self.H.d * 2 ** len(self.H.strides))

        self.ar = Transformer(
            d=self.H.ar_d,
            num_cats=self.fsq.codebook_size,
            num_layers=self.H.ar_num_layers,
            num_heads=self.H.ar_num_heads,
        )
        self.head = get_head(self.H.head)(self.H.data_num_cats)

    def get_encoder_input(self, x: jax.Array):
        bs, seq_len = x.shape

        encoder_input = rearrange(x, "bs seq_len -> (bs seq_len)")
        encoder_input = get_sinusoidal_embeddings(encoder_input, self.H.d)
        encoder_input = rearrange(
            encoder_input, "(bs seq_len) d -> bs seq_len d", bs=bs
        )
        return encoder_input

    def get_token_utilization(self, tokens):
        sorted_tokens = jnp.sort(tokens, axis=None)
        diffs = jnp.diff(sorted_tokens)
        assert len(sorted_tokens.shape) == 1
        assert len(diffs.shape) == 1
        return (jnp.sum(diffs != 0) + 1) / self.fsq.codebook_size

    def __call__(self, x: jax.Array, rng, **kwargs):
        # Train auto-encoder
        z = self.encoder(self.get_encoder_input(x.raw.squeeze()))
        z_hat = self.fsq.quantize(self.enc_to_fsq(z))
        reconstruction = self.decoder(self.fsq_to_dec(z_hat))
        metrics = self.head.loss(reconstruction, x)

        # Train AR model
        tokens = self.fsq.codes_to_indexes(z_hat)
        logits = self.ar(tokens)
        ar_loss = -log_likelihood(logits, tokens)
        token_utilization = self.get_token_utilization(tokens)

        metrics = {
            **metrics,
            "loss": metrics["loss"] + ar_loss,
            "ar_loss": ar_loss,
            "token_utilization": token_utilization,
        }
        return metrics["loss"], metrics

    def sample_prior(self, gen_len, n_samples, rng):
        gen_len = gen_len // np.prod(self.H.strides)

        tokens = self.ar.sample(n_samples, gen_len, rng, self.H.temperature)
        z_hat = self.fsq.indexes_to_codes(tokens)

        reconstruction = self.decoder(self.fsq_to_dec(z_hat))
        sample = self.head(reconstruction)

        return sample
