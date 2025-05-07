import flax.linen as nn
import jax.numpy as jnp

from data import PaddedArray
from models.losses import mel_loss, padded_log_likelihood


class ContinuousHead(nn.Module):
    num_cats: int

    def setup(self):
        self.final_layer = nn.Dense(1)

    def loss(self, reconstruction, original: PaddedArray):
        bs, seq_len, _ = reconstruction.shape
        original = original.raw.squeeze()
        assert original.shape == (bs, seq_len)

        reconstruction = self.final_layer(reconstruction)
        reconstruction = nn.sigmoid(reconstruction)
        reconstruction = reconstruction.squeeze()

        original = original / self.num_cats

        wave_l1 = jnp.mean(jnp.abs(reconstruction - original))
        mel_l1 = mel_loss(reconstruction, original)
        return {
            "loss": wave_l1 + mel_l1,
            "wave_l1": wave_l1,
            "mel_l1": mel_l1,
        }

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

    def loss(self, reconstruction, original: PaddedArray):
        bs, seq_len, _ = reconstruction.shape
        assert original.raw.shape[:2] == (bs, seq_len)

        logits = self.final_layer(reconstruction)

        wave_nll = -padded_log_likelihood(logits, original)
        mel_l1 = mel_loss(logits.argmax(axis=-1), original.raw.squeeze(axis=-1))
        return {
            "loss": wave_nll + mel_l1,
            "wave_nll": wave_nll,
            "mel_l1": mel_l1,
        }

    def __call__(self, reconstruction):
        logits = self.final_layer(reconstruction)
        return jnp.argmax(logits, axis=-1, keepdims=True)
