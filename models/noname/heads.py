from typing import Optional

import flax.linen as nn
import jax.numpy as jnp
import librosa
from einops import einsum
from jax import random
from jax.scipy.signal import stft


def log_likelihood(logits, x):
    bat, seq, cat = logits.shape
    x = x.squeeze()
    assert x.shape == (bat, seq)
    return jnp.mean(
        jnp.take_along_axis(nn.log_softmax(logits), x[..., None], -1)
    ) / jnp.log(2)


def wav_to_mel_spectrogram(
    wavs,
    frame_length: int = 1000,
    sample_rate: int = 16_000,
    num_features: int = 64,
    lower_edge_hertz: float = 80.0,
    upper_edge_hertz: Optional[float] = 7600.0,
):
    _, _, spectrograms = stft(wavs, nperseg=frame_length)
    spectrograms = jnp.abs(spectrograms)
    linear_to_mel = librosa.filters.mel(
        sr=sample_rate,
        n_fft=frame_length,
        n_mels=num_features,
        fmin=lower_edge_hertz,
        fmax=upper_edge_hertz,
        htk=True,
        norm=None,
    )

    spectrograms = einsum(
        spectrograms, linear_to_mel, "bs freq time, mel freq -> bs mel time"
    )
    spectrograms = jnp.log(spectrograms + 1e-6)
    return spectrograms


def mel_loss(reconstruction, original):
    reconstruction = wav_to_mel_spectrogram(reconstruction)
    original = wav_to_mel_spectrogram(original)

    return jnp.mean(jnp.abs(reconstruction - original))


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

    def loss(self, reconstruction, original):
        bs, seq_len, d = reconstruction.shape
        assert original.shape == (bs, seq_len)

        logits = self.final_layer(reconstruction)

        wave_nll = -log_likelihood(logits, original)
        mel_l1 = mel_loss(logits.argmax(axis=-1), original)
        return {
            "loss": wave_nll + mel_l1,
            "wave_nll": wave_nll,
            "mel_l1": mel_l1,
        }

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
