from typing import Optional

import jax
import jax.numpy as jnp
import librosa
from einops import einsum
from flax import linen as nn
from jax.scipy.signal import stft

from data import PaddedArray


def padded_log_likelihood(logits, x: PaddedArray):
    bs, seq_len, num_cats = logits.shape
    assert x.raw.squeeze().shape == (bs, seq_len)
    assert x.lengths.shape == (bs,)

    mask = (
        jnp.arange(seq_len, dtype=jnp.int32)[jnp.newaxis, :]
        < x.lengths[:, jnp.newaxis]
    )[..., jnp.newaxis].astype(jnp.float32)

    ll = jnp.sum(
        jax.nn.log_softmax(logits)
        * mask
        * nn.one_hot(x.raw.squeeze(), num_cats)
    )
    normalizer = jnp.sum(x.lengths) * jnp.log(2)

    return ll / normalizer


def log_likelihood(logits, x):
    bs, seq_len, _ = logits.shape
    x = x.squeeze()
    assert x.shape == (bs, seq_len)

    return jnp.mean(
        jnp.take_along_axis(nn.log_softmax(logits), x[..., None], -1)
    ) / jnp.log(2)


def mu_law_decode(x):
    # See https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
    # [0, 255] -> [0, 1]
    x = x.astype(jnp.float32) / 255
    x = jnp.clip(x, 0, 1)
    # [0, 1] -> [-1, 1]
    x = 2 * x - 1
    # [-1, 1] (squeezed) -> [-1, 1]
    x = jnp.sign(x) * ((1 + 255) ** jnp.abs(x) - 1) / 255
    return x


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
    linear_to_mel = jnp.array(
        librosa.filters.mel(
            sr=sample_rate,
            n_fft=frame_length,
            n_mels=num_features,
            fmin=lower_edge_hertz,
            fmax=upper_edge_hertz,
            htk=True,
            norm=None,
        )
    )

    spectrograms = einsum(
        spectrograms, linear_to_mel, "bs freq time, mel freq -> bs mel time"
    )
    spectrograms = jnp.log(spectrograms + 1e-6)
    return spectrograms


def mel_loss(reconstruction, original):
    reconstruction = wav_to_mel_spectrogram(mu_law_decode(reconstruction))
    original = wav_to_mel_spectrogram(mu_law_decode(original))

    return jnp.mean(jnp.abs(reconstruction - original))
