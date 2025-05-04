import itertools

import jax
import jax.numpy as jnp
from flax import linen as nn

import models.recurrentgemma.jax as recurrentgemma
from data import PaddedArray
from hps import Hyperparams


class RecurrentGemmaHyperparams(Hyperparams):
    width: int = 256
    mlp_expanded_width: int = 3 * 256
    num_heads: int = 8
    num_blocks: int = 8
    embeddings_scale_by_sqrt_dim: bool = True
    attention_window_size: int = 256
    logits_soft_cap: float = 30.0

    @property
    def model(self):
        return RecurrentGemma(self)

    @property
    def sample_prior(self):
        return RecurrentGemma.sample_prior


def log_likelihood(logits, x: PaddedArray):
    bat, seq, cat = logits.shape
    assert x.raw.squeeze().shape == (bat, seq)
    assert x.lengths.shape == (bat,)
    mask = (
        jnp.arange(seq, dtype=jnp.int32)[jnp.newaxis, :]
        < x.lengths[:, jnp.newaxis]
    )[..., jnp.newaxis].astype(jnp.float32)
    return jnp.sum(
        jax.nn.log_softmax(logits) * mask * nn.one_hot(x.raw.squeeze(), cat)
    )


def loss_and_metrics(logits, x: PaddedArray):
    _, _, chan = x.raw.shape
    assert chan == 1
    normalizer = chan * jnp.sum(x.lengths) * jnp.log(2)
    ll = log_likelihood(logits, x) / normalizer
    loss = -ll
    return loss, {
        "loss": loss,
        "log-like": ll,
        "mean_0": jnp.mean(logits[:, 0]),
        "max_0": jnp.max(logits[:, 0]),
        "min_0": jnp.min(logits[:, 0]),
        "mean_l": jnp.mean(logits[:, -1]),
        "max_l": jnp.max(logits[:, -1]),
        "min_l": jnp.min(logits[:, -1]),
    }


class RecurrentGemma(nn.Module):
    H: RecurrentGemmaHyperparams

    def setup(self):
        griffin_pattern = itertools.cycle(
            [
                recurrentgemma.TemporalBlockType.RECURRENT,
                recurrentgemma.TemporalBlockType.RECURRENT,
                recurrentgemma.TemporalBlockType.ATTENTION,
            ]
        )
        model_config = recurrentgemma.GriffinConfig(
            vocab_size=self.H.data_num_cats,
            width=self.H.width,
            mlp_expanded_width=self.H.mlp_expanded_width,
            num_heads=self.H.num_heads,
            block_types=tuple(
                itertools.islice(griffin_pattern, self.H.num_blocks)
            ),
            embeddings_scale_by_sqrt_dim=self.H.embeddings_scale_by_sqrt_dim,
            attention_window_size=self.H.attention_window_size,
            logits_soft_cap=self.H.logits_soft_cap,
            # TODO: Linear Pallas seems to interfere with the manual sharding that we do
            scan_type=recurrentgemma.ScanType.LINEAR_NATIVE,
        )
        self.model = recurrentgemma.Griffin(
            model_config, param_dtype=jnp.bfloat16
        )

    def __call__(self, x: PaddedArray, rng=None, training=False):
        bs, seq_len, c = x.raw.shape
        assert c == 1

        # TODO: use a proper BOS
        model_input = jnp.full((bs, seq_len), 128)
        model_input = model_input.at[:, 1:].set(x.raw[:, :-1, 0])
        pos = jnp.repeat(jnp.arange(seq_len)[None], bs, axis=0)
        logits, _ = self.model(model_input, pos, return_cache=False)
        return loss_and_metrics(logits, x)

    def sample_prior(self, gen_len, n_samples, rng):
        raise NotImplementedError
