import dataclasses
import itertools
from typing import List, Tuple

import jax.numpy as jnp
from flax import linen as nn

import models.recurrentgemma.jax as recurrentgemma
from data import PaddedArray
from hps import Hyperparams
from models.losses import padded_log_likelihood
from models.recurrentgemma.jax import ShardingSpec


@dataclasses.dataclass(frozen=True)
class RecurrentGemmaHyperparams(Hyperparams):
    width: int = 256
    mlp_expansion_factor: int = 3
    num_heads: int = 8
    num_blocks: int = 8
    embeddings_scale_by_sqrt_dim: bool = True
    attention_window_size: int = 256
    logits_soft_cap: float = 30.0
    pattern: Tuple[str, ...] = ("recurrent", "recurrent", "attention")

    @property
    def model(self):
        return RecurrentGemma(self)

    @property
    def sample_fn(self):
        def _sample_fn(weights, seq_len, num_samples, rng):
            griffin_config = get_griffin_config(self)
            sampler = recurrentgemma.Sampler(
                model=recurrentgemma.Griffin(
                    griffin_config, param_dtype=jnp.bfloat16
                ),
                vocab=AudioVocabulary(),
                params=weights["params"]["Griffin_0"],
                deterministic_sampling=False,
            )

            sample = sampler(
                input_strings=[""] * num_samples,
                total_generation_steps=seq_len,
                rng=rng,
                return_logits=True,
            ).tokens
            sample = jnp.stack([sample[i] for i in range(num_samples)])[
                :, :, jnp.newaxis
            ]

            return sample

        return _sample_fn


@dataclasses.dataclass
class AudioVocabulary:
    num_cats: int = 256
    _bos_id: int = 128
    _eos_id: int = 256
    _pad_id: int = 128

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id

    def pad_id(self):
        return self._pad_id

    def EncodeAsIds(self, input_string: str):
        tokens = input_string.split()
        tokens = list(map(int, tokens))
        return tokens

    def DecodeIds(self, tokens: List[int]):
        return " ".join(map(str, tokens))


def loss_and_metrics(logits, x: PaddedArray):
    ll = padded_log_likelihood(logits, x)
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


def get_griffin_config(H: RecurrentGemmaHyperparams):
    str_to_block = {
        "recurrent": recurrentgemma.TemporalBlockType.RECURRENT,
        "attention": recurrentgemma.TemporalBlockType.ATTENTION,
    }
    pattern = map(str_to_block.__getitem__, H.pattern)
    pattern = itertools.cycle(pattern)
    block_types = tuple(itertools.islice(pattern, H.num_blocks))

    return recurrentgemma.GriffinConfig(
        vocab_size=H.data_num_cats,
        width=H.width,
        mlp_expanded_width=H.width * H.mlp_expansion_factor,
        num_heads=H.num_heads,
        block_types=block_types,
        embeddings_scale_by_sqrt_dim=H.embeddings_scale_by_sqrt_dim,
        attention_window_size=H.attention_window_size,
        logits_soft_cap=H.logits_soft_cap,
    )


class RecurrentGemma(nn.Module):
    H: RecurrentGemmaHyperparams

    @nn.compact
    def __call__(self, x: PaddedArray, rng=None, training=False):
        bs, seq_len, c = x.raw.shape
        assert c == 1

        sharding_spec = ShardingSpec(
            mesh=self.H._mesh(bs),
            batch_axis_name="batch",
            sequence_axis_name="seq",
        )
        model = recurrentgemma.Griffin(
            get_griffin_config(self.H),
            param_dtype=jnp.bfloat16,
            scan_sharding_spec=sharding_spec,
        )

        # TODO: use a proper BOS
        model_input = jnp.full((bs, seq_len), 128)
        model_input = model_input.at[:, 1:].set(x.raw[:, :-1, 0])
        pos = jnp.repeat(jnp.arange(seq_len)[None], bs, axis=0)
        logits, _ = model(model_input, pos, return_cache=False)
        return loss_and_metrics(logits, x)
