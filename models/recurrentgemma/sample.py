import dataclasses
from pathlib import Path
from typing import List

import jax.numpy as jnp
from jax import random
from jsonargparse import auto_cli
from tqdm import tqdm

from data import load_data, mu_law_decode, np_to_wav
from log_util import logprint
from models.recurrentgemma import jax as recurrentgemma
from models.recurrentgemma.model import (
    RecurrentGemmaHyperparams,
    get_griffin_config,
)
from train import load_train_state


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


def main():
    H = auto_cli(
        {
            "recurrentgemma": RecurrentGemmaHyperparams,
        },
        as_positional=False,
    )
    logprint(H, "Loading data (to get correct checkpoint id)")
    H, _ = load_data(H)
    logprint(H, "Loading train state")
    S = load_train_state(H)
    assert S.step != 0

    sampler = recurrentgemma.Sampler(
        model=recurrentgemma.Griffin(
            get_griffin_config(H),
            param_dtype=jnp.bfloat16,
        ),
        vocab=AudioVocabulary(),
        params=S.weights_ema["params"]["model"],
        deterministic_sampling=False,
    )

    batch_size = 8
    num_batches = 4
    base_dir = Path(f"./samples/{H.id}_{S.step}")
    base_dir.mkdir(parents=True, exist_ok=True)

    for batch_id in tqdm(range(num_batches)):
        sampled_output = sampler(
            input_strings=[""] * batch_size,
            total_generation_steps=16_000,
            rng=random.key(batch_id),
            return_logits=True,
        )

        for audio_id in range(batch_size):
            audio = sampled_output.tokens[audio_id]
            audio = mu_law_decode(audio)
            audio = np_to_wav(
                audio, base_dir / f"{batch_id}_{audio_id}.wav", 16_000
            )


if __name__ == "__main__":
    main()
