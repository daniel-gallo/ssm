from pathlib import Path

from jax import random
from jsonargparse import auto_cli
from tqdm import tqdm

from data import load_data, mu_law_decode, np_to_wav
from log_util import logprint
from models.recurrentgemma.model import (
    RecurrentGemmaHyperparams,
)
from train import load_train_state


def main():
    override_id = None
    batch_size = 8
    num_batches = 4
    rng = random.key(0)

    H = auto_cli(
        {
            "recurrentgemma": RecurrentGemmaHyperparams,
        },
        as_positional=False,
    )
    logprint(H, "Loading data (to get correct checkpoint id)")
    H, _ = load_data(H)
    logprint(H, "Loading train state")
    S = load_train_state(H, override_id)
    assert S.step != 0

    base_dir = Path(f"./samples/{H.id}_{S.step}")
    base_dir.mkdir(parents=True, exist_ok=True)
    for batch_id in tqdm(range(num_batches)):
        rng, rng_sample = random.split(rng, 2)
        batch = H.sample_fn(
            S.weights_ema, H.data_seq_length, batch_size, rng_sample
        )

        for audio_id in range(batch_size):
            audio = mu_law_decode(batch[audio_id])
            audio = np_to_wav(
                audio, base_dir / f"{batch_id}_{audio_id}.wav", H.data_framerate
            )


if __name__ == "__main__":
    main()
