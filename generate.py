from pathlib import Path

import jax.numpy as jnp
from jax import random
from jsonargparse import auto_cli
from tqdm import tqdm

from data import load_data, mu_law_decode, np_to_wav
from models.patch_autoregressive import PatchARHyperparams
from models.recurrentgemma.model import RecurrentGemmaHyperparams
from train import load_train_state

override_id = None
override_id = "b577bff7"
num_samples = 10_240
batch_size = 64
num_batches = num_samples // batch_size
assert num_samples % batch_size == 0

H = auto_cli(
    {
        "patch-ar": PatchARHyperparams,
        "recurrentgemma": RecurrentGemmaHyperparams,
    },
    as_positional=False,
)
print("Loading data...")
H, data = load_data(H)
print(H.id)
print("Loading train state...")
S = load_train_state(H, override_id)
print(S.step)


def save_samples(samples, batch_id):
    _, _, num_channels = samples.shape
    assert num_channels == 1
    samples = jnp.squeeze(samples, 2)

    sample_dir = Path(H.sample_dir) / H.id
    sample_dir.mkdir(parents=True, exist_ok=True)

    # [0, 255] -> [-2**15, 2**15 - 1]
    samples = mu_law_decode(samples)

    sample_filenames = []
    for sample_id, sample in enumerate(samples):
        sample_path = sample_dir / f"batch-{batch_id}-audio-{sample_id}.wav"
        np_to_wav(sample, sample_path, H.data_framerate)
        sample_filenames.append(str(sample_path))


rng = random.key(0)
for batch_id in tqdm(range(num_batches)):
    rng, sample_rng = random.split(rng, 2)
    samples = H.sample_fn(
        S.weights_ema, H.data_seq_length, batch_size, sample_rng
    )
    save_samples(samples, batch_id)
