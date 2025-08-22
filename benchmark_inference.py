from pathlib import Path
import timeit

import jax.numpy as jnp
from jax import random
from jax import jit
from jsonargparse import auto_cli
from tqdm import tqdm

from data import load_data, mu_law_decode, np_to_wav
from models.patch_autoregressive import PatchARHyperparams
from models.recurrentgemma.model import RecurrentGemmaHyperparams
from train import load_train_state

override_id = None
override_id = "b577bff7"
num_samples = 10_240
batch_size = 32
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
print("Loading train state...")
S = load_train_state(H, override_id)


rng = random.key(0)
#for batch_id in tqdm(range(num_batches)):

sample_fn_compiled = jit(H.sample_fn, static_argnums=(1,2))
# Ensure compiled
sample_fn_compiled(
    S.weights_ema, H.data_seq_length, batch_size, rng
).block_until_ready()

# Throughput in toks/s
print(
    (batch_size * H.data_seq_length)
    / timeit.timeit(lambda: sample_fn_compiled(
        S.weights_ema, H.data_seq_length, batch_size, rng
    ).block_until_ready(), number=1)
)
