import dataclasses
import os
from os import path
from typing import Callable, Optional
from zlib import adler32

import optax
import tyro

from log_util import logprint


@dataclasses.dataclass(frozen=True)
class Hyperparams:
    # Command line options
    data_dir: str = "data"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    sample_dir: str = "samples"

    # change to tuples per block
    encoder_rnn_layers: tuple[int, ...] = (2, 2)
    encoder_hidden: tuple[int, ...] = (64, 64)
    encoder_features: tuple[int, ...] = (32, 32, 32)

    decoder_enc_source: tuple[int, ...] = (1, 0)
    decoder_rnn_layers: tuple[int, ...] = (2, 2)
    decoder_hidden: tuple[int, ...] = (64, 64)
    decoder_zdim: tuple[int, ...] = (32, 32)
    decoder_features: tuple[int, ...] = (16, 16, 16)

    rnn_init_minval: float = 0.4
    rnn_init_maxval: float = 0.99
    rnn_norm_input: bool = True

    dataset: str = "binarized-mnist"
    seed: int = 0
    batch_size: int = 32
    learning_rate: float = 1e-3
    grad_clip: float = 200
    skip_threshold: float = 1000
    shuffle_before_epoch: bool = False

    enable_wandb: bool = False

    steps_per_print: int = 1000
    epochs_per_eval: int = 1
    mins_per_checkpoint: float = 30
    num_samples_per_eval: int = 8

    num_epochs: int = 1
    batch_size_eval: int = 128

    # Other useful meta-data, set automatically
    data_seq_length: Optional[int] = None
    data_num_channels: Optional[int] = None
    data_num_cats: Optional[int] = None
    data_preprocess_fn: Optional[Callable] = None

    @property
    def data_shape(self):
        return self.data_seq_length, self.data_num_channels

    @property
    def optimizer(self):
        return optax.adamw(self.learning_rate)

    def logprint(self, *args, **kwargs):
        logprint(self.log_dir, self.id, self.enable_wandb, *args, **kwargs)

    @property
    def id(self):
        # Choose an id that depends deterministically on the model hyperparams.
        # This will allow us to refer to runs (for reloading checkpoints etc.)
        # using only the hyperparams (without having to save the id).
        hash_int = adler32(
            repr(
                (
                    self.dataset,
                    self.seed,
                    self.batch_size,
                    self.learning_rate,
                    self.grad_clip,
                    self.skip_threshold,
                    self.shuffle_before_epoch,
                    self.encoder_rnn_layers,
                    self.encoder_hidden,
                    self.encoder_features,
                    self.decoder_enc_source,
                    self.decoder_rnn_layers,
                    self.decoder_hidden,
                    self.decoder_zdim,
                    self.decoder_features,
                    self.rnn_init_minval,
                    self.rnn_init_maxval,
                    self.rnn_norm_input,
                )
            ).encode("utf-8")
        )
        return f"{hash_int:08x}"

    @property
    def checkpoint_prefix(self):
        return self.id + "_"


def load_options():
    H = tyro.cli(Hyperparams)

    os.makedirs(H.log_dir, exist_ok=True)
    with open(path.join(H.log_dir, H.id + ".yaml"), "w") as f:
        f.write(tyro.to_yaml(H))

    if H.enable_wandb:
        import wandb

        wandb.init(
            config=dataclasses.asdict(H),
            name=H.id,
        )
    H.logprint("Launching run", id=H.id)
    return H
