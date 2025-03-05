import dataclasses
import os
from os import path
from typing import Callable, Optional
from zlib import adler32

import optax
import tyro

from log_util import logprint

_early_logsteps = set(2**e for e in range(12))


@dataclasses.dataclass(frozen=True)
class Hyperparams:
    # Command line options
    data_dir: str = "data"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    sample_dir: str = "samples"

    encoder_rnn_layers: tuple[int, ...] = (2, 2)
    decoder_rnn_layers: tuple[int, ...] = (2, 2)

    zdim: int = 8

    pool_scale: int = 4
    pool_features: int = 2

    rnn_init_minval: float = 0.6
    rnn_init_maxval: float = 0.999
    rnn_norm_input: bool = True
    rnn_hidden_size: int = 128
    rnn_out_size: int = 16
    rnn_pos_embedding: bool = True
    rnn_block: str = "rglru"

    # temp. parameters for autoregressive
    autoregressive: bool = False
    ar_base_dim: int = 64
    ar_ff_dropout: float = 0.2
    ar_ff_expand: int = 2

    ar_n_layers: int = 4
    ar_last_scale: float = 0.25
    ar_pool: tuple[int, ...] = (4, 4)
    ar_expand: tuple[int, ...] = (2, 2)
    # ================================

    scan_implementation: str = "linear_pallas"

    dataset: str = "sc09"
    seed: int = 0
    batch_size: int = 8
    learning_rate: float = 1e-4
    grad_clip: float = 200
    skip_threshold: Optional[float] = None
    shuffle_before_epoch: bool = True

    enable_wandb: bool = False

    steps_per_print: int = 1000
    epochs_per_eval: int = 1
    epochs_per_gen: int = 50
    mins_per_checkpoint: float = 30
    num_samples_per_eval: int = 8

    num_epochs: int = 500
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
        logprint(self.log_dir, self.id, *args, **kwargs)

    def logtrain(self, step, metrics):
        if int(step) in _early_logsteps or not step % self.steps_per_print:
            self.logprint(step=step, **metrics)
        if self.enable_wandb:
            import wandb

            wandb.log(metrics, step)

    def log(self, step, metrics):
        self.logprint(step=step, **metrics)
        if self.enable_wandb:
            import wandb

            wandb.log(metrics, step)

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
                    self.decoder_rnn_layers,
                    self.zdim,
                    self.pool_scale,
                    self.pool_features,
                    self.rnn_init_minval,
                    self.rnn_init_maxval,
                    self.rnn_norm_input,
                    self.rnn_hidden_size,
                    self.rnn_out_size,
                    self.rnn_pos_embedding,
                    self.rnn_block,
                    self.scan_implementation,
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
