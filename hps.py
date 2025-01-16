import dataclasses
from typing import Optional
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
    dataset: str = "binarized-mnist"
    seed: int = 0
    batch_size: int = 32
    learning_rate: float = 1e-3
    enable_wandb: bool = False
    steps_per_print: int = 1000
    mins_per_checkpoint: float = 30
    num_epochs: int = 1

    # Other useful meta-data, set automatically
    data_seq_length: Optional[int] = None
    data_num_channels: Optional[int] = None

    @property
    def data_shape(self):
        return self.data_seq_length, self.data_num_channels

    @property
    def optimizer(self):
        return optax.adam(self.learning_rate)

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
                )
            ).encode("utf-8")
        )
        return f"{hash_int:08x}"

    @property
    def checkpoint_prefix(self):
        return self.id + "_"


def load_options():
    H = tyro.cli(Hyperparams)
    if H.enable_wandb:
        import wandb

        wandb.init(
            config=dataclasses.asdict(H),
            name=H.id,
        )
    H.logprint("Launching run", id=H.id)
    return H
