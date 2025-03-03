import dataclasses
from abc import abstractmethod
from typing import Callable, Optional
from zlib import adler32

import flax.linen as nn
import optax

from log_util import logprint

_early_logsteps = set(2**e for e in range(12))


@dataclasses.dataclass(frozen=True)
class Hyperparams:
    # Directories
    data_dir: str = "data"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    sample_dir: str = "samples"

    # Training options
    seed: int = 0
    batch_size: int = 32
    learning_rate: float = 1e-3
    grad_clip: float = 200
    skip_threshold: float = 1000
    shuffle_before_epoch: bool = True
    enable_wandb: bool = True
    steps_per_print: int = 1000
    epochs_per_eval: int = 1
    mins_per_checkpoint: float = 30
    num_samples_per_eval: int = 8
    num_epochs: int = 30
    batch_size_eval: int = 128

    # Dataset
    dataset: str = "binarized-mnist"
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

        # By default, we will use all attributes except for these
        blacklist = set(
            [
                "checkpoint_dir",
                "data_dir",
                "data_preprocess_fn",
                "enable_wandb",
                "epochs_per_eval",
                "log_dir",
                "mins_per_checkpoint",
                "num_epochs",
                "num_samples_per_eval",
                "sample_dir",
                "steps_per_print",
            ],
        )
        all_attributes = dataclasses.asdict(self)
        assert all(
            blacklisted in all_attributes.keys() for blacklisted in blacklist
        )
        attributes = tuple(
            value
            for attribute, value in all_attributes.items()
            if attribute not in blacklist
        )
        hash_int = adler32(repr(attributes).encode("utf-8"))
        return f"{hash_int:08x}"

    @property
    def checkpoint_prefix(self):
        return self.id + "_"

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        pass

    @property
    @abstractmethod
    def sample_prior(self) -> Callable:
        pass
