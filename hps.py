import dataclasses
from abc import abstractmethod
from typing import Callable, Literal, Optional
from zlib import adler32

import flax.linen as nn
import optax
from optax.schedules import Schedule


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
    learning_rate_scheduler: Literal["constant", "warmup_cosine_decay"] = (
        "constant"
    )
    b2: float = 0.999
    learning_rate_warmup_steps: int = 1000
    weight_decay: float = 1e-4
    grad_clip: Optional[float] = 200
    skip_threshold: Optional[float] = 1000
    ema_rate: float = 0.999
    shuffle_before_epoch: bool = True
    enable_wandb: bool = True
    steps_per_print: int = 1000
    epochs_per_eval: int = 1
    mins_per_checkpoint: float = 10
    epochs_per_gen: int = 50
    num_samples_per_eval: int = 8
    num_epochs: int = 30
    batch_size_eval: int = 128

    # Note the semantics on GPU are different to TPU for the two
    # lower-precision settings.
    matmul_precision: Literal["bfloat16", "bfloat16_3x", "float32"] = (
        "bfloat16"
    )

    # Dataset
    dataset: str = "binarized-mnist"
    # Other useful meta-data, set automatically during data loading
    data_seq_length: Optional[int] = None
    data_num_channels: Optional[int] = None
    data_num_cats: Optional[int] = None
    data_preprocess_fn: Optional[Callable] = None
    data_num_training_samples: Optional[int] = None
    data_framerate: Optional[int] = None

    @property
    def data_shape(self):
        return self.data_seq_length, self.data_num_channels

    @property
    def scheduler(self) -> Schedule:
        num_training_steps = (
            self.data_num_training_samples // self.batch_size * self.num_epochs
        )

        match self.learning_rate_scheduler:
            case "constant":
                return optax.warmup_constant_schedule(
                    init_value=0,
                    peak_value=self.learning_rate,
                    warmup_steps=self.learning_rate_warmup_steps,
                )
            case "warmup_cosine_decay":
                return optax.warmup_cosine_decay_schedule(
                    init_value=0,
                    peak_value=self.learning_rate,
                    warmup_steps=self.learning_rate_warmup_steps,
                    decay_steps=num_training_steps,
                )

    @property
    def optimizer(self):
        return optax.adamw(
            self.scheduler, weight_decay=self.weight_decay, b2=self.b2
        )

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
                "epochs_per_gen",
                "log_dir",
                "mins_per_checkpoint",
                "num_samples_per_eval",
                "sample_dir",
                "steps_per_print",
                "data_seq_length",
                "data_num_channels",
                "data_num_cats",
                "data_num_training_samples",
                "data_framerate",
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
