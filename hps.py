import dataclasses
from typing import Optional
from zlib import adler32

import tyro

from log_util import logprint


@dataclasses.dataclass(frozen=True)
class Hyperparams:
    # Command line options
    data_dir: str = 'data'
    log_dir: str = 'logs'
    dataset: str = 'binarized-mnist'
    seed: int = 0
    batch_size: int = 32
    run_name: Optional[str] = None
    enable_wandb: bool = False

    # Other useful meta-data, set automatically
    seq_length: Optional[int] = None

    def logprint(self, *args, **kwargs):
        logprint(self.log_dir, self.id, self.enable_wandb, *args, **kwargs)

    @property
    def id(self):
        # Choose an id that depends deterministically on the model hyperparams.
        # This will allow us to refer to runs (for reloading checkpoints etc.)
        # using only the hyperparams (without having to save the id).
        hash_int = adler32(repr((
            self.dataset,
            self.seed,
            self.batch_size,
        )).encode("utf-8"))
        return f'{hash_int:08x}'


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
