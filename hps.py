import dataclasses
from typing import Optional
from zlib import adler32

import tyro


@dataclasses.dataclass(frozen=True)
class Hyperparams:
    # Command line options
    data_dir: str = 'data'
    log_dir: str = 'logs'
    dataset: str = 'binarized-mnist'
    seed: int = 0
    batch_size: int = 32
    run_name: Optional[str] = None

    # Other useful meta-data, set automatically
    seq_length: Optional[int] = None

    def logprint(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def name(self):
        # By default, choose a name that depends deterministically on the
        # model hyperparams. This will allow us to refer to runs using only the
        # hyperparams (without having to know the name).
        hash_int = adler32(repr((
            self.dataset,
            self.seed,
            self.batch_size,
        )).encode("utf-8"))
        return f'{hash_int:08x}' if self.run_name is None else self.run_name


def load_options():
    return tyro.cli(Hyperparams)
