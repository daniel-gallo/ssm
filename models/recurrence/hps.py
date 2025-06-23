import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class RNNHyperparams:
    block_type: Literal["rnn", "lru", "rglru"] = "rglru"
    scan_implementation: Literal[
        "linear_pallas", "linear_native", "associative_native"
    ] = "linear_pallas"

    d_hidden: int = 256
    only_real: bool = False
    input_norm: bool = True
    pos_embedding: bool = False
    n_diag_blocks: int = 32

    # Parameter initialization
    log_a_scale: float = -8.0
    init_minval_real: float = 0.9
    init_maxval_real: float = 0.99
    init_maxval_imag: float = 0.1
    adaptive_phase: bool = False
    adaptive_d: bool = False
