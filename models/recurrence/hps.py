import dataclasses
from typing import Literal

DTYPE_SCALING = {
    "real": 1,
    "complex": 2,
    "quaternion": 4,
}


@dataclasses.dataclass(frozen=True)
class RNNHyperparams:
    block_type: Literal["rnn", "lru", "old_lru", "rglru"] = "rglru"
    scan_implementation: Literal[
        "linear_pallas", "linear_native", "associative_native"
    ] = "linear_pallas"

    d_hidden: int = 256
    dtype_hidden: Literal["real", "complex", "quaternion"] = "complex"
    input_norm: Literal["learnable", "fixed", "none"] = "fixed"
    pos_embedding: bool = False

    # Parameter initialization
    log_a_scale: float = -1.0
    init_minval_real: float = 0.9
    init_maxval_real: float = 0.99
    init_maxval_imag: float = 0.1
    adaptive_phase: bool = False
    adaptive_d: bool = False

    # Parametrisation
    param_real: Literal["softplus", "exponential"] = "exponential"
    param_imag: Literal["linear", "tanh", "exponential"] = "linear"

    # Gating mechanisms used (default = RGLRU)
    n_diag_blocks: int = 32
    gate_x: Literal["sigmoid", "tanh", "mlp", "none"] = "sigmoid"
    gate_a_real: Literal["log_sigmoid", "mlp", "pre_mlp", "tanh","none"] = "log_sigmoid"
    gate_a_imag: Literal["sigmoid", "tanh", "mlp", "same", "none"] = "same"

    @property
    def dtype_scaling(self) -> int:
        return DTYPE_SCALING[self.dtype_hidden]
