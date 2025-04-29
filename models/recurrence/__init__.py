import flax.linen as nn

from models.recurrence.hps import RNNHyperparams
from models.recurrence.lru import LRU
from models.recurrence.rglru import RGLRU
from models.recurrence.rnn import RNN
from models.recurrence.lstm import LSTMScalar


def get_recurrent_block(H: RNNHyperparams):
    match H.block_type.lower():
        case "rnn":
            return RNN
        case "rglru":
            return RGLRU
        case "lru":
            return LRU
        case "lstm":
            return LSTMScalar
        case _:
            raise ValueError(f"Unknown reccurent block type: {H.block_type}")


class RNNBlock(nn.Module):
    H: RNNHyperparams
    d_out: int
    bidirectional: bool = False
    residual: bool = False
    last_scale: float = 1.0

    def setup(self):
        recurrent_block = get_recurrent_block(self.H)
        self.forward = recurrent_block(
            self.H,
            d_hidden=self.H.d_hidden,
            d_out=self.d_out,
        )
        if self.bidirectional:
            self.backward = recurrent_block(
                self.H,
                d_hidden=self.H.d_hidden,
                d_out=self.d_out,
                reverse=True,
            )
        self.last_dense = nn.Dense(self.d_out)
        self.norm = nn.LayerNorm()

    def __call__(self, x):
        identity = x
        x = self.norm(x)
        x_fwd, _ = self.forward(x)
        x = (x_fwd + self.backward(x)[0]) / 2 if self.bidirectional else x_fwd

        x = self.last_dense(nn.gelu(x))
        x = x * self.last_scale
        x = x + identity if self.residual else x
        return x


__all__ = [RNN, RGLRU, LRU, RNNHyperparams, get_recurrent_block, RNNBlock]
