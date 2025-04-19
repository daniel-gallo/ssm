import flax.linen as nn
import jax.numpy as jnp
from jax import random


class SampleRNN(nn.Module):
    d: int = 128
    num_cats: int = 256

    def setup(self):
        # Model the BOS token at position self.num_cats
        self.embed = nn.Embed(self.num_cats + 1, self.d)
        self.init_state = self.param(
            "init_state", nn.initializers.normal(), (self.d,)
        )
        self.cell = nn.GRUCell(features=self.d)
        self.cls_head = nn.Dense(features=self.num_cats)

    def _init_state(self, bs: int, seq_len: int):
        inputs = jnp.zeros((bs, seq_len + 1), dtype=int)
        inputs = inputs.at[:, 0].set(self.num_cats)

        states = jnp.zeros((bs, seq_len + 1, self.d))
        states = states.at[:, 0, :].set(self.init_state)

        indices = jnp.arange(1, seq_len + 1)[jnp.newaxis, :]

        return inputs, states, indices

    def _scan(self, inputs, states, indices, rngs=None, temperature=1.0):
        """
        The states vary both in training and sampling,
        and the inputs vary in sampling. Thus, we pass them around in the "carry".

        The indices are passed as "x".
        """

        def body_fn(cell, carry, x):
            inputs, states = carry
            i = x.squeeze()

            cell_input = self.embed(inputs[:, i - 1])
            cell_state = states[:, i - 1, :]
            cell_state, cell_output = cell(cell_state, cell_input)

            states = states.at[:, i, :].set(cell_state)
            logits = self.cls_head(cell_output)

            sampling_mode = rngs is not None
            if sampling_mode:
                sampled_tokens = random.categorical(
                    rngs[i - 1], logits / temperature
                )
                inputs = inputs.at[:, i].set(sampled_tokens)
                output = sampled_tokens
            else:
                output = logits

            return (inputs, states), output

        scan_fn = nn.scan(
            body_fn,
            in_axes=1,
            out_axes=1,
            variable_broadcast="params",
            split_rngs={"params": False},
        )
        return scan_fn(self.cell, (inputs, states), indices)[1]

    def __call__(self, x):
        """
        tokens (bs, seq) -> logits (bs, seq, num_cats)
        """
        bs, seq_len = x.shape
        inputs, states, indices = self._init_state(bs, seq_len)
        inputs = inputs.at[:, 1:].set(x)

        logits = self._scan(inputs, states, indices)
        return logits

    def sample(self, bs: int, seq_len: int, rng, temperature: float):
        """
        -> (bs, seq), with numbers in [0, self.num_cats - 1]
        """
        inputs, states, indices = self._init_state(bs, seq_len)
        rngs = random.split(rng, seq_len)

        sample = self._scan(inputs, states, indices, rngs, temperature)
        return sample
