import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from optax.losses import softmax_cross_entropy_with_integer_labels
from typing_extensions import Union

from hps import Hyperparams
from rnn import RNNBlock


def loss_and_metrics(logits, target):
    loss = softmax_cross_entropy_with_integer_labels(logits, target[..., 0])
    loss = jnp.mean(loss)
    return loss, {"loss": loss}


class DownPool(nn.Module):
    # TODO: add support for padding
    H: Hyperparams
    input_dim: int
    pool_scale: Union[int, None] = None
    pool_features: Union[int, None] = None

    def setup(self):
        pool_features = self.pool_features or self.H.pool_features

        self.linear = nn.Dense(self.input_dim * pool_features)

    def __call__(self, x):
        pool_scale = self.pool_scale or self.H.pool_scale

        batch_size, seq_len, dim = x.shape
        x = rearrange(x, "...(l m) d -> ... l (d m)", m=pool_scale)
        return self.linear(x), None

    def step(self, x, state):
        pool_scale = self.pool_scale or self.H.pool_scale

        if x is None:
            return None, state
        state = jnp.concatenate([state, x], axis=1)
        if state.shape[1] == pool_scale:
            batch_size, seq_len, dim = x.shape
            x = rearrange(state, "... h s -> ... (h s)")
            x = x[:, None, :]
            x = self.linear(x)
            return x, jnp.zeros((batch_size, 0, dim))
        else:
            return None, state

    def default_state(self, batch_size):
        return jnp.zeros((batch_size, 0, self.input_dim))


class UpPool(nn.Module):
    # TODO: add support for padding
    H: Hyperparams
    input_dim: int
    pool_scale: Union[int, None] = None
    pool_features: Union[int, None] = None

    def setup(self):
        pool_scale = self.pool_scale or self.H.pool_scale
        pool_features = self.pool_features or self.H.pool_features

        assert (self.input_dim * pool_scale) % pool_features == 0
        self.linear = nn.Dense((self.input_dim * pool_scale) // pool_features)

    def __call__(self, x):
        pool_scale = self.pool_scale or self.H.pool_scale

        batch_size, seq_len, dim = x.shape
        x = self.linear(x)
        # not sure about it, was in sushi code though
        x = jnp.pad(x[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
        x = rearrange(x, "... l (d m) -> ... (l m) d", m=pool_scale)
        return x, None

    def step(self, x, state):
        pool_scale = self.pool_scale or self.H.pool_scale

        assert len(state) > 0
        y, state = state[:, 0], state[:, 1:]
        if state.shape[1] == 0:
            assert x is not None
            x = self.linear(x)
            x = rearrange(x, "... l (d m) -> ... (l m) d", m=pool_scale)
            state = x
        else:
            assert x is None
        return y, state

    def default_state(self, batch_size):
        pool_scale = self.pool_scale or self.H.pool_scale

        return jnp.zeros((batch_size, pool_scale, self.input_dim))


class ResBlock(nn.Module):
    H: Hyperparams
    expand: Union[int, None] = None

    @nn.compact
    def __call__(self, x, deterministic=False):
        bs, seq_len, dim = x.shape
        expand = self.expand or self.H.ar_ff_expand
        z = nn.LayerNorm()(x)
        z = nn.Dense(dim * expand)(x)
        # z = nn.Dropout(self.H.ar_ff_dropout)(
        #     nn.gelu(z), deterministic=deterministic
        # )
        z = nn.gelu(z)
        z = nn.Dense(dim)(z)
        return x + z, None

    def step(self, x, state):
        return self(x), state

    def default_state(self, batch_size):
        return None


class ARModel(nn.Module):
    H: Hyperparams

    def setup(self):
        self.input_mlp = nn.Dense(self.H.ar_base_dim)
        self.cls_mlp = nn.Dense(self.H.data_num_cats)

        d_layers = []
        model_dim = self.H.ar_base_dim
        for p, expand in zip(self.H.ar_pool, self.H.ar_expand):
            d_layers.append(
                DownPool(
                    self.H,
                    model_dim,
                    pool_scale=p,
                    pool_features=expand,
                )
            )
            model_dim = model_dim * expand

        c_layers = []
        for _ in range(self.H.ar_n_layers):
            c_layers.append(RNNBlock(self.H, model_dim, residual=True))
            c_layers.append(ResBlock(self.H))

        u_layers = []
        for p, expand in zip(self.H.ar_pool[::-1], self.H.ar_expand[::-1]):
            block = []
            block.append(
                UpPool(
                    self.H,
                    model_dim,
                    pool_scale=p,
                    pool_features=expand,
                )
            )

            model_dim = model_dim // expand

            for _ in range(self.H.ar_n_layers):
                block.append(RNNBlock(self.H, model_dim, residual=True))
                block.append(ResBlock(self.H))
            u_layers.append(block)

        self.d_layers = d_layers
        self.c_layers = c_layers
        self.u_layers = u_layers

        assert model_dim == self.H.ar_base_dim

    def __call__(self, x, rng=None):
        target = x.copy()
        # temp. treat zero vector as kinda eos token

        x = jnp.pad(x[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
        x = self.input_mlp(x)
        outputs = []
        outputs.append(x)

        for layer in self.d_layers:
            x, _ = layer(x)
            outputs.append(x)

        for layer in self.c_layers:
            x, _ = layer(x)
        x = x + outputs.pop()

        for block in self.u_layers:
            for layer in block:
                x, _ = layer(x)
                if isinstance(layer, UpPool):
                    x = x + outputs.pop()
                    outputs.append(x)
            x = x + outputs.pop()

        x = self.cls_mlp(x)
        return loss_and_metrics(x, target)

    def default_state(self, batch_size):
        layers = list(self.d_layers) + list(self.c_layers) + list(self.u_layers)
        return [layer.default_state(batch_size) for layer in layers]

    def step(self, x, state):
        state = state[::-1]

        outputs = []
        next_state = []
        for layer in self.d_layers:
            outputs.append(x)
            x, _next_state = layer.step(x, state.pop())
            next_state.append(_next_state)
            if x is None:
                break

        if x is None:
            skipped = len(self.d_layers) - len(outputs)
            for _ in range(skipped + len(self.c_layers)):
                next_state.append(state.pop())
            for i in range(skipped):
                for _ in range(len(self.u_layers[i])):
                    next_state.append(state.pop())
            u_layers = list(self.u_layers)[skipped:]
        else:
            outputs.append(x)
            for layer in self.c_layers:
                x, _next_state = layer.step(x, state.pop())
                next_state.append(_next_state)
            x = x + outputs.pop()
            u_layers = self.u_layers

        for block in u_layers:
            for layer in block:
                x, _next_state = layer.step(x, state.pop())
                next_state.append(_next_state)
                if isinstance(layer, UpPool):
                    x = x + outputs.pop()
                    outputs.append(x)
            x = x + outputs.pop()

        x = self.cls_mlp(x)
        return x, next_state

    def sample_prior(self, gen_len, n_samples, rng, data_preprocess_fn=None):
        state = self.default_state(n_samples)
        x = jnp.zeros((n_samples, 1, self.H.data_num_channels))

        output = []
        for _ in range(gen_len):
            iter_rng, rng = jax.random.split(rng)
            x, state = self.step(x, state)
            x = jax.random.categorical(iter_rng, x, axis=-1)
            output.append(jax.nn.one_hot(x, num_classes=self.H.data_num_cats))
            x = x[..., None]

        return jnp.concatenate(output, axis=1)
