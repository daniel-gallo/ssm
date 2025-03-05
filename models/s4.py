from dataclasses import dataclass

import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import normal
from jax.numpy.linalg import eigh, inv, matrix_power
from jax.scipy.signal import convolve

from hps import Hyperparams


def scan_SSM(Ab, Bb, Cb, u, x0):
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)


def causal_convolution(u, K, nofft=False):
    if nofft:
        return convolve(u, K, mode="full")[: u.shape[0]]
    else:
        assert K.shape[0] == u.shape[0]
        ud = np.fft.rfft(np.pad(u, (0, K.shape[0])))
        Kd = np.fft.rfft(np.pad(K, (0, u.shape[0])))
        out = ud * Kd
        return np.fft.irfft(out)[: u.shape[0]]


@jax.jit
def cauchy(v, omega, lambd):
    """Cauchy matrix multiplication: (n), (l), (n) -> (l)"""

    def cauchy_dot(_omega):
        return (v / (_omega - lambd)).sum()

    return jax.vmap(cauchy_dot)(omega)


def discrete_DPLR(Lambda, P, Q, B, C, step, L):
    # Convert parameters to matrices
    B = B[:, np.newaxis]
    Ct = C[np.newaxis, :]

    N = Lambda.shape[0]
    A = np.diag(Lambda) - P[:, np.newaxis] @ Q[:, np.newaxis].conj().T
    Id = np.eye(N)

    # Forward Euler
    A0 = (2.0 / step) * Id + A

    # Backward Euler
    D = np.diag(1.0 / ((2.0 / step) - Lambda))
    Qc = Q.conj().T.reshape(1, -1)
    P2 = P.reshape(-1, 1)
    A1 = D - (D @ P2 * (1.0 / (1 + (Qc @ D @ P2))) * Qc @ D)

    # A bar and B bar
    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    # Recover Cbar from Ct
    Cb = Ct @ inv(Id - matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()


def kernel_DPLR(Lambda, P, Q, B, C, step, L):
    # Evaluate at roots of unity
    # Generating function is (-)z-transform, so we evaluate at (-)root
    Omega_L = np.exp((-2j * np.pi) * (np.arange(L) / L))

    aterm = (C.conj(), Q.conj())
    bterm = (B, P)

    g = (2.0 / step) * ((1.0 - Omega_L) / (1.0 + Omega_L))
    c = 2.0 / (1.0 + Omega_L)

    # Reduction to core Cauchy kernel
    k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, Lambda)
    atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    out = np.fft.ifft(atRoots, L).reshape(L)
    return out.real


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


def init(x):
    def _init(key, shape):
        assert shape == x.shape
        return x

    return _init


def make_HiPPO(N):
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def make_NPLR_HiPPO(N):
    # Make -HiPPO
    nhippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return nhippo, P, B


def make_DPLR_HiPPO(N):
    """Diagonalize NPLR representation"""
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    # Check skew symmetry
    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)
    # assert np.allclose(Lambda_real, S_diag, atol=1e-3)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V


def hippo_initializer(N):
    Lambda, P, B, _ = make_DPLR_HiPPO(N)
    return init(Lambda.real), init(Lambda.imag), init(P), init(B)


class S4Layer(nn.Module):
    N: int
    l_max: int
    decode: bool = False

    # Special parameters with multiplicative factor on lr and no weight decay (handled by main train script)
    lr = {
        "Lambda_re": 0.1,
        "Lambda_im": 0.1,
        "P": 0.1,
        "B": 0.1,
        "log_step": 0.1,
    }

    def setup(self):
        # Learned Parameters (C is complex!)
        init_A_re, init_A_im, init_P, init_B = hippo_initializer(self.N)
        self.Lambda_re = self.param("Lambda_re", init_A_re, (self.N,))
        self.Lambda_im = self.param("Lambda_im", init_A_im, (self.N,))
        # Ensure the real part of Lambda is negative
        # (described in the SaShiMi follow-up to S4)
        self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        self.P = self.param("P", init_P, (self.N,))
        self.B = self.param("B", init_B, (self.N,))
        # C should be init as standard normal
        # This doesn't work due to how JAX handles complex optimizers https://github.com/deepmind/optax/issues/196
        # self.C = self.param("C", normal(stddev=1.0, dtype=np.complex64), (self.N,))
        self.C = self.param("C", normal(stddev=0.5**0.5), (self.N, 2))
        self.C = self.C[..., 0] + 1j * self.C[..., 1]
        self.D = self.param("D", nn.initializers.ones, (1,))
        self.step = np.exp(self.param("log_step", log_step_initializer(), (1,)))

        if not self.decode:
            # CNN mode, compute kernel.
            self.K = kernel_DPLR(
                self.Lambda,
                self.P,
                self.P,
                self.B,
                self.C,
                self.step,
                self.l_max,
            )

        else:
            # RNN mode, discretize

            # Flax trick to cache discrete form during decoding.
            def init_discrete():
                return discrete_DPLR(
                    self.Lambda,
                    self.P,
                    self.P,
                    self.B,
                    self.C,
                    self.step,
                    self.l_max,
                )

            ssm_var = self.variable("prime", "ssm", init_discrete)
            if self.is_mutable_collection("prime"):
                ssm_var.value = init_discrete()
            self.ssm = ssm_var.value

            # RNN Cache
            self.x_k_1 = self.variable(
                "cache", "cache_x_k", np.zeros, (self.N,), np.complex64
            )

    def __call__(self, u):
        # This is identical to SSM Layer
        if not self.decode:
            # CNN Mode
            return causal_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(*self.ssm, u[:, np.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


def cloneLayer(layer):
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1, "cache": 1, "prime": 1},
        split_rngs={"params": True},
    )


S4Layer = cloneLayer(S4Layer)


class Embedding(nn.Embed):
    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self, x):
        y = nn.Embed(self.num_embeddings, self.features)(x[..., 0])
        return np.where(x > 0, y, 0.0)


class SequenceBlock(nn.Module):
    layer_cls: nn.Module
    layer: dict  # Hyperparameters of inner layer
    dropout: float
    d_model: int
    prenorm: bool = True
    glu: bool = True
    training: bool = True
    decode: bool = False

    def setup(self):
        self.seq = self.layer_cls(**self.layer, decode=self.decode)
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(self.d_model)
        if self.glu:
            self.out2 = nn.Dense(self.d_model)
        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x):
        skip = x
        if self.prenorm:
            x = self.norm(x)
        x = self.seq(x)
        x = self.drop(nn.gelu(x))
        if self.glu:
            x = self.out(x) * jax.nn.sigmoid(self.out2(x))
        else:
            x = self.out(x)
        x = skip + self.drop(x)
        if not self.prenorm:
            x = self.norm(x)
        return x


class StackedModel(nn.Module):
    layer_cls: nn.Module
    layer: dict  # Extra arguments to pass into layer constructor
    d_output: int
    d_model: int
    n_layers: int
    prenorm: bool = True
    dropout: float = 0.0
    embedding: bool = False  # Use nn.Embed instead of nn.Dense encoder
    classification: bool = False
    training: bool = True
    decode: bool = False  # Probably should be moved into layer_args

    def setup(self):
        if self.embedding:
            self.encoder = Embedding(self.d_output, self.d_model)
        else:
            self.encoder = nn.Dense(self.d_model)
        self.decoder = nn.Dense(self.d_output)
        self.layers = [
            SequenceBlock(
                layer_cls=self.layer_cls,
                layer=self.layer,
                prenorm=self.prenorm,
                d_model=self.d_model,
                dropout=self.dropout,
                training=self.training,
                decode=self.decode,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):
        if not self.classification:
            if not self.embedding:
                pass
                # TODO: I removed the normalization
                # x = x / 255.0  # Normalize
            if not self.decode:
                x = np.pad(x[:-1], [(1, 0), (0, 0)])
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        if self.classification:
            x = np.mean(x, axis=0)
        x = self.decoder(x)
        # return nn.log_softmax(x, axis=-1)
        return x


BatchStackedModel = nn.vmap(
    StackedModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True},
)


def log_likelihood(logits, x):
    bat, seq, chan, cat = logits.shape
    assert x.shape == (bat, seq, chan)
    return np.sum(
        np.take_along_axis(jax.nn.log_softmax(logits), x[..., None], -1)
    )


def loss_and_metrics(logits, x):
    normalizer = x.size * np.log(2)
    ll = log_likelihood(logits, x) / normalizer
    loss = -ll
    return loss, {"loss": loss, "log-like": ll}


@dataclass(frozen=True)
class S4Hyperparams(Hyperparams):
    d_model: int = 128
    d_ssm: int = 64
    n_layers: int = 4
    dropout: float = 0.0
    # Hyperparams the following:
    # data_seq_length
    # data_preprocess_fn
    # data_num_channels

    @property
    def model(self):
        return S4(self)

    @property
    def sample_prior(self):
        return S4.sample_prior


class S4(nn.Module):
    H: S4Hyperparams

    def setup(self):
        self.train_model = BatchStackedModel(
            layer_cls=S4Layer,
            d_output=self.H.data_num_cats,
            classification=False,
            d_model=self.H.d_model,
            n_layers=self.H.n_layers,
            decode=False,
            dropout=self.H.dropout,
            layer={"N": self.H.d_ssm, "l_max": self.H.data_seq_length},
        )

        self.inference_model = BatchStackedModel(
            layer_cls=S4Layer,
            d_output=self.H.data_num_cats,
            classification=False,
            d_model=self.H.d_model,
            n_layers=self.H.n_layers,
            decode=True,
            dropout=self.H.dropout,
            training=False,
            layer={"N": self.H.d_ssm, "l_max": self.H.data_seq_length},
        )

    def __call__(self, x, rng):
        # Training: CNN mode (decode=False)
        logits = self.train_model(self.H.data_preprocess_fn(x))
        logits = logits[:, :, None, :]  # Add channel dim
        return loss_and_metrics(logits, x)

    def sample_prior(self, gen_len, n_samples, rng):
        # Initialize with zeros; only the first token is used
        start = np.zeros(
            (n_samples, gen_len, self.H.data_num_channels), dtype=np.int32
        )
        trained_params = self.variables["params"]["train_model"]

        variables = self.inference_model.init(rng, start)

        # Initialize variables with trained parameters
        vars = {
            "params": trained_params,
            "prime": variables["prime"],
            "cache": variables["cache"],
        }

        # Precompute SSM parameters using trained weights (only once!)
        _, prime_vars = self.inference_model.apply(
            vars,
            start,
            mutable=["prime"],
            rngs={"dropout": jax.random.PRNGKey(0)},
        )

        # JAX-compatible loop state: (x, rng, cache)
        initial_state = (
            start,
            rng,
            vars["cache"],  # Initialize cache from trained model
        )

        def loop_body(i, state):
            x, rng, cache = state
            rng, sample_rng = jax.random.split(rng)

            # JIT-safe dynamic slice instead of x[:,i:i+1,:]
            current_input = jax.lax.dynamic_slice(
                x,
                (0, i, 0),  # Start indices (batch, seq, channel)
                (
                    x.shape[0],
                    1,
                    x.shape[2],
                ),  # Slice size (full batch, 1 step, all channels)
            )

            # Run model
            out, new_vars = self.inference_model.apply(
                {
                    "params": trained_params,
                    "prime": prime_vars["prime"],
                    "cache": cache,
                },
                self.H.data_preprocess_fn(current_input),
                mutable=["cache"],
            )

            # Sample next token
            next_token = jax.random.categorical(sample_rng, out[..., 0, :])

            # JIT-safe dynamic update instead of x.at[:,i+1,:].set()
            x = jax.lax.dynamic_update_slice(
                x,
                next_token[..., None, None],  # Add seq and channel dims
                (0, i + 1, 0),  # Update position
            )

            return (x, rng, new_vars["cache"])

        # Execute optimized XLA loop
        final_x, _, _ = jax.lax.fori_loop(
            0,  # Start index
            self.H.data_seq_length,  # End index (exclusive)
            jax.jit(loop_body),
            initial_state,
        )

        final_x = jax.nn.one_hot(final_x, self.H.data_num_cats).squeeze()
        return final_x
