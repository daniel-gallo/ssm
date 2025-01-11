from dataclasses import dataclass
from functools import partial
from typing import Any

from flax.training import checkpoints
from jax import random
import jax.numpy as jnp
import jax
import optax
import numpy as np

from hps import load_options, Hyperparams
from data import load_data
from model import VSSM


def reshape_batches(batch_size, data):
    num_batches = len(data) // batch_size
    return np.reshape(
        data[:batch_size * num_batches],
        (num_batches, batch_size) + data.shape[1:]
    )

@jax.tree_util.register_dataclass
@dataclass
class TrainState:
    weights: Any
    optimizer_state: Any
    step: int
    prng_state: Any

def load_train_state(H: Hyperparams) -> TrainState:
    latest_checkpoint_path = checkpoints.latest_checkpoint(
        H.checkpoint_dir, H.checkpoint_prefix
    )
    if latest_checkpoint_path is not None:
        H.logprint(f"Restoring checkpoint from {latest_checkpoint_path}")
        state = checkpoints.restore_checkpoint(
            H.checkpoint_dir, target=None, prefix=H.checkpoint_prefix
        )
    else:
        H.logprint(f"No checkpoint found, initializing")
        init_rng, train_rng = random.split(random.PRNGKey(H.seed))
        weights = VSSM(H).init(
            init_rng,
            jnp.zeros((H.batch_size,) + H.data_shape),
            random.PRNGKey(0),
        )
        optimizer_state = H.optimizer.init(weights)
        state = TrainState(weights, optimizer_state, 0, train_rng)
    return H, state

@partial(jax.jit, static_argnums=0)
def train_iter(H: Hyperparams, state: TrainState, batch):
    run_rng, iter_rng = random.split(state.prng_state)
    def lossfun(weights):
        return VSSM(H).apply(weights, batch, iter_rng)
    gradval, metrics = jax.grad(lossfun, has_aux=True)(state.weights)
    updates, optimizer_state = H.optimizer.update(
        gradval, state.optimizer_state
    )
    weights = optax.apply_updates(state.weights, updates)
    return (
        TrainState(weights, optimizer_state, state.step + 1, run_rng),
        metrics,
    )

def train_epoch(H: Hyperparams, state: TrainState, data):
    # TODO shuffle data
    for batch in reshape_batches(H.batch_size, data):
        state, metrics = train_iter(H, state, batch)
        # TODO:
        #  - Do not log on every iteration
        H.logprint("Train step", step=state.step, **metrics)
    return state

def train(H: Hyperparams, state: TrainState, data):
    data_train, data_test = data
    for e in range(H.num_epochs):
        state = train_epoch(H, state, data_train)
        # TODO:
        #  - evaluate on data_test
        #  - optionally generate and save samples
        #  - optionally save a checkpoint

def main():
    H = load_options()
    H.logprint("Loading data")
    H, data = load_data(H)
    H, train_state = load_train_state(H)
    H.logprint("Training")
    train(H, train_state, data)
    if H.enable_wandb:
        import wandb
        wandb.finish()

if __name__ == "__main__":
    main()
