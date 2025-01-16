from dataclasses import dataclass
from functools import partial
from typing import Any
import time
from os import path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints
from jax import random

from data import load_data
from hps import Hyperparams, load_options
from model import VSSM


def reshape_batches(batch_size, data):
    num_batches = len(data) // batch_size
    return np.reshape(
        data[: batch_size * num_batches],
        (num_batches, batch_size) + data.shape[1:]
    )


def get_epoch(step, batch_size, data_size):
    num_batches = data_size // batch_size
    assert step % num_batches == 0
    return step // num_batches


@jax.tree_util.register_dataclass
@dataclass
class TrainState:
    weights: Any
    optimizer_state: Any
    step: int
    prng_state: Any


def load_train_state(H: Hyperparams):
    init_rng, train_rng = random.split(random.PRNGKey(H.seed))
    weights = VSSM(H).init(
        init_rng,
        jnp.zeros((H.batch_size,) + H.data_shape),
        random.PRNGKey(0),
    )
    optimizer_state = H.optimizer.init(weights)
    S = TrainState(weights, optimizer_state, 0, train_rng)

    latest_checkpoint_path = checkpoints.latest_checkpoint(
        H.checkpoint_dir, H.checkpoint_prefix
    )
    if latest_checkpoint_path is not None:
        S = checkpoints.restore_checkpoint(
            path.abspath(H.checkpoint_dir), target=S,
            prefix=H.checkpoint_prefix
        )
        H.logprint(f"Checkpoint restored from {latest_checkpoint_path}")
    else:
        H.logprint("No checkpoint found")
    return H, S


@partial(jax.jit, static_argnums=0)
def train_iter(H: Hyperparams, S: TrainState, batch):
    run_rng, iter_rng = random.split(S.prng_state)

    def lossfun(weights):
        return VSSM(H).apply(weights, batch, iter_rng)

    gradval, metrics = jax.grad(lossfun, has_aux=True)(S.weights)
    updates, optimizer_state = H.optimizer.update(
        gradval, S.optimizer_state, S.weights
    )
    weights = optax.apply_updates(S.weights, updates)
    return (
        TrainState(weights, optimizer_state, S.step + 1, run_rng),
        metrics,
    )


def train_epoch(H: Hyperparams, S: TrainState, data):
    early_logsteps = set(2**e for e in range(12))

    def should_log(step):
        return step.item() in early_logsteps or not step % H.steps_per_print

    # TODO shuffle data
    for batch in reshape_batches(H.batch_size, data):
        S, metrics = train_iter(H, S, batch)
        if should_log(S.step):
            H.logprint("Train step", step=S.step, **metrics)
    return S


def train(H: Hyperparams, S: TrainState, data):
    data_train, data_test = data

    t_last_checkpoint = time.time()
    # In case we're resuming a run
    start_epoch = get_epoch(S.step, H.batch_size, len(data_train))
    for e in range(start_epoch, H.num_epochs):
        S = train_epoch(H, S, data_train)
        if (time.time() - t_last_checkpoint) / 60 > H.mins_per_checkpoint:
            H.logprint("Saving checkpoint", step=S.step)
            checkpoints.save_checkpoint(
                path.abspath(H.checkpoint_dir), S, S.step, H.checkpoint_prefix
            )
            t_last_checkpoint = time.time()
        #  - evaluate on data_test
        #  - optionally generate and save samples


def main():
    H = load_options()
    H.logprint("Loading data")
    H, data = load_data(H)
    H, S = load_train_state(H)
    H.logprint("Training")
    train(H, S, data)
    if H.enable_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
