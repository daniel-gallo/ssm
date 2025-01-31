import dataclasses
import time
from dataclasses import dataclass
from functools import partial
from os import path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints
from jax import random, tree_util
from jax.util import safe_map

from data import load_data
from hps import Hyperparams, load_options
from vssm import VSSM

map = safe_map


def reshape_batches(batch_size, data):
    num_batches = len(data) // batch_size
    return np.reshape(
        data[: batch_size * num_batches],
        (num_batches, batch_size) + data.shape[1:],
    )


def get_epoch(step, batch_size, data_size):
    num_batches = data_size // batch_size
    assert step % num_batches == 0
    return step // num_batches


def accum_metrics(metrics):
    return tree_util.tree_map(lambda *args: jnp.mean(jnp.array(args)), *metrics)


def prepend_to_keys(d, s):
    return {s + k: v for k, v in d.items()}


def clip_grad(H: Hyperparams, g, metrics):
    g_flat, treedef = tree_util.tree_flatten(g)
    norm = jnp.linalg.norm(jnp.array(map(jnp.linalg.norm, g_flat)))
    clip_coeff = jnp.minimum(H.grad_clip / (norm + 1e-6), 1)

    skip = jnp.isnan(metrics["loss"]) | ~(norm < H.skip_threshold)
    assert jnp.isscalar(skip)

    return treedef.unflatten([clip_coeff * x for x in g_flat]), skip


def cond(pred, true_val, false_val):
    return tree_util.tree_map(partial(jnp.where, pred), true_val, false_val)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainState:
    weights: Any
    optimizer_state: Any
    step: int
    rng: Any


def load_train_state(H: Hyperparams):
    rng_init, rng_train = random.split(random.PRNGKey(H.seed))
    weights = VSSM(H).init(
        rng_init,
        jnp.zeros((H.batch_size,) + H.data_shape, "int32"),
        random.PRNGKey(0),
    )
    optimizer_state = H.optimizer.init(weights)
    S = TrainState(weights, optimizer_state, 0, rng_train)

    latest_checkpoint_path = checkpoints.latest_checkpoint(
        H.checkpoint_dir, H.checkpoint_prefix
    )
    if latest_checkpoint_path is not None:
        S = checkpoints.restore_checkpoint(
            path.abspath(H.checkpoint_dir), target=S, prefix=H.checkpoint_prefix
        )
        H.logprint(f"Checkpoint restored from {latest_checkpoint_path}")
    else:
        H.logprint("No checkpoint found")
    return S


@partial(jax.jit, static_argnums=0)
def train_iter(H: Hyperparams, S: TrainState, batch):
    rng, rng_iter = random.split(S.rng)

    def lossfun(weights):
        return VSSM(H).apply(weights, batch, rng_iter)

    gradval, metrics = jax.grad(lossfun, has_aux=True)(S.weights)
    gradval, skip = clip_grad(H, gradval, metrics)

    updates, optimizer_state_new = H.optimizer.update(
        gradval, S.optimizer_state, S.weights
    )
    weights_new = optax.apply_updates(S.weights, updates)

    optimizer_state, weights = cond(
        skip, (S.optimizer_state, S.weights), (optimizer_state_new, weights_new)
    )

    return (
        TrainState(weights, optimizer_state, S.step + 1, rng),
        metrics,
    )


def train_epoch(H: Hyperparams, S: TrainState, data):
    early_logsteps = set(2**e for e in range(12))

    def should_log(step):
        return int(step) in early_logsteps or not step % H.steps_per_print

    if H.shuffle_before_epoch:
        rng, rng_shuffle = random.split(S.rng)
        S = dataclasses.replace(S, rng=rng)
        data = random.permutation(rng_shuffle, data)
    for batch in reshape_batches(H.batch_size, data):
        S, metrics = train_iter(H, S, batch)
        if should_log(S.step):
            metrics = prepend_to_keys(metrics, "train/")
            H.logprint("Train step", step=S.step, **metrics)
    return S


@partial(jax.jit, static_argnums=0)
def eval_iter(H: Hyperparams, S: TrainState, rng_iter, batch):
    _, metrics = VSSM(H).apply(S.weights, batch, rng_iter)
    return metrics


def eval(H: Hyperparams, S: TrainState, data):
    # TODO: don't skip last batch
    # We don't care too much about reproducibility here:
    rng = random.PRNGKey(int(time.time()))
    metrics = []
    for batch in reshape_batches(H.batch_size_eval, data):
        rng, rng_iter = random.split(rng)
        metrics.append(eval_iter(H, S, rng_iter, batch))
    return prepend_to_keys(accum_metrics(metrics), "eval/")


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
        if not e % H.epochs_per_eval:
            H.logprint("Eval", step=S.step, **eval(H, S, data_test))
        # TODO:
        #  - optionally generate and save samples


def main():
    H = load_options()
    H.logprint("Loading data")
    H, data = load_data(H)
    H.logprint("Loading train state")
    S = load_train_state(H)
    H.logprint("Training")
    train(H, S, data)
    if H.enable_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
