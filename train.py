import dataclasses
import os
import time
from dataclasses import dataclass
from functools import partial
from os import path
from typing import Annotated, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training import checkpoints
from jax import random, tree, tree_util
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.util import safe_map

from data import load_data, save_samples
from hps import Hyperparams
from models import (
    ARHyperparams,
    DiffusionHyperparams,
    FlowHyperparams,
    PatchARHyperparams,
    S4Hyperparams,
    VSSMHyperparams,
)

map = safe_map
_mesh = jax.make_mesh((jax.device_count(),), ("batch",))
SHARDING_REPLICATED = NamedSharding(_mesh, P())
SHARDING_BATCH = NamedSharding(_mesh, P("batch"))


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
    return tree.map(lambda *args: jnp.mean(jnp.array(args)), *metrics)


def prepend_to_keys(d, s):
    return {s + k: d[k] for k in d}


def clip_grad(H: Hyperparams, g, metrics):
    g_flat, treedef = tree.flatten(g)
    norm = jnp.linalg.norm(jnp.array(map(jnp.linalg.norm, g_flat)))
    clip_coeff = (
        jnp.minimum(H.grad_clip / (norm + 1e-6), 1) if H.grad_clip else 1
    )

    skip = (
        jnp.isnan(metrics["loss"]) | ~(norm < H.skip_threshold)
        if H.skip_threshold
        else jnp.isnan(metrics["loss"])
    )
    assert jnp.isscalar(skip)

    metrics["grad_norm"] = norm

    return treedef.unflatten([clip_coeff * x for x in g_flat]), skip, metrics


def cond(pred, true_val, false_val):
    return tree.map(partial(jnp.where, pred), true_val, false_val)


@tree_util.register_dataclass
@dataclass(frozen=True)
class TrainState:
    weights: Any
    weights_ema: Any
    optimizer_state: Any
    step: int
    rng: Any


def load_train_state(H: Hyperparams):
    rng_init, rng_train = random.split(random.PRNGKey(H.seed))
    weights = H.model.init(
        rng_init,
        jnp.zeros((H.batch_size,) + H.data_shape, "int32"),
        random.PRNGKey(0),
    )

    optimizer_state = H.optimizer.init(weights)
    S = TrainState(weights, weights, optimizer_state, 0, rng_train)

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
    S = jax.device_put(S, SHARDING_REPLICATED)
    return S


@partial(jax.jit, static_argnums=0)
def train_iter(H: Hyperparams, S: TrainState, batch):
    rng, rng_iter = random.split(S.rng)

    def lossfun(weights):
        # TODO: use JAX rng instead of FLAX (temporary fix for the S4 code)
        return H.model.apply(
            weights, batch, rng_iter, rngs={"dropout": rng_iter}
        )

    gradval, metrics = jax.grad(lossfun, has_aux=True)(S.weights)
    gradval, skip, metrics = clip_grad(H, gradval, metrics)

    updates, optimizer_state_new = H.optimizer.update(
        gradval, S.optimizer_state, S.weights
    )
    weights_new = optax.apply_updates(S.weights, updates)
    weights_ema_new = tree.map(
        lambda w, e: (1 - H.ema_rate) * w + H.ema_rate * e,
        weights_new,
        S.weights_ema,
    )

    optimizer_state, weights, weights_ema = cond(
        skip,
        (S.optimizer_state, S.weights, S.weights_ema),
        (optimizer_state_new, weights_new, weights_ema_new),
    )

    return (
        TrainState(weights, weights_ema, optimizer_state, S.step + 1, rng),
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
        batch = jax.device_put(batch, SHARDING_BATCH)
        S, metrics = train_iter(H, S, batch)
        metrics = prepend_to_keys(metrics, "train/")
        metrics["lr"] = H.scheduler(S.step)
        H.logtrain(S.step, metrics)
    return S


@partial(jax.jit, static_argnums=0)
def eval_iter(H: Hyperparams, S: TrainState, rng_iter, batch):
    # TODO: use JAX rng instead of FLAX (temporary fix for the S4 code)
    _, metrics = H.model.apply(
        S.weights_ema, batch, rng_iter, rngs={"dropout": rng_iter}
    )
    return metrics


def eval(H: Hyperparams, S: TrainState, data):
    # TODO: don't skip last batch
    # We don't care too much about reproducibility here:
    rng = random.PRNGKey(int(time.time()))
    metrics = []
    for batch in reshape_batches(H.batch_size_eval, data):
        rng, rng_iter = random.split(rng)
        batch = jax.device_put(batch, SHARDING_BATCH)
        metrics.append(eval_iter(H, S, rng_iter, batch))
    return prepend_to_keys(accum_metrics(metrics), "eval/")


def generate_samples(H: Hyperparams, S: TrainState):
    save_samples(
        H,
        S.step,
        H.model.apply(
            S.weights_ema,
            H.data_seq_length,
            H.num_samples_per_eval,
            S.rng,
            method=H.sample_prior,
        ),
    )


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
        if not (e + 1) % H.epochs_per_eval:
            H.log(S.step, eval(H, S, data_test))

        if H.num_samples_per_eval and (not (e + 1) % H.epochs_per_gen):
            generate_samples(H, S)


def log_configuration(H: Hyperparams, S: TrainState):
    os.makedirs(H.log_dir, exist_ok=True)
    with open(path.join(H.log_dir, H.id + ".yaml"), "w") as f:
        f.write(tyro.to_yaml(H))

    if H.enable_wandb:
        import wandb

        config = dataclasses.asdict(H)
        num_parameters = sum(leaf.size for leaf in tree.leaves(S.weights))
        H.logprint(f"Number of parameters: {num_parameters}")
        config["num_parameters"] = num_parameters
        config["model"] = H.model.__class__.__name__

        wandb.init(
            config=config,
            name=H.id,
        )


def main():
    H = tyro.cli(
        Annotated[VSSMHyperparams, tyro.conf.subcommand("vssm")]
        | Annotated[S4Hyperparams, tyro.conf.subcommand("s4")]
        | Annotated[ARHyperparams, tyro.conf.subcommand("ar")]
        | Annotated[PatchARHyperparams, tyro.conf.subcommand("patch-ar")]
        | Annotated[DiffusionHyperparams, tyro.conf.subcommand("diffusion")]
        | Annotated[FlowHyperparams, tyro.conf.subcommand("flow")]
    )
    H, data = load_data(H)
    H.logprint("Loading train state")
    S = load_train_state(H)
    log_configuration(H, S)
    H.logprint("Training", id=H.id)
    train(H, S, data)
    if H.enable_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
