import dataclasses
import time
from functools import partial
from typing import Any, Optional

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints
from jax import random, tree, tree_util
from jsonargparse import auto_cli

from data import Dataset, PaddedArray, load_data, save_samples
from hps import Hyperparams
from log_util import log, logprint, logtrain
from models import (
    ARHyperparams,
    DiffusionHyperparams,
    HaarHyperparams,
    PatchARHyperparams,
    RecurrentGemmaHyperparams,
    S4Hyperparams,
    VSSMHyperparams,
)
from models.noname.noname import NoNameHyperparameters
from util import safe_map

jax.config.update("jax_threefry_partitionable", True)
flax.config.update("flax_use_orbax_checkpointing", False)
map = safe_map


def reshape_batches(batch_size, data: PaddedArray):
    num_batches = len(data.raw) // batch_size

    def reshape(a):
        return np.reshape(
            a[: num_batches * batch_size],
            (num_batches, batch_size) + a.shape[1:],
        )

    return map(PaddedArray, reshape(data.raw), reshape(data.lengths))


def device_put_padded_array(H: Hyperparams, data: PaddedArray) -> PaddedArray:
    return dataclasses.replace(
        data,
        raw=jax.device_put(data.raw, H.sharding_batch),
        lengths=jax.device_put(data.lengths, H.sharding_lengths),
    )


def shuffle(rng, data: PaddedArray):
    perm = random.permutation(rng, len(data.raw))

    def take_perm(a):
        return jnp.take(a, perm, 0, unique_indices=True)

    return PaddedArray(take_perm(data.raw), take_perm(data.lengths))


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TrainState:
    weights: Any
    weights_ema: Any
    optimizer_state: Any
    step: int
    rng: Any


def save_checkpoint(H: Hyperparams, S: TrainState):
    if jax.process_index() == 0:
        logprint(H, "Saving checkpoint", step=S.step)
        checkpoints.save_checkpoint(
            H.checkpoint_dir,
            dataclasses.asdict(S),
            S.step,
            H.checkpoint_prefix,
        )


def restore_checkpoint(
    H: Hyperparams, S: TrainState, override_id: Optional[str]
):
    if override_id:
        checkpoint_prefix = f"{override_id}_"
    else:
        checkpoint_prefix = H.checkpoint_prefix

    latest_checkpoint_path = checkpoints.latest_checkpoint(
        H.checkpoint_dir, checkpoint_prefix
    )
    if latest_checkpoint_path is not None:
        S_dict = checkpoints.restore_checkpoint(
            H.checkpoint_dir,
            target=dataclasses.asdict(S),
            prefix=checkpoint_prefix,
        )
        S = TrainState(**S_dict)
        logprint(H, f"Checkpoint restored from {latest_checkpoint_path}")
    else:
        logprint(H, "No checkpoint found")
    return S


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


def load_train_state(H: Hyperparams, override_id: Optional[str] = None):
    rng_init, rng_train = random.split(random.PRNGKey(H.seed))
    weights = jax.jit(H.model.init)(
        rng_init,
        PaddedArray(
            jnp.zeros((H.batch_size,) + H.data_shape, "int32"),
            jnp.zeros((H.batch_size,), "int32"),
        ),
        random.PRNGKey(0),
    )

    optimizer_state = H.optimizer.init(weights)
    S = TrainState(weights, weights, optimizer_state, 0, rng_train)
    S = restore_checkpoint(H, S, override_id)
    S = jax.device_put(S, H.sharding_train_state)
    return S


@partial(jax.jit, static_argnums=0)
def train_iter(H: Hyperparams, S: TrainState, batch: PaddedArray):
    def loss_fn(weights, minibatch, rng):
        rng_iter, rng_dropout = random.split(rng, 2)
        return H.model.apply(
            weights,
            minibatch,
            rng_iter,
            training=True,
            rngs={"dropout": rng_dropout},
        )

    def get_grads_and_metrics(weights, batch, rng, num_minibatches):
        assert len(batch) % num_minibatches == 0
        minibatch_size = len(batch) // num_minibatches

        grads = None
        metrics = None

        for i in range(num_minibatches):
            rng, rng_iter = random.split(rng)
            low = i * minibatch_size
            high = (i + 1) * minibatch_size
            minibatch = PaddedArray(
                raw=batch.raw[low:high],
                lengths=batch.lengths[low:high],
            )

            minibatch_grads, minibatch_metrics = jax.grad(
                loss_fn, has_aux=True
            )(weights, minibatch, rng_iter)

            if grads is None:
                grads = minibatch_grads
                metrics = minibatch_metrics
            else:
                grads = jax.tree.map(jnp.add, grads, minibatch_grads)
                metrics = jax.tree.map(jnp.add, metrics, minibatch_metrics)

        grads = jax.tree.map(lambda x: x / num_minibatches, grads)
        metrics = jax.tree.map(lambda x: x / num_minibatches, metrics)

        return grads, metrics

    rng, rng_grads = random.split(S.rng, 2)
    grads, metrics = get_grads_and_metrics(
        S.weights, batch, rng_grads, H.num_minibatches
    )
    grads, skip, metrics = clip_grad(H, grads, metrics)

    updates, optimizer_state_new = H.optimizer.update(
        grads, S.optimizer_state, S.weights
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


def train_epoch(H: Hyperparams, S: TrainState, data: PaddedArray):
    for batch in reshape_batches(H.batch_size, data):
        batch = device_put_padded_array(H, batch)
        S, metrics = train_iter(H, S, batch)
        metrics = prepend_to_keys(metrics, "train/")
        metrics["lr"] = H.scheduler(S.step)
        logtrain(H, S.step, metrics)
    return S


@partial(jax.jit, static_argnums=0)
def eval_iter(H: Hyperparams, S: TrainState, rng_iter, batch: PaddedArray):
    # TODO: use JAX rng instead of FLAX (temporary fix for the S4 code)
    _, metrics = H.model.apply(
        S.weights_ema, batch, rng_iter, rngs={"dropout": rng_iter}
    )
    return metrics


def eval(H: Hyperparams, S: TrainState, data: PaddedArray, split_name: str):
    # TODO: don't skip last batch
    # We don't care too much about reproducibility here:
    rng = random.PRNGKey(int(time.time()))
    metrics = []
    for batch in reshape_batches(H.batch_size_eval, data):
        rng, rng_iter = random.split(rng)
        batch = device_put_padded_array(H, batch)
        metrics.append(eval_iter(H, S, rng_iter, batch))
    return prepend_to_keys(accum_metrics(metrics), f"{split_name}/")


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


def train(H: Hyperparams, S: TrainState, data: Dataset):
    t_last_checkpoint = time.time()
    # In case we're resuming a run
    start_epoch = get_epoch(S.step, H.batch_size, H.data_num_training_samples)
    for e in range(start_epoch, H.num_epochs):
        data_train = data.train
        if H.shuffle_before_epoch or e == 0:
            rng, rng_shuffle = random.split(S.rng)
            S = dataclasses.replace(S, rng=rng)
            data_train = shuffle(rng_shuffle, data_train)
        S = train_epoch(H, S, data_train)
        if (time.time() - t_last_checkpoint) / 60 > H.mins_per_checkpoint:
            save_checkpoint(H, S)
            t_last_checkpoint = time.time()
        if not (e + 1) % H.epochs_per_eval:
            log(H, S.step, eval(H, S, data.val, split_name="val"))
        if not (e + 1) % H.epochs_per_test:
            log(H, S.step, eval(H, S, data.test, split_name="test"))

        if H.num_samples_per_eval and (not (e + 1) % H.epochs_per_gen):
            generate_samples(H, S)


def profile(H: Hyperparams, S: TrainState, data: Dataset):
    jax.profiler.start_trace("traces/")

    metrics = None
    for train_step, batch in zip(
        range(3), reshape_batches(H.batch_size, data.train)
    ):
        batch = device_put_padded_array(H, batch)
        with jax.profiler.StepTraceAnnotation(
            "train_step", step_num=train_step
        ):
            S, metrics = train_iter(H, S, batch)
    metrics["loss"].block_until_ready()

    metrics = []
    for val_step, batch in zip(
        range(3), reshape_batches(H.batch_size_eval, data.val)
    ):
        batch = device_put_padded_array(H, batch)
        with jax.profiler.StepTraceAnnotation("val_step", step_num=val_step):
            metrics.append(eval_iter(H, S, S.rng, batch))
    metrics[-1]["loss"].block_until_ready()
    jax.profiler.stop_trace()


def log_configuration(H: Hyperparams):
    if H.enable_wandb and jax.process_index() == 0:
        import wandb

        wandb.init(
            config=dataclasses.asdict(H),
            name=H.id,
        )
        wandb.mark_preempting()


def main():
    H = auto_cli(
        {
            "vssm": VSSMHyperparams,
            "s4": S4Hyperparams,
            "ar": ARHyperparams,
            "patch-ar": PatchARHyperparams,
            "diffusion": DiffusionHyperparams,
            "haar": HaarHyperparams,
            "noname": NoNameHyperparameters,
            "recurrentgemma": RecurrentGemmaHyperparams,
        },
        as_positional=False,
    )
    log_configuration(H)
    logprint(H, "Loading data")
    H, data = load_data(H)
    logprint(H, "Loading train state")
    S = load_train_state(H)
    if S.step == 0:
        log(
            H,
            0,
            dict(
                num_parameters=sum(w.size for w in tree.leaves(S.weights)),
                model=H.model.__class__.__name__,
            ),
        )

    if H.profile:
        logprint(H, "Profiling", id=H.id)
        profile(H, S, data)
    else:
        logprint(H, "Training", id=H.id)
        train(H, S, data)
    if H.enable_wandb and jax.process_index() == 0:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
