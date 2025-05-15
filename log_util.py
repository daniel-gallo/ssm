import json
import os
import time
from os import path

import jax
from jax import tree

from hps import Hyperparams


def cast_jax_scalars(d):
    # We assume that all Array metrics are, in fact, scalars
    return tree.map(lambda v: v.item() if isinstance(v, jax.Array) else v, d)


def _logprint(log_dir, name, *args, **kwargs):
    args, kwargs = cast_jax_scalars((args, kwargs))

    ctime = time.ctime()

    os.makedirs(log_dir, exist_ok=True)
    fname = path.join(log_dir, name) + ".txt"

    argdict = {}
    if args:
        argdict["message"] = " ".join([str(x) for x in args])
    argdict.update(kwargs)

    text = f"[{ctime}] {argdict}"
    print(text, flush=True)

    with open(fname, "a+") as f:
        print(text, file=f, flush=True)

    fname_jsonl = path.join(log_dir, name) + ".jsonl"
    text_jsonl = json.dumps({"time": ctime} | argdict)

    with open(fname_jsonl, "a+") as f:
        print(text_jsonl, file=f, flush=True)


def logprint(H: Hyperparams, *args, **kwargs):
    _logprint(H.log_dir, H.id, *args, **kwargs)


def logtrain(H: Hyperparams, step, metrics):
    step, metrics = cast_jax_scalars((step, metrics))
    early_logsteps = set(2**e for e in range(12))

    if int(step) in early_logsteps or not step % H.steps_per_print:
        logprint(H, step=step, **metrics)
    if H.enable_wandb and jax.process_index() == 0:
        import wandb

        wandb.log(metrics, step)


def log(H: Hyperparams, step, metrics):
    step, metrics = cast_jax_scalars((step, metrics))
    logprint(H, step=step, **metrics)
    if H.enable_wandb and jax.process_index() == 0:
        import wandb

        wandb.log(metrics, step)
