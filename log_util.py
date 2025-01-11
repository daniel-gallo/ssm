import time
from os import path
import os

import jax
from jax import tree


def cast_jax_scalars(d):
    # We assume that all Array metrics are, in fact, scalars
    return tree.map(lambda v: v.item() if isinstance(v, jax.Array) else v, d)

def logprint(log_dir, name, log_wandb, *args, **kwargs):
    args, kwargs = cast_jax_scalars((args, kwargs))

    os.makedirs(log_dir, exist_ok=True)
    fname = path.join(log_dir, name) + ".txt"

    argdict = {}
    if args:
        argdict['message'] = ' '.join([str(x) for x in args])
    argdict.update(kwargs)

    if log_wandb:
        import wandb
        wandb.log(argdict)

    text = f'[{time.ctime()}] {argdict}'
    print(text, flush=True)

    with open(fname, "a+") as f:
        print(text, file=f, flush=True)
