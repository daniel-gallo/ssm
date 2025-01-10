from dataclasses import dataclass

from hps import load_options, Hyperparams
from data import load_data


@dataclass
class TrainState:
    # Neural network weights, epoch/iteration number, and PRNG state
    pass

def load_train_state(H: Hyperparams) -> TrainState:
    # Load latest checkpoint from disk if any exists, otherwise initialize
    # state
    raise NotImplementedError

# @jax.jit
def train_iter(H: Hyperparams, state: TrainState, batch) -> TrainState:
    raise NotImplementedError

def train_epoch(H: Hyperparams, state: TrainState, data) -> TrainState:
    raise NotImplementedError

def train(H: Hyperparams, state: TrainState, data):
    raise NotImplementedError

def main():
    H = load_options()
    H.logprint("Loading data")
    H, data = load_data(H)
    H.logprint("Training")
    # TODO:
    # H, train_state = load_train_state(H)
    # train(H, train_state, data)
    if H.enable_wandb:
        import wandb
        wandb.finish()

if __name__ == "__main__":
    main()
