import dataclasses
import os
from functools import partial
from os import path
from urllib.request import urlretrieve

import numpy as np

from hps import Hyperparams


def load_data(H: Hyperparams):
    match H.dataset:
        case "binarized-mnist":
            H, data = mnist_binarized(H)
        case _:
            raise ValueError(f'Invalid dataset "{H.dataset}".')
    return H, data


def mnist_binarized(H):
    root_dir = path.join(H.data_dir, "mnist-binarized")
    fname_train_amat = path.join(root_dir, "train.amat")
    fname_val_amat = path.join(root_dir, "val.amat")
    fname_test_amat = path.join(root_dir, "test.amat")

    fname_train_np = path.join(root_dir, "train.npy")
    fname_test_np = path.join(root_dir, "test.npy")

    os.makedirs(root_dir, exist_ok=True)
    if not path.isfile(fname_train_amat):
        urlretrieve(
            "http://www.cs.toronto.edu/~larocheh/public/datasets/"
            "binarized_mnist/binarized_mnist_train.amat",
            fname_train_amat,
        )
    if not path.isfile(fname_val_amat):
        urlretrieve(
            "http://www.cs.toronto.edu/~larocheh/public/datasets/"
            "binarized_mnist/binarized_mnist_valid.amat",
            fname_val_amat,
        )
    if not path.isfile(fname_test_amat):
        urlretrieve(
            "http://www.cs.toronto.edu/~larocheh/public/datasets/"
            "binarized_mnist/binarized_mnist_test.amat",
            fname_test_amat,
        )

    loadtxt = partial(np.loadtxt, dtype="int32")
    if path.isfile(fname_train_np):
        train = np.load(fname_train_np)
    else:
        train = np.concatenate(
            [loadtxt(fname_train_amat), loadtxt(fname_val_amat)]
        )
        np.save(fname_train_np, train)

    if path.isfile(fname_test_np):
        test = np.load(fname_test_np)
    else:
        test = loadtxt(fname_test_amat)
        np.save(fname_test_np, test)

    # Add a dummy channel dim
    train, test = np.expand_dims(train, -1), np.expand_dims(test, -1)

    assert train.shape == (60_000, 784, 1)
    assert test.shape == (10_000, 784, 1)

    H = dataclasses.replace(
        H,
        data_seq_length=784,
        data_num_channels=1,
        data_num_cats=2,
        data_preprocess_fn=lambda x: 2 * x - 1,
    )
    return H, (train, test)
