import dataclasses
import os
import wave
import zipfile
from functools import partial
from os import path
from pathlib import Path
from urllib.request import urlretrieve

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from PIL import Image

from hps import Hyperparams


def load_data(H: Hyperparams):
    match H.dataset:
        case "binarized-mnist":
            H, data = load_mnist_binarized(H)
        case "sc09":
            H, data = load_sc09(H)
        case _:
            raise ValueError(f'Invalid dataset "{H.dataset}".')
    return H, data


def load_mnist_binarized(H: Hyperparams):
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


def load_sc09(H):
    seq_len = 16_000
    num_cats = 256
    base = Path(H.data_dir) / "sc09"

    def maybe_download_data():
        if base.exists():
            return

        zip_file = base.parent / "sc09.zip"
        zip_file.parent.mkdir(parents=True, exist_ok=True)
        if not zip_file.exists():
            H.logprint("Downloading SC09 zip file...")
            urlretrieve(
                "https://huggingface.co/datasets/krandiash/sc09/resolve/main/sc09.zip?download=true",
                zip_file,
            )

        H.logprint("Extracting SC09 zip file...")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(zip_file.parent)

    def wav_to_np(path: str) -> np.ndarray:
        with wave.open(path, "rb") as wav_file:
            n_frames = wav_file.getnframes()
            signal = wav_file.readframes(n_frames)
            return np.frombuffer(signal, dtype=np.int16)

    def np_to_wav(x: np.ndarray, path: str):
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(x.tobytes())

    def pad(x, seq_len=16_000):
        n_samples = len(x)
        padded_x = np.zeros((n_samples, seq_len), dtype=x[0].dtype)
        for samp_idx, sample in enumerate(x):
            padded_x[samp_idx, : len(sample)] = sample
        return padded_x

    def mu_law_encode(x):
        # See https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
        # [-2**15, 2**15 - 1] -> [0, 1]
        x = (x.astype(np.float32) + 32_768) / 65_536
        # [0, 1] -> [-1, 1]
        x = 2 * x - 1
        # [-1, 1] -> [-1, 1] (but squeezed)
        x = np.sign(x) * np.log1p(255 * np.abs(x)) / np.log1p(255)
        # [-1, 1] -> [0, 1]
        x = (x + 1) / 2
        # [0, 1] -> [0, 255]
        x = 255 * x
        x = x.astype(np.int32)
        return x

    def mu_law_decode(x):
        # See https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
        # [0, 255] -> [0, 1]
        x = x.astype(np.float32) / 255
        x = np.clip(x, 0, 1)
        # [0, 1] -> [-1, 1]
        x = 2 * x - 1
        # [-1, 1] (squeezed) -> [-1, 1]
        x = np.sign(x) * ((1 + 255) ** np.abs(x) - 1) / 255
        # [-1, 1] -> [0, 1]
        x = (x + 1) / 2
        # [0, 1] -> [-2**15, 2**15 - 1]
        x = (x * 65_536 - 32_768).astype(np.int16)
        return x

    def maybe_cache_data():
        if (base / "cache.npz").exists():
            return

        testing_list = base / "testing_list.txt"
        validation_list = base / "validation_list.txt"
        test_tracks = set(
            testing_list.read_text().splitlines()
            + validation_list.read_text().splitlines()
        )

        train = []
        test = []
        H.logprint("Reading SC09 wav files...")
        for track in base.glob("**/*.wav"):
            label = track.parent.name

            x = train
            if f"{label}/{track.name}" in test_tracks:
                x = test

            x.append(wav_to_np(str(track)))

        train = mu_law_encode(pad(train)).astype(np.uint8)
        test = mu_law_encode(pad(test)).astype(np.uint8)

        np.savez(base / "cache.npz", train=train, test=test)

    maybe_download_data()
    maybe_cache_data()

    cache = np.load(base / "cache.npz")
    # The cache is stored as uint8, but I don't want to deal with overflows
    train = cache["train"].astype(np.int16)
    test = cache["test"].astype(np.int16)

    # Add a dummy channel dim
    train, test = np.expand_dims(train, -1), np.expand_dims(test, -1)

    assert train.dtype == test.dtype == np.int16
    assert train.shape == (31158, 16000, 1)
    assert test.shape == (7750, 16000, 1)

    H = dataclasses.replace(
        H,
        data_seq_length=seq_len,
        data_num_channels=1,
        data_num_cats=num_cats,
        data_preprocess_fn=lambda x: (2 * x / 256) - 1,
    )
    return H, (train, test)


def save_samples(H: Hyperparams, step, samples):
    match H.dataset:
        case "binarized-mnist":
            save_mnist_binarized(H, step, samples)


def save_mnist_binarized(H: Hyperparams, step, samples):
    batch_size, seq_length, num_channels = samples.shape
    assert seq_length == 28 * 28
    assert num_channels == 2

    sample_dir = Path(H.sample_dir) / H.id
    sample_dir.mkdir(parents=True, exist_ok=True)

    samples = jnp.reshape(samples, (batch_size, 28, 28, 2))
    samples = jnp.repeat(samples, 4, 1)
    samples = jnp.repeat(samples, 4, 2)
    samples = jnp.astype(
        255 * jnp.hstack(nn.softmax(samples)[:, :, :, 1]), "uint8"
    )
    samples = Image.fromarray(np.array(samples), mode="L")
    samples.save(sample_dir / f"step-{step}.png")
    if H.enable_wandb:
        import wandb

        wandb.log({"samples": wandb.Image(samples)}, step)
