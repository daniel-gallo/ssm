import dataclasses
import os
import wave
import zipfile
from functools import partial
from multiprocessing import Pool
from os import path
from pathlib import Path
from typing import List
from urllib.request import urlretrieve

import ffmpeg
import jax.numpy as jnp
import numpy as np
from PIL import Image
from scipy.io import wavfile
from tensorflow.io import gfile

from hps import Hyperparams


def load_data(H: Hyperparams):
    os.makedirs(H.data_dir, exist_ok=True)
    match H.dataset:
        case "binarized-mnist":
            H, data = load_mnist_binarized(H)
        case "sc09":
            H, data = load_sc09(H)
        case "sc09-mp3":
            H, data = load_sc09_mp3(H)
        case "sc09-mp3-downsampled":
            H, data = load_sc09_mp3_downsampled(H)
        case "beethoven":
            H, data = load_beethoven(H)
        case "youtube_mix":
            H, data = load_youtube_mix(H)
        case _:
            raise ValueError(f'Invalid dataset "{H.dataset}".')
    return H, data


def load_mnist_binarized(H: Hyperparams):
    data_num_training_samples = 60_000
    H = dataclasses.replace(
        H,
        data_seq_length=784,
        data_num_channels=1,
        data_num_cats=2,
        data_preprocess_fn=lambda x: 2 * x - 1,
        data_num_training_samples=data_num_training_samples,
    )

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

    assert train.shape == (data_num_training_samples, 784, 1)
    assert test.shape == (10_000, 784, 1)

    return H, (train, test)


def maybe_download(H, url: str, path: Path):
    if path.exists():
        return

    H.logprint(f"Downloading {url}...")
    urlretrieve(url, path)


def unzip(H, zip_file: Path, extract_dir: Path):
    H.logprint(f"Extracting {zip_file}...")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def wav_to_np(path: Path) -> np.ndarray:
    rate, data = wavfile.read(path)
    return data


def np_to_wav(x: np.ndarray, path: Path, framerate):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(framerate)
        wav_file.writeframes(x.tobytes())


def pad(xs: List[np.ndarray], seq_len: int) -> np.ndarray:
    xs_padded = np.zeros((len(xs), seq_len), dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        xs_padded[i, : len(x)] = x
    return xs_padded


def mu_law_encode(audio):
    # Based on implementation from S4 repo. Audio is assumed to have shape
    # (batch, length), i.e. single channel.
    def minmax_scale(x, range_min, range_max):
        assert x.ndim == 2
        min_val = np.min(x, axis=1, keepdims=True)
        max_val = np.max(x, axis=1, keepdims=True)
        return range_min + (range_max - range_min) * (x - min_val) / (
            max_val - min_val + 1e-6
        )

    bits = 8
    mu = (1 << bits) - 1
    audio = minmax_scale(audio, range_min=-1, range_max=1)

    numerator = np.log1p(mu * np.abs(audio + 1e-8))
    denominator = np.log1p(mu)
    encoded = np.sign(audio) * (numerator / denominator)

    encoded = (encoded + 1) / 2

    quantized = np.int32((256 - 0.01) * encoded + 0.005)
    return quantized


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


def load_sc09(H):
    data_num_training_samples = 31158
    seq_len = 16_000
    num_cats = 256
    url = "https://huggingface.co/datasets/krandiash/sc09/resolve/main/sc09.zip"
    base_dir = Path(H.data_dir) / "sc09"
    zip_file = base_dir / "sc09.zip"
    unzipped_dir = base_dir / "sc09"
    cache_file = base_dir / "cache.npz"
    cache_url = "gs://ssm-datasets/sc09.npz"

    os.makedirs(base_dir, exist_ok=True)

    H = dataclasses.replace(
        H,
        data_seq_length=seq_len,
        data_num_channels=1,
        data_num_cats=num_cats,
        data_preprocess_fn=lambda x: (2 * x / 256) - 1,
        data_num_training_samples=data_num_training_samples,
        data_framerate=16000,
    )

    if gfile.exists(cache_url) and not cache_file.exists():
        H.logprint("Loading sc09 from GCS")
        gfile.copy(cache_url, cache_file)

    if not cache_file.exists():
        if not unzipped_dir.exists():
            maybe_download(H, url, zip_file)
            unzip(H, zip_file, base_dir)

        testing_list = unzipped_dir / "testing_list.txt"
        validation_list = unzipped_dir / "validation_list.txt"
        test_tracks = set(
            testing_list.read_text().splitlines()
            + validation_list.read_text().splitlines()
        )

        train = []
        test = []
        H.logprint("Reading SC09 wav files...")
        for track in unzipped_dir.glob("**/*.wav"):
            if f"{track.parent.name}/{track.name}" in test_tracks:
                test.append(wav_to_np(track))
            else:
                train.append(wav_to_np(track))

        train = mu_law_encode(pad(train, seq_len)).astype(np.uint8)
        test = mu_law_encode(pad(test, seq_len)).astype(np.uint8)

        np.savez(cache_file, train=train, test=test)

    cache = np.load(cache_file)
    # The cache is stored as uint8, but I don't want to deal with overflows
    train = cache["train"].astype(np.int16)
    test = cache["test"].astype(np.int16)

    # Add a dummy channel dim
    train, test = train[..., np.newaxis], test[..., np.newaxis]

    assert train.dtype == test.dtype == np.int16
    assert train.shape == (data_num_training_samples, 16000, 1)
    assert test.shape == (7750, 16000, 1)

    np.random.RandomState(H.seed).shuffle(train)

    return H, (train, test)


def wav_to_mp3_to_wav(fname, outrate=8000):
    base, extension = path.splitext(fname)
    fname_mp3 = base + ".mp3"
    ffmpeg.input(fname).output(fname_mp3, audio_bitrate="16k").run(
        overwrite_output=True, quiet=True
    )
    ffmpeg.input(fname_mp3).output(str(fname), ar=outrate).run(
        overwrite_output=True, quiet=True
    )


def load_sc09_mp3(H):
    data_num_training_samples = 31158
    seq_len = 16_000
    num_cats = 256
    url = "https://huggingface.co/datasets/krandiash/sc09/resolve/main/sc09.zip"
    base_dir = Path(H.data_dir) / "sc09-mp3"
    zip_file = base_dir / "sc09.zip"
    unzipped_dir = base_dir / "sc09"
    cache_file = base_dir / "cache.npz"
    cache_url = "gs://ssm-datasets/sc09-mp3.npz"

    os.makedirs(base_dir, exist_ok=True)

    H = dataclasses.replace(
        H,
        data_seq_length=seq_len,
        data_num_channels=1,
        data_num_cats=num_cats,
        data_preprocess_fn=lambda x: (2 * x / 256) - 1,
        data_num_training_samples=data_num_training_samples,
        data_framerate=16000,
    )

    if gfile.exists(cache_url) and not cache_file.exists():
        H.logprint("Loading sc09-mp3-downsampled from GCS")
        gfile.copy(cache_url, cache_file)

    if not cache_file.exists():
        if not unzipped_dir.exists():
            maybe_download(H, url, zip_file)
            unzip(H, zip_file, base_dir)

        testing_list = unzipped_dir / "testing_list.txt"
        validation_list = unzipped_dir / "validation_list.txt"
        test_tracks = set(
            testing_list.read_text().splitlines()
            + validation_list.read_text().splitlines()
        )

        train = []
        test = []
        H.logprint("Reading SC09 wav files...")
        tracks = list(base_dir.glob("**/*.wav"))
        H.logprint("Converting to mp3 and back to wav...")
        with Pool(16) as P:
            P.map(partial(wav_to_mp3_to_wav, outrate=16000), tracks)
        for track in tracks:
            if f"{track.parent.name}/{track.name}" in test_tracks:
                test.append(wav_to_np(track))
            else:
                train.append(wav_to_np(track))

        H.logprint("Converting to NumPy array...")
        train = mu_law_encode(pad(train, seq_len)).astype(np.uint8)
        test = mu_law_encode(pad(test, seq_len)).astype(np.uint8)

        np.savez(cache_file, train=train, test=test)

    cache = np.load(cache_file)
    # The cache is stored as uint8, but I don't want to deal with overflows
    train = cache["train"].astype(np.int16)
    test = cache["test"].astype(np.int16)

    # Add a dummy channel dim
    train, test = train[..., np.newaxis], test[..., np.newaxis]

    assert train.dtype == test.dtype == np.int16
    assert train.shape == (data_num_training_samples, 16000, 1)
    assert test.shape == (7750, 16000, 1)

    np.random.RandomState(H.seed).shuffle(train)

    return H, (train, test)


def load_sc09_mp3_downsampled(H):
    data_num_training_samples = 31158
    seq_len = 8_000
    num_cats = 256
    url = "https://huggingface.co/datasets/krandiash/sc09/resolve/main/sc09.zip"
    base_dir = Path(H.data_dir) / "sc09-mp3-downsampled"
    zip_file = base_dir / "sc09.zip"
    unzipped_dir = base_dir / "sc09"
    cache_file = base_dir / "cache.npz"
    cache_url = "gs://ssm-datasets/sc09-mp3-downsampled.npz"

    os.makedirs(base_dir, exist_ok=True)

    H = dataclasses.replace(
        H,
        data_seq_length=seq_len,
        data_num_channels=1,
        data_num_cats=num_cats,
        data_preprocess_fn=lambda x: (2 * x / 256) - 1,
        data_num_training_samples=data_num_training_samples,
        data_framerate=8000,
    )

    if gfile.exists(cache_url) and not cache_file.exists():
        H.logprint("Loading sc09-mp3-downsampled from GCS")
        gfile.copy(cache_url, cache_file)

    if not cache_file.exists():
        if not unzipped_dir.exists():
            maybe_download(H, url, zip_file)
            unzip(H, zip_file, base_dir)

        testing_list = unzipped_dir / "testing_list.txt"
        validation_list = unzipped_dir / "validation_list.txt"
        test_tracks = set(
            testing_list.read_text().splitlines()
            + validation_list.read_text().splitlines()
        )

        train = []
        test = []
        H.logprint("Reading SC09 wav files...")
        tracks = list(base_dir.glob("**/*.wav"))
        H.logprint("Converting to mp3 and back to wav...")
        with Pool(16) as P:
            P.map(wav_to_mp3_to_wav, tracks)
        for track in tracks:
            if f"{track.parent.name}/{track.name}" in test_tracks:
                test.append(wav_to_np(track))
            else:
                train.append(wav_to_np(track))

        H.logprint("Converting to NumPy array...")
        train = mu_law_encode(pad(train, seq_len)).astype(np.uint8)
        test = mu_law_encode(pad(test, seq_len)).astype(np.uint8)

        np.savez(cache_file, train=train, test=test)

    cache = np.load(cache_file)
    # The cache is stored as uint8, but I don't want to deal with overflows
    train = cache["train"].astype(np.int16)
    test = cache["test"].astype(np.int16)

    # Add a dummy channel dim
    train, test = train[..., np.newaxis], test[..., np.newaxis]

    assert train.dtype == test.dtype == np.int16
    assert train.shape == (data_num_training_samples, 8000, 1)
    assert test.shape == (7750, 8000, 1)

    np.random.RandomState(H.seed).shuffle(train)

    return H, (train, test)


def load_beethoven(H):
    data_num_training_samples = 3808
    seq_len = 128_000
    num_cats = 256
    base_dir = Path(H.data_dir) / "beethoven"
    zip_file = base_dir / "beethoven.zip"
    unzipped_dir = base_dir / "beethoven"
    cache_file = base_dir / "cache.npz"

    os.makedirs(base_dir, exist_ok=True)

    H = dataclasses.replace(
        H,
        data_seq_length=seq_len,
        data_num_channels=1,
        data_num_cats=num_cats,
        data_preprocess_fn=lambda x: (2 * x / 256) - 1,
        data_num_training_samples=data_num_training_samples,
        data_framerate=16000,
    )

    maybe_download(
        H,
        "https://huggingface.co/datasets/krandiash/beethoven/resolve/main/beethoven.zip?download=true",
        zip_file,
    )
    if not unzipped_dir.exists():
        unzip(H, zip_file, base_dir)

    if not cache_file.exists():
        H.logprint("Reading Beethoven wav files...")
        train = []
        for i in range(3808):
            train.append(wav_to_np(unzipped_dir / f"{i}.wav"))

        test = []
        # validation is [3808, 4067]
        for i in range(3808, 4328):
            test.append(wav_to_np(unzipped_dir / f"{i}.wav"))

        train = mu_law_encode(pad(train, seq_len)).astype(np.uint8)
        test = mu_law_encode(pad(test, seq_len)).astype(np.uint8)

        np.savez(cache_file, train=train, test=test)

    cache = np.load(cache_file)
    # The cache is stored as uint8, but I don't want to deal with overflows
    train = cache["train"].astype(np.int16)
    test = cache["test"].astype(np.int16)

    # Add a dummy channel dim
    train, test = np.expand_dims(train, -1), np.expand_dims(test, -1)

    assert train.dtype == test.dtype == np.int16
    assert train.shape == (3808, seq_len, 1)
    assert test.shape == (520, seq_len, 1)

    return H, (train, test)


def load_youtube_mix(H):
    data_num_training_samples = 212
    seq_len = 960_512
    num_cats = 256
    base_dir = Path(H.data_dir) / "youtube_mix"
    zip_file = base_dir / "youtube_mix.zip"
    unzipped_dir = base_dir / "youtube_mix"
    cache_file = base_dir / "cache.npz"

    os.makedirs(base_dir, exist_ok=True)

    H = dataclasses.replace(
        H,
        data_seq_length=seq_len,
        data_num_channels=1,
        data_num_cats=num_cats,
        data_preprocess_fn=lambda x: (2 * x / 256) - 1,
        data_num_training_samples=data_num_training_samples,
        data_framerate=16000,
    )

    maybe_download(
        H,
        "https://huggingface.co/datasets/krandiash/youtubemix/resolve/main/youtube_mix.zip?download=true",
        zip_file,
    )
    if not base_dir.exists():
        unzip(H, zip_file, base_dir)

    if not cache_file.exists():
        H.logprint("Reading Youtube Mix wav files...")
        train = []
        for i in range(212):
            idx = str(i).zfill(3)
            train.append(wav_to_np(unzipped_dir / f"out{idx}.wav"))

        test = []
        # validation is [212, 225]
        for i in range(212, 241):
            idx = str(i).zfill(3)
            test.append(wav_to_np(unzipped_dir / f"out{idx}.wav"))

        train = mu_law_encode(pad(train, seq_len)).astype(np.uint8)
        test = mu_law_encode(pad(test, seq_len)).astype(np.uint8)

        np.savez(cache_file, train=train, test=test)

    cache = np.load(cache_file)
    # The cache is stored as uint8, but I don't want to deal with overflows
    train = cache["train"].astype(np.int16)
    test = cache["test"].astype(np.int16)

    # Add a dummy channel dim
    train, test = np.expand_dims(train, -1), np.expand_dims(test, -1)

    assert train.dtype == test.dtype == np.int16
    assert train.shape == (data_num_training_samples, seq_len, 1)
    assert test.shape == (29, seq_len, 1)

    return H, (train, test)


def save_samples(H: Hyperparams, step, samples):
    match H.dataset:
        case "binarized-mnist":
            save_mnist_binarized(H, step, samples)
        case "sc09" | "sc09-mp3" | "sc09-mp3-downsampled" | "beethoven" | "youtube_mix":
            save_audio(H, step, samples)
        case _:
            H.logprint(f"Dataset {H.dataset} does not support saving sampling")


def save_mnist_binarized(H: Hyperparams, step, samples):
    batch_size, seq_length, num_channels = samples.shape
    assert seq_length == 28 * 28
    assert num_channels == 1
    samples = jnp.squeeze(samples, 2)

    sample_dir = Path(H.sample_dir) / H.id
    sample_dir.mkdir(parents=True, exist_ok=True)

    samples = jnp.reshape(samples, (batch_size, 28, 28))
    samples = jnp.repeat(samples, 4, 1)
    samples = jnp.repeat(samples, 4, 2)
    samples = jnp.astype(255 * jnp.hstack(samples), "uint8")
    samples = Image.fromarray(np.array(samples), mode="L")
    samples.save(sample_dir / f"step-{step}.png")
    if H.enable_wandb:
        import wandb

        wandb.log({"samples": wandb.Image(samples)}, step)


def save_audio(H: Hyperparams, step, samples):
    batch_size, seq_length, num_channels = samples.shape
    assert num_channels == 1
    samples = jnp.squeeze(samples, 2)

    sample_dir = Path(H.sample_dir) / H.id
    sample_dir.mkdir(parents=True, exist_ok=True)

    # [0, 255] -> [-2**15, 2**15 - 1]
    samples = mu_law_decode(samples)

    sample_filenames = []
    for sample_id, sample in enumerate(samples):
        sample_path = sample_dir / f"step-{step}-audio-{sample_id}.wav"
        np_to_wav(sample, sample_path, H.data_framerate)
        sample_filenames.append(str(sample_path))

    if H.enable_wandb:
        import wandb

        wandb.log({"samples": [wandb.Audio(f) for f in sample_filenames]}, step)
