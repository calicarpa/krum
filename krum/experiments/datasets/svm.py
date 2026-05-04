###
# @file   svm.py
# @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
#
# @section LICENSE
#
# Copyright © 2021 École Polytechnique Fédérale de Lausanne (EPFL).
# See LICENSE file.
#
# @section DESCRIPTION
#
# Lazy-(down)load and pre-process datasets from LIBSVM.
# Website: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
###

"""LIBSVM dataset loaders.

This module provides builder functions that can be registered automatically by
the :class:`experiments.Dataset` loader because they are listed in ``__all__``.
Each builder downloads the raw LIBSVM file on first use, caches a pre-processed
PyTorch tensor version, and returns an infinite-batch generator.

Example:
-------
>>> from experiments import Dataset
>>> dataset = Dataset("svm-phishing", train=True, download=True)
>>> inputs, labels = dataset.sample()

See Also:
--------
experiments.batch_dataset : helper used internally to create the infinite
    sampler from raw tensors.
"""

__all__ = ["phishing"]

import requests
import torch

from .. import dataset, tools

# ---------------------------------------------------------------------------- #
# Configuration

#: Default URL for the raw phishing dataset (LIBSVM format).
default_url_phishing = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing"

#: Default directory where pre-processed datasets are cached.
default_root = dataset.Dataset.get_default_root()

# ---------------------------------------------------------------------------- #
# Dataset lazy-loaders

#: In-memory cache for the phishing dataset after first load.
raw_phishing = None


def get_phishing(root, url):
    """Lazy-load (and optionally download) the phishing dataset.

    The dataset is downloaded from *url* in LIBSVM text format, parsed into
    dense tensors, and cached as ``phishing.pt`` under *root*. Subsequent
    calls return the cached tensors directly.

    Parameters
    ----------
    root : pathlib.Path or str
        Directory used to store the cached ``phishing.pt`` file.
    url : str or None
        URL to fetch the raw dataset from. If ``None`` and the cache is
        missing, a :class:`RuntimeError` is raised.

    Returns:
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(inputs, labels)`` where *inputs* has shape ``(11055, 68)`` and
        *labels* has shape ``(11055, 1)``.

    Raises:
    ------
    RuntimeError
        If the cache is missing and *url* is ``None``, or if the download
        or parsing fails.
    """
    global raw_phishing
    const_filename = "phishing.pt"
    const_features = 68
    const_datatype = torch.float32

    # Fast path: already loaded in memory
    if raw_phishing is not None:
        return raw_phishing

    dataset_file = root / const_filename

    # Fast path: pre-processed file already exists
    if dataset_file.exists():
        with dataset_file.open("rb") as fd:
            dataset = torch.load(fd)
            raw_phishing = dataset
            return dataset
    elif url is None:
        raise RuntimeError("Phishing dataset not in cache and download disabled")

    # Download raw dataset
    tools.info("Downloading dataset...", end="", flush=True)
    try:
        response = requests.get(url)
    except requests.exceptions.SSLError:
        tools.warning(" SSL verification failed, retrying without verification...", end="", flush=True)
        try:
            response = requests.get(url, verify=False)
        except Exception as err:
            tools.warning(" fail.")
            raise RuntimeError(f"Unable to get dataset (at {url}): {err}") from err
    except Exception as err:
        tools.warning(" fail.")
        raise RuntimeError(f"Unable to get dataset (at {url}): {err}") from err
    tools.info(" done.")
    if response.status_code != 200:
        raise RuntimeError(f"Unable to fetch raw dataset (at {url}): GET status code {response.status_code}")

    # Pre-process dataset
    tools.info("Pre-processing dataset...", end="", flush=True)
    entries = response.text.strip().split("\n")
    inputs = torch.zeros(len(entries), const_features, dtype=const_datatype)
    labels = torch.empty(len(entries), dtype=const_datatype)
    for index, entry in enumerate(entries):
        entry = entry.split(" ")
        # Set label
        labels[index] = 1 if entry[0] == "1" else 0
        # Set input
        line = inputs[index]
        for pos, setter in enumerate(entry[1:]):
            try:
                offset, value = setter.split(":")
                line[int(offset) - 1] = float(value)
            except Exception as err:
                tools.warning(" fail.")
                raise RuntimeError(f"Unable to parse dataset (line {index + 1}, position {pos + 1}): {err}") from err
    labels.unsqueeze_(1)
    tools.info(" done.")

    # Save pre-processed cache
    try:
        with dataset_file.open("wb") as fd:
            torch.save((inputs, labels), fd)
    except Exception as err:
        tools.warning(f"Unable to save pre-processed dataset: {err}")

    dataset = (inputs, labels)
    raw_phishing = dataset
    return dataset


# ---------------------------------------------------------------------------- #
# Dataset generators


def phishing(train=True, batch_size=None, root=None, download=False, *args, **kwargs):
    r"""Phishing dataset builder returning an infinite-batch generator.

    Parameters
    ----------
    train : bool, optional
        Whether to return the training split. If ``False``, the test split
        is returned instead.
    batch_size : int or None, optional
        Number of samples per batch. ``None`` or ``0`` yields the full split
        in a single batch.
    root : pathlib.Path or str or None, optional
        Cache directory. ``None`` defaults to
        :meth:`experiments.dataset.Dataset.get_default_root`.
    download : bool, optional
        Whether to allow downloading the raw file if the cache is missing.
    *args : object
        Ignored (kept for API compatibility).
    **kwargs : object
        Ignored (kept for API compatibility).

    Returns:
    -------
    generator
        Infinite sampler yielding ``(inputs, labels)`` tuples.

    Notes:
    -----
    The dataset is split at position ``8400`` (≈ 76 % train / 24 % test).
    The split point was chosen for good divisibility
    (:math:`8400 = 2^4 \times 3 \times 5^2 \times 7`).
    """
    with tools.Context("phishing", None):
        inputs, labels = get_phishing(
            root or default_root,
            None if download is None else default_url_phishing,
        )
        return dataset.batch_dataset(inputs, labels, train, batch_size, split=8400)
