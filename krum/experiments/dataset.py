###
# @file   dataset.py
# @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
#
# @section LICENSE
#
# Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
# See LICENSE file.
#
# @section DESCRIPTION
#
# Dataset wrappers/helpers.
###

"""
Dataset loading, batching, and sampling utilities.

This module wraps ``torchvision.datasets`` and custom dataset modules into
uniform infinite-batch generators. It also provides helpers for train/test
splitting and raw-tensor batching.

Example
-------

>>> from experiments import Dataset, make_datasets
>>> trainset, testset = make_datasets("cifar10", train_batch=128)
>>> inputs, targets = trainset.sample(config)
"""

__all__ = [
    "Dataset",
    "batch_dataset",
    "get_default_transform",
    "make_datasets",
    "make_sampler",
]

import pathlib
import random
import tempfile
import types

import torch
import torchvision

from .. import tools

# ---------------------------------------------------------------------------- #
# Default image transformations

transforms_horizontalflip = [
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
]
transforms_mnist = [
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
]
transforms_cifar = [
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]

# Per-dataset image transformations
transforms = {
    "mnist": (transforms_mnist, transforms_mnist),
    "fashionmnist": (transforms_horizontalflip, transforms_horizontalflip),
    "cifar10": (transforms_cifar, transforms_cifar),
    "cifar100": (transforms_cifar, transforms_cifar),
    "imagenet": (transforms_horizontalflip, transforms_horizontalflip),
}


def get_default_transform(dataset, train):
    """
    Return the default transform for a torchvision dataset.

    Parameters
    ----------
    dataset : str or None
        Case-sensitive dataset name. ``None`` returns ``None``.
    train : bool
        Whether to return the training transform. Ignored when
        ``dataset`` is ``None``.

    Returns
    -------
    torchvision.transforms.Compose or None
        Composed transform, or ``None`` if the dataset is unknown.
    """
    global transforms
    transform = transforms.get(dataset)
    if transform is None:
        return None
    return torchvision.transforms.Compose(transform[0 if train else 1])


# ---------------------------------------------------------------------------- #
# Dataset wrapper class


class Dataset:
    """
    Unified dataset wrapper producing infinite batches.

    This class can wrap:

    - A ``torchvision`` dataset loaded by name.
    - A custom generator yielding batches forever.
    - A single fixed batch repeated forever.

    Parameters
    ----------
    data : str, generator, or object
        Dataset name, infinite generator, or single batch.
    name : str or None, optional
        User-defined name for debugging.
    root : str or pathlib.Path or None, optional
        Cache root directory. ``None`` uses the default.
    *args : object
        Forwarded to the dataset constructor when ``data`` is a string.
    **kwargs : object
        Forwarded to the dataset constructor when ``data`` is a string.

    Raises
    ------
    tools.UnavailableException
        If ``data`` is an unknown dataset name.
    TypeError
        If constructor arguments are invalid.
    """

    # Default dataset root directory path
    __default_root = None

    @classmethod
    def get_default_root(cls):
        """
        Lazily initialize and return the default dataset cache directory.

        Returns
        -------
        pathlib.Path
            Path to the dataset cache. Falls back to the system temp
            directory if the default does not exist.
        """
        # Fast-path already loaded
        if cls.__default_root is not None:
            return cls.__default_root
        # Generate the default path
        cls.__default_root = pathlib.Path(__file__).parent / "datasets" / "cache"
        # Warn if the path does not exist and fallback to '/tmp'
        if not cls.__default_root.exists():
            tmpdir = tempfile.gettempdir()
            tools.warning(
                f"Default dataset root {str(cls.__default_root)!r} does not exist, "
                f"falling back to local temporary directory {tmpdir!r}",
                context="experiments",
            )
            cls.__default_root = pathlib.Path(tmpdir)
        return cls.__default_root

    # Map 'lower-case names' -> 'dataset class' available in PyTorch
    __datasets = None

    @classmethod
    def _get_datasets(cls):
        """
        Lazily build the name-to-builder mapping for datasets.

        This includes all ``torchvision.datasets`` plus custom datasets
        discovered under ``experiments/datasets/``.

        Returns
        -------
        dict[str, callable]
            Lower-case dataset names mapped to builder functions.
        """
        global transforms
        # Fast-path already loaded
        if cls.__datasets is not None:
            return cls.__datasets
        # Initialize the dictionary
        cls.__datasets = {}
        # Populate with TorchVision's datasets
        for name in dir(torchvision.datasets):
            if len(name) == 0 or name[0] == "_":
                continue
            constructor = getattr(torchvision.datasets, name)
            if isinstance(constructor, type):

                def make_builder(constructor, name):
                    def builder(
                        root,
                        batch_size=None,
                        shuffle=False,
                        num_workers=1,
                        *args,
                        **kwargs,
                    ):
                        data = constructor(root, *args, **kwargs)
                        assert isinstance(data, torch.utils.data.Dataset)
                        if name not in transforms:
                            transforms[name] = torchvision.transforms.ToTensor()
                        batch_size = batch_size or len(data)
                        loader = torch.utils.data.DataLoader(
                            data,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                        )
                        return make_sampler(loader)

                    return builder

                cls.__datasets[name.lower()] = make_builder(constructor, name)

        # Dynamically add custom datasets from subdirectory 'datasets/'
        def add_custom_datasets(name, module, _):
            nonlocal cls
            exports = getattr(module, "__all__", None)
            if exports is None:
                tools.warning(
                    f"Dataset module {name!r} does not provide '__all__'; falling back to '__dict__' for name discovery"
                )
                exports = (n for n in dir(module) if len(n) > 0 and n[0] != "_")
            exported = False
            for dataset in exports:
                if not isinstance(dataset, str):
                    tools.warning(f"Dataset module {name!r} exports non-string name {dataset!r}; ignored")
                    continue
                constructor = getattr(module, dataset, None)
                if not callable(constructor):
                    continue
                exported = True
                fullname = f"{name}-{dataset}"
                if fullname in cls.__datasets:
                    tools.warning(
                        f"Unable to make available dataset {dataset!r} from module "
                        f"{name!r}, as the name {fullname!r} already exists"
                    )
                    continue
                cls.__datasets[fullname] = constructor
            if not exported:
                tools.warning(f"Dataset module {name!r} does not export any valid constructor name through '__all__'")

        with tools.Context("datasets", None):
            tools.import_directory(
                pathlib.Path(__file__).parent / "datasets",
                {"__package__": f"{__package__}.datasets"},
                post=add_custom_datasets,
            )
        return cls.__datasets

    def __init__(self, data, name=None, root=None, *args, **kwargs):
        """
        Initialize the dataset wrapper.

        Parameters
        ----------
        data : str, generator, or object
            Dataset source.
        name : str or None, optional
            Debug name.
        root : str or pathlib.Path or None, optional
            Cache directory for named datasets.
        *args : object
            Forwarded to the dataset constructor.
        **kwargs : object
            Forwarded to the dataset constructor.
        """
        # Handle different dataset types
        if isinstance(data, str):
            if name is None:
                name = data
            datasets = type(self)._get_datasets()
            build = datasets.get(name, None)
            if build is None:
                raise tools.UnavailableException(datasets, name, what="dataset name")
            root = root or type(self).get_default_root()
            self._iter = build(root=root, *args, **kwargs)
        elif isinstance(data, types.GeneratorType):
            if name is None:
                name = "<generator>"
            self._iter = data
        else:
            if name is None:
                name = "<single-batch>"

            def single_batch():
                while True:
                    yield data

            self._iter = single_batch()
        # Finalization
        self.name = name

    def __str__(self):
        """
        Return a printable representation.

        Returns
        -------
        str
            Human-readable dataset name.
        """
        return f"dataset {self.name}"

    def sample(self, config=None):
        """
        Sample the next batch.

        Parameters
        ----------
        config : experiments.Configuration or None, optional
            Target configuration for tensor placement.

        Returns
        -------
        tuple
            Next batch, optionally moved to the target device.
        """
        tns = next(self._iter)
        if config is not None:
            tns = type(tns)(tn.to(device=config["device"], non_blocking=config["non_blocking"]) for tn in tns)
        return tns

    def epoch(self, config=None):
        """
        Return a finite epoch iterator.

        .. note::

           Only works for DataLoader-based datasets.

        Parameters
        ----------
        config : experiments.Configuration or None, optional
            Target configuration for tensor placement.

        Returns
        -------
        generator
            Finite iterator over one epoch.
        """
        assert isinstance(self._loader, torch.utils.data.DataLoader), (
            "Full epoch iteration only possible for PyTorch DataLoader-based datasets"
        )
        epoch = self._loader.__iter__()

        def generator():
            nonlocal epoch
            try:
                while True:
                    tns = next(epoch)
                    if config is not None:
                        tns = type(tns)(
                            tn.to(
                                device=config["device"],
                                non_blocking=config["non_blocking"],
                            )
                            for tn in tns
                        )
                    yield tns
            except StopIteration:
                return

        return generator()


# ---------------------------------------------------------------------------- #
# Dataset helpers


def make_sampler(loader):
    """
    Create an infinite sampler from a DataLoader.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        Finite data loader.

    Yields
    ------
    tuple
        Batches, transparently restarting the loader when exhausted.
    """
    itr = None
    while True:
        for _ in range(2):
            if itr is not None:
                try:
                    yield next(itr)
                    break
                except StopIteration:
                    pass
            itr = iter(loader)
        else:
            raise RuntimeError("Unable to sample a new batch from dataset")


def make_datasets(
    dataset,
    train_batch=None,
    test_batch=None,
    train_transforms=None,
    test_transforms=None,
    num_workers=1,
    **custom_args,
):
    """
    Build training and testing dataset wrappers.

    Parameters
    ----------
    dataset : str
        Case-sensitive dataset name.
    train_batch : int or None, optional
        Training batch size. ``None`` or ``0`` for full-batch.
    test_batch : int or None, optional
        Testing batch size. ``None`` or ``0`` for full-batch.
    train_transforms : callable or None, optional
        Transform for the training set. ``None`` uses the default.
    test_transforms : callable or None, optional
        Transform for the testing set. ``None`` uses the default.
    num_workers : int or tuple[int, int], optional
        Number of workers for the training and testing loaders. An ``int``
        applies to both; a tuple specifies ``(train_workers, test_workers)``.
    **custom_args : object
        Additional keyword arguments forwarded to the dataset constructor.

    Returns
    -------
    tuple[Dataset, Dataset]
        Training and testing dataset wrappers.
    """
    train_transforms = train_transforms or get_default_transform(dataset, True)
    test_transforms = test_transforms or get_default_transform(dataset, False)
    num_workers_errmsg = "Expected either a positive int or a tuple of 2 positive ints for parameter 'num_workers'"
    if isinstance(num_workers, int):
        assert num_workers > 0, num_workers_errmsg
        train_workers = test_workers = num_workers
    else:
        assert isinstance(num_workers, tuple) and len(num_workers) == 2, num_workers_errmsg
        train_workers, test_workers = num_workers
        assert isinstance(train_workers, int) and train_workers > 0, num_workers_errmsg
        assert isinstance(test_workers, int) and test_workers > 0, num_workers_errmsg
    trainset = Dataset(
        dataset,
        train=True,
        download=True,
        batch_size=train_batch,
        shuffle=True,
        num_workers=train_workers,
        transform=train_transforms,
        **custom_args,
    )
    testset = Dataset(
        dataset,
        train=False,
        download=False,
        batch_size=test_batch,
        shuffle=False,
        num_workers=test_workers,
        transform=test_transforms,
        **custom_args,
    )
    return trainset, testset


def batch_dataset(inputs, labels, train=False, batch_size=None, split=0.75):
    """
    Batch a raw tensor dataset into infinite sampler generators.

    Parameters
    ----------
    inputs : torch.Tensor
        Input data tensor.
    labels : torch.Tensor
        Label tensor with the same first-dimension size as ``inputs``.
    train : bool, optional
        Whether to build a training set (adds shuffling) or a test set.
    batch_size : int or None, optional
        Batch size. ``None`` or ``0`` uses the full split size.
    split : float or int, optional
        Fraction of samples for training when ``< 1``, or absolute count
        when ``>= 1``.

    Returns
    -------
    generator
        Infinite sampler generator.
    """

    def train_gen(inputs, labels, batch):
        cursor = 0
        datalen = len(inputs)
        shuffle = list(range(datalen))
        random.shuffle(shuffle)
        while True:
            end = cursor + batch
            if end > datalen:
                select = shuffle[cursor:]
                random.shuffle(shuffle)
                select += shuffle[: (end % datalen)]
            else:
                select = shuffle[cursor:end]
            yield inputs[select], labels[select]
            cursor = end % datalen

    def test_gen(inputs, labels, batch):
        cursor = 0
        datalen = len(inputs)
        while True:
            end = cursor + batch
            if end > datalen:
                select = list(range(cursor, datalen)) + list(range(end % datalen))
                yield inputs[select], labels[select]
            else:
                yield inputs[cursor:end], labels[cursor:end]
            cursor = end % datalen

    dataset_len = len(inputs)
    if dataset_len < 1 or len(labels) != dataset_len:
        raise RuntimeError(
            f"Invalid or different input/output tensor lengths, got "
            f"{len(inputs)} for inputs, got {len(labels)} for labels"
        )
    split_pos = min(
        max(1, int(dataset_len * split)) if split < 1 else split,
        dataset_len - 1,
    )
    if train:
        train_len = split_pos
        batch_size = min(batch_size or train_len, train_len)
        return train_gen(inputs[:split_pos], labels[:split_pos], batch_size)
    test_len = dataset_len - split_pos
    batch_size = min(batch_size or test_len, test_len)
    return test_gen(inputs[split_pos:], labels[split_pos:], batch_size)
