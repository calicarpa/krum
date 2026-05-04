###
# @file   __init__.py
# @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
#
# @section LICENSE
#
# Copyright © 2018-2021 École Polytechnique Fédérale de Lausanne (EPFL).
# See LICENSE file.
#
# @section DESCRIPTION
#
# Dataset/model/... wrappers/helpers, for more convenient gradient extraction and operations.
# Heavily relies on the module 'torchvision'.
###

"""
Experiment components for model training, dataset loading, and evaluation.

This module groups the building blocks of a Krum training loop:

- :mod:`experiments.configuration` — device and dtype configuration.
- :mod:`experiments.model` — model wrapper with parameter flattening.
- :mod:`experiments.dataset` — dataset loading and infinite batch sampling.
- :mod:`experiments.loss` — derivable loss composition.
- :mod:`experiments.optimizer` — optimizer wrapper with LR control.
- :mod:`experiments.checkpoint` — save and restore state dictionaries.

Custom models and datasets can be added under ``experiments/models/`` and
``experiments/datasets/``; they are discovered automatically at import time.

Example
-------

.. code-block:: python

    from experiments import (
        Configuration, Model, Dataset,
        Loss, Criterion, Optimizer,
        make_datasets,
    )

    config = Configuration(device="cuda:0")
    model = Model("resnet18", config, num_classes=10)
    trainset, testset = make_datasets("cifar10", train_batch=128)
"""

import pathlib

import tools

# ---------------------------------------------------------------------------- #
# Load all local modules

with tools.Context("experiments", None):
    tools.import_directory(pathlib.Path(__file__).parent, globals())
