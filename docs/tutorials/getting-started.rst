Getting started
===============

This tutorial will take you through your first end-to-end experiment with
Krum. You will train a small convolutional network on MNIST twice:

1. First with a simple baseline aggregator and no attack.
2. Then with a Byzantine attack and a robust aggregator.

By the end, you will have produced two training logs and seen how a
Byzantine-resilient rule keeps accuracy stable under attack.

.. note::

   This tutorial assumes you have Python 3.8+ and PyTorch installed. If you
   plan to use CUDA, ensure your PyTorch build matches your CUDA version.


Before you start
----------------

1. Clone the repository::

       git clone https://github.com/calicarpa/krum.git
       cd krum

2. Install the dependencies. The project only requires PyTorch and
   torchvision::

       pip install torch torchvision

3. Verify that the training script is available::

       python train.py --help

   You should see a long list of command-line options. That confirms the
   environment is ready.


Step 1: Train a clean baseline
------------------------------

Run a short training session with the ``average`` aggregator and no actual
Byzantine workers. This gives you a reference point.

.. code-block:: bash

    python train.py \
        --nb-workers 11 \
        --nb-decl-byz 4 \
        --nb-real-byz 0 \
        --gar average \
        --attack nan \
        --nb-steps 300 \
        --evaluation-delta 100 \
        --result-directory ./results/baseline

What to expect
~~~~~~~~~~~~~~

After a few seconds you will see output similar to this::

    Accuracy (step 0)... 9.82%.
    Training... done.
    Accuracy (step 100)... 85.40%.
    Training... done.
    Accuracy (step 200)... 91.23%.
    Training... done.
    Accuracy (step 300)... 93.15%.

The accuracy climbs steadily because every worker is honest and the
aggregator simply averages their gradients.

A folder ``results/baseline`` has been created. It contains::

    results/baseline/
    ├── config          # Human-readable configuration
    ├── config.json     # Machine-readable configuration
    ├── eval            # Tab-separated step / accuracy values
    └── study           # Detailed training metrics

Open ``results/baseline/eval`` in a text editor. You will see one line per
evaluation milestone::

    # Step number	Cross-accuracy
    0	0.0982
    100	0.8540
    200	0.9123
    300	0.9315

Keep this file open. You will compare it with the next run.


Step 2: Train with Krum under attack
------------------------------------

Now run the same experiment, but this time activate four real Byzantine
workers that send ``NaN`` gradients, and defend with the ``krum`` aggregator.

.. code-block:: bash

    python train.py \
        --nb-workers 11 \
        --nb-decl-byz 4 \
        --nb-real-byz 4 \
        --gar krum \
        --attack nan \
        --nb-steps 300 \
        --evaluation-delta 100 \
        --result-directory ./results/krum_nan

What to expect
~~~~~~~~~~~~~~

The console output will look similar to the baseline::

    Accuracy (step 0)... 9.82%.
    Training... done.
    Accuracy (step 100)... 82.15%.
    Training... done.
    Accuracy (step 200)... 89.67%.
    Training... done.
    Accuracy (step 300)... 92.04%.

Accuracy still increases. Open ``results/krum_nan/eval``::

    # Step number	Cross-accuracy
    0	0.0982
    100	0.8215
    200	0.8967
    300	0.9204

For comparison, try the same attack with ``average`` instead of ``krum``::

    python train.py \
        --nb-workers 11 \
        --nb-decl-byz 4 \
        --nb-real-byz 4 \
        --gar average \
        --attack nan \
        --nb-steps 300 \
        --evaluation-delta 100 \
        --result-directory ./results/average_nan

The accuracy will collapse because ``average`` has no defense against
``NaN`` gradients. The ``eval`` file will show near-random accuracy after
the first evaluation.


Step 3: Compare the runs
------------------------

You now have three result directories::

    results/
    ├── baseline/      # Average, no attack
    ├── krum_nan/      # Krum, NaN attack
    └── average_nan/   # Average, NaN attack

Plotting the ``eval`` files side by side makes the difference visible::

    import matplotlib.pyplot as plt
    import numpy as np

    def load_eval(path):
        data = np.loadtxt(path, skiprows=1)
        return data[:, 0], data[:, 1]

    fig, ax = plt.subplots()

    for name, directory in [
        ("Average, no attack", "baseline"),
        ("Krum, NaN attack", "krum_nan"),
        ("Average, NaN attack", "average_nan"),
    ]:
        steps, acc = load_eval(f"results/{directory}/eval")
        ax.plot(steps, acc * 100, label=name, marker="o")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend()
    ax.grid(True)
    plt.savefig("comparison.png")
    print("Saved comparison.png")

Run the script::

    pip install matplotlib
    python plot_comparison.py

Open ``comparison.png``. You should see the baseline and the Krum curve
rising together, while the ``average_nan`` curve stays flat near 10%.


What to do next
---------------

- Try a different robust aggregator::

      python train.py ... --gar bulyan

- Try a different attack::

      python train.py ... --attack identical --attack-args variant:little

- Read the :doc:`How-to guides </how-to/index>` to learn how to add your own aggregation rule or attack.

- Read the :doc:`Explanation </explanation/key-concepts>` section to understand the registration system and why returned tensors must not alias inputs.
