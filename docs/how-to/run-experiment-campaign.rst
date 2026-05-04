How to run an experiment campaign
===================================

An experiment campaign is a batch of training runs with different
hyper-parameters. This guide shows how to use the ``tools.jobs`` utilities to
launch and manage campaigns.

Step 1: Define the parameter grid
---------------------------------

Create a Python script that defines the combinations you want to test::

    from tools import Command, Jobs

    # Base command
    base = ["python", "train.py", "--nb-workers", "11", "--nb-decl-byz", "4"]

    # Parameter grid
    gars = ["average", "krum", "bulyan"]
    attacks = ["nan", "identical"]

    commands = []
    for gar in gars:
        for attack in attacks:
            cmd = Command(base + ["--gar", gar, "--attack", attack])
            commands.append(cmd)

Step 2: Launch the jobs
-----------------------

Create a ``Jobs`` instance and submit the commands::

    jobs = Jobs("./results/campaign", devices=["cuda:0", "cuda:1"])

    for cmd in commands:
        jobs.submit(cmd)

    jobs.run()

The ``Jobs`` class handles queueing, device allocation, and result
directory creation automatically.

Step 3: Analyse the results
---------------------------

After the campaign finishes, collect the evaluation files::

    import pathlib
    import numpy as np

    results = []
    for path in pathlib.Path("./results/campaign").rglob("eval"):
        data = np.loadtxt(path, skiprows=1)
        results.append({
            "path": str(path),
            "final_accuracy": data[-1, 1],
        })

Sort by final accuracy to identify the best configuration.
