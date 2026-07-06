"""Shared plotting helpers for the NIPS 2017 (Blanchard et al.) experiments.

Reproduces the figures of Blanchard, El Mhamdi, Guerraoui, Stainer, "Machine
learning with adversaries: Byzantine tolerant gradient descent", NIPS 2017.

The paper reports **misclassification error** (not accuracy) on the test set,
so all helpers below read from the ``"error"`` channel.
"""

from typing import Any

import matplotlib.pyplot as plt

from krum.orchestration.dataframe import MetricDataFrame

_LEGEND_STYLE = {
    ("Average", 0): {"color": "tab:purple", "linestyle": "-", "linewidth": 2.0},
    ("Average", "byz"): {"color": "tab:blue", "linestyle": "--", "linewidth": 2.0},
    ("Krum", 0): {"color": "tab:green", "linestyle": "-", "linewidth": 2.0},
    ("Krum", "byz"): {"color": "tab:green", "linestyle": "--", "linewidth": 2.0},
    ("MultiKrum", "byz"): {"color": "tab:blue", "linestyle": "--", "linewidth": 2.0},
}


def _extract_aggregator_and_f(run_label: str) -> tuple[str, int] | None:
    """Parse a run label of the form ``"{agg}_f{f}[...]"``.

    Handles two label shapes emitted by the experiments:
    - ``"{agg}_f{f}"`` (e.g. ``"Average_f0"``, ``"Krum_f6"``)
    - ``"{ds}_{agg}_f{f}[_bs{bs}]"`` (e.g. ``"spambase_Average_f0_bs3"``)

    Returns:
        A ``(aggregator_name, f)`` pair, or ``None`` if the label does not
        match the expected pattern. The dataset prefix and the optional
        ``_bs{...}`` suffix are stripped before matching.
    """
    # Strip the optional dataset prefix (e.g. "spambase_", "mnist_").
    for prefix in ("spambase_", "mnist_"):
        if run_label.startswith(prefix):
            run_label = run_label[len(prefix) :]
            break
    # Strip the optional "_bs{...}" suffix.
    if "_bs" in run_label:
        run_label = run_label.split("_bs", 1)[0]

    # Match "{agg}_f{f}" where {agg} may contain underscores.
    idx = run_label.find("_f")
    if idx < 0 or not run_label[idx + 2 :].isdigit():
        return None
    agg_name = run_label[:idx]
    try:
        f = int(run_label[idx + 2 :])
    except ValueError:
        return None
    if not agg_name:
        return None
    return agg_name, f


def plot_error_curves_by_f(error_data: MetricDataFrame) -> None:
    """Reproduce Figure 4 of Blanchard et al. (NIPS 2017).

    Plots test error for Spambase under Average and Krum, side-by-side: the
    left subplot shows ``f=0`` (honest), the right subplot shows ``f=6``
    (33% Gaussian Byzantine).

    The y-axis is auto-scaled with a 5% headroom margin so curves at the
    bottom of the [0, 1] range are still readable. To use a fixed range
    (e.g. ``(0, 1)`` to match the paper verbatim), pass ``ylim=...``.
    """
    frame = error_data.to_pandas()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, f_target, title_suffix in (
        (axes[0], 0, "0% byzantine"),
        (axes[1], 6, "33% byzantine"),
    ):
        plotted_handles: list[plt.Line2D] = []
        # Track the global min/max across runs in this subplot so we can
        # pick a tight ylim with a small headroom.
        values_min, values_max = float("inf"), float("-inf")
        for run_label, group in frame.groupby("label", sort=False):
            # The dataset prefix (e.g. "spambase_") is optional: experiment_1
            # emits "Average_f0", experiment_2 emits "spambase_Average_f0_bs3".
            # The _extract_aggregator_and_f helper handles both.
            parsed = _extract_aggregator_and_f(run_label)
            if parsed is None:
                continue
            agg_name, f_value = parsed
            if f_value != f_target:
                continue
            key: tuple[str, Any] = (agg_name, 0 if f_value == 0 else "byz")
            style = _LEGEND_STYLE.get(key)
            if style is None:
                continue
            group = group.sort_values("step")
            (line,) = ax.plot(
                group["step"],
                group["value"],
                label=agg_name.lower(),
                **style,
            )
            plotted_handles.append(line)
            values_min = min(values_min, float(group["value"].min()))
            values_max = max(values_max, float(group["value"].max()))

        ax.set_title(title_suffix, fontsize=11)
        ax.set_xlabel("round")
        # Tight y-range with 5% headroom above and below; clamp to [0, 1]
        # since misclassification error is a rate.
        if plotted_handles:
            margin = max(0.05 * (values_max - values_min), 0.01)
            ax.set_ylim(max(0.0, values_min - margin), min(1.0, values_max + margin))
        else:
            ax.set_ylim(0.0, 1.0)
        ax.set_xlim(left=0)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(handles=plotted_handles, loc="upper right", fontsize=9)

    axes[0].set_ylabel("error")
    fig.suptitle(
        "Figure 4 — Cross-validation error evolution with rounds (Spambase)",
        fontsize=12,
    )
    fig.tight_layout()
    plt.show()


def plot_error_vs_batch_size(error_data: MetricDataFrame) -> None:
    """Reproduce Figure 5 of Blanchard et al. (NIPS 2017).

    Plots test error at round 500 (y-axis) as a function of the mini-batch
    size (x-axis) for Spambase and MNIST, with Average and Krum under
    ``f=0`` and ``f=9`` (45% Omniscient Byzantine) configurations. The
    right half of the figure shows zoomed-in versions of the left half.

    The paper reports the error at the **last** available sample. We
    therefore pick the last sample per run.
    """
    frame = error_data.to_pandas()
    batch_sizes = [3, 5, 10, 20, 40, 80, 160]
    datasets = ["spambase", "mnist"]

    # Each dataset has two views: full range (left) and zoomed (right).
    # The paper's Spambase zoom focuses on batch sizes [3, 40]; the MNIST
    # zoom covers [3, 40] as well.
    zoom_window: dict[str, tuple[int, int]] = {
        "spambase": (3, 40),
        "mnist": (3, 40),
    }
    # Y-axis range for the zoomed views: capped at 0.5 to mimic the paper.
    zoom_ylim = (0.0, 0.5)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # Collect one (dataset_prefix, view) → list of (agg, f, bs, error) points.
    runs_by_view: dict[tuple[str, str], list[tuple[str, int, int, float]]] = {}
    for run_label, group in frame.groupby("label", sort=False):
        for dataset_prefix in datasets:
            if not run_label.startswith(dataset_prefix):
                continue
            parsed = _extract_aggregator_and_f(run_label)
            if parsed is None:
                continue
            agg_name, f_value = parsed
            if "_bs" not in run_label:
                continue
            try:
                bs = int(run_label.rsplit("_bs", 1)[1])
            except ValueError:
                continue
            if bs not in batch_sizes:
                continue
            last = group.sort_values("step").iloc[-1]
            runs_by_view.setdefault((dataset_prefix, "full"), []).append((agg_name, f_value, bs, float(last["value"])))
            runs_by_view.setdefault((dataset_prefix, "zoom"), []).append((agg_name, f_value, bs, float(last["value"])))
            break  # dataset_prefix matched; don't double-count under the other dataset

    view_specs: list[tuple[Any, str, str, tuple[int, int] | None, tuple[float, float] | None]] = [
        (axes[0], "spambase", "full", None, None),
        (axes[1], "mnist", "full", None, None),
        (axes[2], "spambase", "zoom", zoom_window["spambase"], zoom_ylim),
        (axes[3], "mnist", "zoom", zoom_window["mnist"], zoom_ylim),
    ]

    for ax, dataset_prefix, view, xlim, ylim in view_specs:
        plotted_handles: list[plt.Line2D] = []
        for (agg_name, f_value), group_pts in _group_by_agg_f(runs_by_view.get((dataset_prefix, view), [])):
            key: tuple[str, Any] = (agg_name, 0 if f_value == 0 else "byz")
            style = _LEGEND_STYLE.get(key)
            if style is None:
                continue
            group_pts.sort(key=lambda p: p[2])  # sort by batch size
            xs = [p[2] for p in group_pts]
            ys = [p[3] for p in group_pts]
            label = f"average ({f_value}% byz)" if agg_name == "Average" else f"krum ({f_value}% byz)"
            (line,) = ax.plot(
                xs,
                ys,
                marker="o",
                label=label,
                **style,
            )
            plotted_handles.append(line)

        ax.set_xscale("log")
        if xlim is not None:
            ax.set_xlim(*xlim)
        else:
            ax.set_xlim(min(batch_sizes), max(batch_sizes))
        ax.set_xticks(batch_sizes)
        ax.set_xticklabels([str(bs) for bs in batch_sizes])
        ax.set_xlabel("batch size")
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0.0, 1.0)
        ax.set_title(dataset_prefix, fontsize=11)
        ax.set_ylabel("error at round 500")
        ax.grid(True, linestyle=":", alpha=0.5)
        if (dataset_prefix, view) == ("spambase", "full"):
            ax.legend(handles=plotted_handles, loc="upper right", fontsize=7)

    fig.suptitle(
        "Figure 5 — Cross-validation error at round 500 vs mini-batch size",
        fontsize=12,
    )
    fig.tight_layout()
    plt.show()


def _group_by_agg_f(
    pts: list[tuple[str, int, int, float]],
) -> list[tuple[tuple[str, int], list[tuple[str, int, int, float]]]]:
    """Group a flat list of ``(agg, f, bs, err)`` points by ``(agg, f)``."""
    out: dict[tuple[str, int], list[tuple[str, int, int, float]]] = {}
    for p in pts:
        out.setdefault((p[0], p[1]), []).append(p)
    return list(out.items())


def plot_error_curves_multi_krum(error_data: MetricDataFrame) -> None:
    """Reproduce Figure 6 of Blanchard et al. (NIPS 2017).

    Plots test error on Spambase for three configurations: Average (f=0,
    honest baseline), Krum (f=6, 33% Gaussian Byzantine), and Multi-Krum
    (f=6, 33% Gaussian Byzantine, ``m = n - f``).

    The y-axis is auto-scaled with a 5% headroom margin so curves at the
    bottom of the [0, 1] range are still readable.
    """
    frame = error_data.to_pandas()
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_handles: list[plt.Line2D] = []
    values_min, values_max = float("inf"), float("-inf")

    for run_label, group in frame.groupby("label", sort=False):
        # No dataset prefix expected (experiment_3 emits "Average_f0", etc.).
        parsed = _extract_aggregator_and_f(run_label)
        if parsed is None:
            continue
        agg_name, f_value = parsed
        key: tuple[str, Any] = (agg_name, 0 if f_value == 0 else "byz")
        style = _LEGEND_STYLE.get(key)
        if style is None:
            continue
        group = group.sort_values("step")
        # Legend text matches the paper's "average (0% byz)" / "krum (33% byz)".
        pretty_name = "average" if agg_name == "Average" else agg_name.lower().replace("_", "-")
        label = f"{pretty_name} ({f_value}% byz)"
        (line,) = ax.plot(
            group["step"],
            group["value"],
            label=label,
            **style,
        )
        plotted_handles.append(line)
        values_min = min(values_min, float(group["value"].min()))
        values_max = max(values_max, float(group["value"].max()))

    ax.set_xlabel("round")
    ax.set_ylabel("error")
    if plotted_handles:
        margin = max(0.05 * (values_max - values_min), 0.01)
        ax.set_ylim(max(0.0, values_min - margin), min(1.0, values_max + margin))
    else:
        ax.set_ylim(0.0, 1.0)
    ax.set_xlim(left=0)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(handles=plotted_handles, loc="upper right", fontsize=9)
    ax.set_title("multi-krum", fontsize=11)
    fig.suptitle(
        "Figure 6 — Cross-validation error evolution with rounds (Spambase)",
        fontsize=12,
    )
    fig.tight_layout()
    plt.show()
