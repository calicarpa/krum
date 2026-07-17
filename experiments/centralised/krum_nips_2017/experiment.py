"""MultiKrum vs Mean under sign-flip attack on the Spambase dataset.

MultiKrum resists Byzantine workers while the coordinate-wise mean diverges,
reproducing the centralised setting of Blanchard et al. (Section 2).
"""

import matplotlib.pyplot as plt
import pandas as pd

from krum.orchestration import Orchestrator
from krum.orchestration.dataframe import MetricDataFrame
from krum.primitives.aggregators.average import Average
from krum.primitives.aggregators.multikrum import MultiKrum
from krum.primitives.attacks.sign_flip import SignFlipAttack
from krum.primitives.models.mlp import Krum2017MLPSpambase

from .run import krum_experiment

ROUNDS = 300
MODEL = Krum2017MLPSpambase
DATASET = "spambase"
N = 20
F = N // 3
BATCH_SIZE = 3
LR = 0.01
SEED = 42
EVAL_EVERY = 15
XAVIER_INIT = True
WEIGHT_DECAY = 1e-4


def _plot_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    styles: dict,
    ylabel: str,
    title: str,
    *,
    exclude: str | None = None,
) -> None:
    """Plot a single panel of the comparison between MultiKrum and Mean under sign-flip attack."""
    for run_label, group in frame.groupby("label", sort=False):
        if run_label == exclude:
            continue
        style = styles.get(run_label)
        if style is None:
            continue
        group = group.sort_values("step")
        ax.plot(group["step"], group["value"], label=run_label, **style, linewidth=1.5)
    ax.set_xlabel("round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, linestyle=":", alpha=0.5)


def plot_comparison(
    test_loss: MetricDataFrame,
    test_accuracy: MetricDataFrame,
    train_loss: MetricDataFrame,
    *,
    f_byz: int,
) -> None:
    """Plot the comparison of MultiKrum and Mean under sign-flip attack."""
    frame_tl = test_loss.to_pandas()
    frame_ta = test_accuracy.to_pandas()
    frame_trl = train_loss.to_pandas()

    styles = {
        "Mean_f0": {"color": "tab:green", "linestyle": "-"},
        f"Mean_f{f_byz}": {"color": "tab:red", "linestyle": "--"},
        "MultiKrum_f0": {"color": "tab:orange", "linestyle": "-"},
        f"MultiKrum_f{f_byz}": {"color": "tab:blue", "linestyle": "--"},
    }

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    mean_f_label = f"Mean_f{f_byz}"

    _plot_panel(fig.add_subplot(gs[0, 0]), frame_tl, styles, "loss", "test loss")
    _plot_panel(fig.add_subplot(gs[0, 1]), frame_tl, styles, "loss", "test loss (excl. Mean_f6)", exclude=mean_f_label)
    _plot_panel(fig.add_subplot(gs[1, 0]), frame_trl, styles, "loss", "train loss")
    _plot_panel(
        fig.add_subplot(gs[1, 1]), frame_trl, styles, "loss", "train loss (excl. Mean_f6)", exclude=mean_f_label
    )
    _plot_panel(fig.add_subplot(gs[2, :]), frame_ta, styles, "accuracy", "test accuracy")
    fig.axes[-1].set_ylim(0.0, 1.0)

    fig.suptitle("MultiKrum vs Mean — sign-flip attack", fontsize=12)
    plt.show()


def main() -> None:
    """Run the experiment and plot the results."""
    orchestrator = Orchestrator("krum_2017_nips_spambase")

    configs = [
        (Average, "Mean_f0", 0, None),
        (Average, f"Mean_f{F}", F, None),
        (MultiKrum, "MultiKrum_f0", 0, {"m": N - 2}),
        (MultiKrum, f"MultiKrum_f{F}", F, {"m": N - F - 2}),
    ]

    for agg, label, f_val, agg_kw in configs:
        orchestrator.run(
            krum_experiment,
            label=label,
            dataset=DATASET,
            model_cls=MODEL,
            aggregator=agg,
            aggregator_kwargs=agg_kw,
            attack=SignFlipAttack,
            attack_kwargs={"scale": 10.0},
            n=N,
            f=f_val,
            rounds=ROUNDS,
            batch_size=BATCH_SIZE,
            lr=LR,
            seed=SEED,
            eval_every=EVAL_EVERY,
            xavier_init=XAVIER_INIT,
            weight_decay=WEIGHT_DECAY,
        )

    print("\nDone.")
    plot_comparison(
        orchestrator.get("test_loss"),
        orchestrator.get("test_accuracy"),
        orchestrator.get("train_loss"),
        f_byz=F,
    )


if __name__ == "__main__":
    main()
