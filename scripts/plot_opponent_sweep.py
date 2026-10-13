"""Plot opponent-fraction and late-window comparisons from an existing sweep."""

import argparse
import json
from pathlib import Path

import numpy as np

from bayes_cash.experiment import COLORS, LABELS
from bayes_cash.model import PAYMENT_MODES


DISTRIBUTIONS = ("loguniform", "uniform")
PANELS = (
    ("opinion_disagreement", "SD across agents' mean opinions", "disagreement_sweep"),
    ("belief_mse", "Belief MSE about true bias", "belief_accuracy_sweep"),
)


def plot_sweep(output):
    """Save comparable seed-mean curves with sample-SD bands, without rerunning data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    output = Path(output)
    summary = json.loads((output / "summary.json").read_text())
    fractions = np.asarray(summary["fractions"])
    horizons = summary["horizons"]
    groups = {
        (row["distribution"], row["mode"], row["opponent_fraction"], row["horizon"]): row
        for row in summary["groups"]
    }
    legend = [Line2D([], [], color=COLORS[mode], marker="o", markersize=4,
                     label=LABELS[mode]) for mode in PAYMENT_MODES]
    config = summary["config"]
    context = (f"{len(summary['seeds'])} paired seeds; N={config['n_agents']}; "
               f"c={config['fixed_payment']:g}, f={config['wealth_fraction']:g}")
    uncertainty = "Lines: seed means; bands: ±1 sample SD, not confidence intervals. Nonnegative metric range shown."
    saved = []

    for metric, label, name in PANELS:
        figure, axes = plt.subplots(2, len(horizons), squeeze=False,
                                    figsize=(5 * len(horizons), 8), sharex=True, sharey=True)
        ymax = 0.0
        for row_index, distribution in enumerate(DISTRIBUTIONS):
            for column, horizon in enumerate(horizons):
                axis = axes[row_index, column]
                for mode in PAYMENT_MODES:
                    values = [groups[distribution, mode, float(p), horizon][metric]
                              for p in fractions]
                    means = np.asarray([value["mean"] for value in values])
                    sd = np.asarray([value["sample_sd"] for value in values])
                    axis.plot(fractions, means, color=COLORS[mode], marker="o",
                              markersize=3.5, linewidth=1.8)
                    axis.fill_between(fractions, means - sd, means + sd,
                                      color=COLORS[mode], alpha=0.11, linewidth=0)
                    ymax = max(ymax, float(np.max(means + sd)))
                axis.set_title(f"Round {horizon:,}", fontsize=12)
                axis.grid(alpha=0.18)
                axis.set(xlim=(0, 1), xticks=np.linspace(0, 1, 6))
                axis.xaxis.set_major_formatter(PercentFormatter(1))
                if row_index == 1:
                    axis.set_xlabel("Probability an edge is an opponent relationship")
                if column == 0:
                    axis.set_ylabel(f"{distribution.capitalize()} initial wealth\n{label}")
                if metric == "belief_mse":
                    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        axes[0, 0].set_ylim(0, ymax * 1.06 if ymax else 1)
        title = ("Disagreement across opponent fractions" if metric == "opinion_disagreement"
                 else "Belief error across opponent fractions | smaller is better")
        figure.suptitle(f"{title}\n{context}", fontsize=15, y=0.987)
        figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.908),
                      ncol=3, frameon=False, fontsize=10)
        figure.text(0.5, 0.018, uncertainty, ha="center", fontsize=9, color="#444444")
        figure.subplots_adjust(left=0.085, right=0.982, bottom=0.10, top=0.835,
                               wspace=0.12, hspace=0.26)
        for extension in ("png", "pdf"):
            path = output / f"{name}.{extension}"
            figure.savefig(path, dpi=160)
            saved.append(path)
        plt.close(figure)

    windows = sorted({(row["window_start"], row["window_end"])
                      for row in summary["late_window_groups"]})
    if len(windows) != 2:
        raise ValueError("Late-persistence figure requires exactly two comparison windows.")
    window_groups = {
        (row["distribution"], row["mode"], row["opponent_fraction"],
         row["window_start"], row["window_end"]): row
        for row in summary["late_window_groups"]
    }
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey="col")
    for column, (metric, label, _) in enumerate(PANELS):
        ymax = 0.0
        for row_index, distribution in enumerate(DISTRIBUTIONS):
            axis = axes[row_index, column]
            for mode in PAYMENT_MODES:
                for window_index, (start, end) in enumerate(windows):
                    values = [window_groups[distribution, mode, float(p), start, end][metric]
                              for p in fractions]
                    means = np.asarray([value["mean"] for value in values])
                    sd = np.asarray([value["sample_sd"] for value in values])
                    axis.plot(fractions, means, color=COLORS[mode], linewidth=1.8,
                              linestyle="--" if window_index == 0 else "-",
                              marker=None if window_index == 0 else "o", markersize=3.5)
                    axis.fill_between(fractions, means - sd, means + sd, color=COLORS[mode],
                                      alpha=0.045 if window_index == 0 else 0.10, linewidth=0)
                    ymax = max(ymax, float(np.max(means + sd)))
            axis.set(title=f"{distribution.capitalize()} initial wealth", ylabel=label,
                     xlim=(0, 1), xticks=np.linspace(0, 1, 6))
            axis.xaxis.set_major_formatter(PercentFormatter(1))
            axis.grid(alpha=0.18)
            if row_index == 1:
                axis.set_xlabel("Probability an edge is an opponent relationship")
            if metric == "belief_mse":
                axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        axes[0, column].set_ylim(0, ymax * 1.06 if ymax else 1)
    window_legend = [Line2D([], [], color="#222222", linestyle=style,
                            label=f"Rounds {start:,}–{end:,}")
                     for (start, end), style in zip(windows, ("--", "-"))]
    figure.suptitle(f"Does late disagreement persist?\n{context}", fontsize=15, y=0.985)
    figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.912),
                  ncol=3, frameon=False, fontsize=9.5)
    figure.legend(handles=window_legend, loc="upper center", bbox_to_anchor=(0.5, 0.877),
                  ncol=2, frameon=False, fontsize=9.5)
    figure.text(0.5, 0.020, "Each seed is averaged over the indicated rounds, then seeds are compared.\n"
                "Bands: ±1 sample SD, not confidence intervals. Nonnegative metric range shown.",
                ha="center", fontsize=9, color="#444444")
    figure.subplots_adjust(left=0.09, right=0.978, bottom=0.115, top=0.808,
                           wspace=0.24, hspace=0.27)
    for extension in ("png", "pdf"):
        path = output / f"late_persistence.{extension}"
        figure.savefig(path, dpi=160)
        saved.append(path)
    plt.close(figure)

    final_horizon = max(horizons)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey="col")
    support_panels = (
        ("truth_excluded_fraction", "Agents with zero probability at truth"),
        ("absorbing_disagreement_floor", "Guaranteed future disagreement floor"),
    )
    for column, (metric, label) in enumerate(support_panels):
        ymax = 0.0
        for row_index, distribution in enumerate(DISTRIBUTIONS):
            axis = axes[row_index, column]
            for mode in PAYMENT_MODES:
                values = [groups[distribution, mode, float(p), final_horizon][metric]
                          for p in fractions]
                means = np.asarray([value["mean"] for value in values])
                sd = np.asarray([value["sample_sd"] for value in values])
                axis.plot(fractions, means, color=COLORS[mode], marker="o",
                          markersize=3.5, linewidth=1.8)
                axis.fill_between(fractions, means - sd, means + sd,
                                  color=COLORS[mode], alpha=0.11, linewidth=0)
                ymax = max(ymax, float(np.max(means + sd)))
            axis.set(title=f"{distribution.capitalize()} initial wealth", ylabel=label,
                     xlim=(0, 1), xticks=np.linspace(0, 1, 6))
            axis.xaxis.set_major_formatter(PercentFormatter(1))
            axis.grid(alpha=0.18)
            if row_index == 1:
                axis.set_xlabel("Probability an edge is an opponent relationship")
            if metric == "truth_excluded_fraction":
                axis.yaxis.set_major_formatter(PercentFormatter(1))
        axes[0, column].set_ylim(0, 1 if column == 0 else ymax * 1.06 if ymax else 1)
    figure.suptitle(f"Support loss and certified persistent disagreement | round {final_horizon:,}\n"
                   f"{context}", fontsize=15, y=0.985)
    figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.909),
                  ncol=3, frameon=False, fontsize=10)
    figure.text(0.5, 0.021,
                "Left: current loss of truth support; allied contacts can restore it.\n"
                "Right: per-seed mathematical bound from interior singleton agents with no allied neighbors.\n"
                "Lines: seed means; bands: ±1 sample SD, not confidence intervals. Nonnegative metric range shown.",
                ha="center", fontsize=9, color="#444444")
    figure.subplots_adjust(left=0.09, right=0.978, bottom=0.137, top=0.833,
                           wspace=0.24, hspace=0.27)
    for extension in ("png", "pdf"):
        path = output / f"absorbing_support.{extension}"
        figure.savefig(path, dpi=160)
        saved.append(path)
    plt.close(figure)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for path in plot_sweep(args.output):
        print(path.resolve())
