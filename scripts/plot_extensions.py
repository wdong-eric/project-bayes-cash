"""Plot the paired tanh-influence and income comparison from saved results."""

import argparse
import json
from pathlib import Path

import numpy as np

from bayes_cash.experiment import COLORS, LABELS
from bayes_cash.model import PAYMENT_MODES


DISTRIBUTIONS = ("loguniform", "uniform")
ARMS = ("baseline", "tanh_only", "income_only", "both")
ARM_LABELS = ("Baseline", "Tanh only", "Income only", "Both")
BELIEF_METRICS = (("opinion_disagreement", "SD across agents' mean opinions"),
                  ("belief_mse", "Belief MSE about true bias"))


def plot_extensions(output):
    """Save belief, cash, and effective-sign plots without repeating simulations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    output = Path(output)
    summary = json.loads((output / "summary.json").read_text())
    fractions = summary["fractions"]
    horizons = summary["horizons"]
    groups = {(row["distribution"], row["mode"], row["opponent_fraction"], row["arm"], row["horizon"]): row
              for row in summary["groups"]}
    config = summary["config"]
    context = (f"{len(summary['seeds'])} paired seeds; N={config['n_agents']}; "
               f"income={summary['income_rate']:.0%} of opening wealth per round when enabled")
    colors_legend = [Line2D([], [], color=COLORS[mode], marker="o", markersize=4,
                            label=LABELS[mode]) for mode in PAYMENT_MODES]
    uncertainty = "Markers: seed means; bars: ±1 sample SD, not confidence intervals. Nonnegative metric range shown."
    maxima = {key: max(row[key]["mean"] + row[key]["sample_sd"] for row in summary["groups"])
              for key, _ in BELIEF_METRICS}
    saved = []
    for horizon in horizons:
        figure, axes = plt.subplots(2, 2 * len(fractions), squeeze=False,
                                    figsize=(8.5 * len(fractions), 8.2))
        for row_index, distribution in enumerate(DISTRIBUTIONS):
            for fraction_index, fraction in enumerate(fractions):
                for metric_index, (key, label) in enumerate(BELIEF_METRICS):
                    axis = axes[row_index, 2 * fraction_index + metric_index]
                    for mode_index, mode in enumerate(PAYMENT_MODES):
                        values = [groups[distribution, mode, fraction, arm, horizon][key]
                                  for arm in ARMS]
                        axis.errorbar(np.arange(len(ARMS)) + (mode_index - 1) * 0.19,
                                      [value["mean"] for value in values],
                                      yerr=[value["sample_sd"] for value in values],
                                      fmt="o", color=COLORS[mode], markersize=5,
                                      capsize=3, elinewidth=1.1)
                    axis.set(title=f"{fraction:.0%} baseline opponents", ylabel=label,
                             xticks=np.arange(len(ARMS)), xticklabels=ARM_LABELS,
                             xlim=(-0.5, len(ARMS) - 0.5),
                             ylim=(0, 1.07 * maxima[key] if maxima[key] else 1))
                    axis.tick_params(axis="x", labelsize=9, rotation=16)
                    axis.grid(axis="y", alpha=0.18)
                    if metric_index == 0:
                        axis.set_ylabel(f"{distribution.capitalize()} wealth\n{label}")
                    else:
                        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        figure.suptitle(f"Belief outcomes with tanh influence and income | round {horizon:,}\n{context}",
                       fontsize=15, y=0.985)
        figure.legend(handles=colors_legend, loc="upper center", bbox_to_anchor=(0.5, 0.913),
                      ncol=3, frameon=False, fontsize=10)
        figure.text(0.5, 0.019, uncertainty, ha="center", fontsize=9, color="#444444")
        figure.subplots_adjust(left=0.062, right=0.982, bottom=0.115, top=0.83,
                               wspace=0.31, hspace=0.35)
        for extension in ("png", "pdf"):
            path = output / f"belief_comparison_{horizon}.{extension}"
            figure.savefig(path, dpi=160)
            saved.append(path)
        plt.close(figure)

    with np.load(output / "trajectories.npz") as trajectories:
        rounds = trajectories["round"]
        figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey="col")
        for row_index, distribution in enumerate(DISTRIBUTIONS):
            for column, (key, label) in enumerate((("wealth_gini", "Wealth Gini coefficient"),
                                                 ("fraction_paying", "Fraction of contact attempts paid"))):
                axis = axes[row_index, column]
                start = int(key == "fraction_paying")
                for mode in PAYMENT_MODES:
                    for arm, style in (("baseline", "--"), ("income_only", "-")):
                        # Cash is independent of baseline relationship signs and influence law.
                        values = trajectories[f"{distribution}_{mode}_{arm}_{key}"][:, 0, start:]
                        means, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
                        axis.plot(rounds[start:], means, color=COLORS[mode], linestyle=style, linewidth=1.8)
                        axis.fill_between(rounds[start:], means - sd, means + sd,
                                          color=COLORS[mode], alpha=0.06, linewidth=0)
                axis.set(title=f"{distribution.capitalize()} initial wealth", ylabel=label,
                         xlim=(0, rounds[-1]), ylim=(0, 1.025))
                axis.grid(alpha=0.18)
                if key == "fraction_paying":
                    axis.yaxis.set_major_formatter(PercentFormatter(1))
                if row_index == 1:
                    axis.set_xlabel("Round")
        line_legend = [Line2D([], [], color="#222222", linestyle="--", label="No income"),
                       Line2D([], [], color="#222222", label="Income enabled")]
        figure.suptitle(f"Income changes cash circulation\n{context}", fontsize=15, y=0.985)
        figure.legend(handles=colors_legend, loc="upper center", bbox_to_anchor=(0.5, 0.912),
                      ncol=3, frameon=False, fontsize=10)
        figure.legend(handles=line_legend, loc="upper center", bbox_to_anchor=(0.5, 0.874),
                      ncol=2, frameon=False, fontsize=10)
        figure.text(0.5, 0.018,
                    "Cash paths are identical across influence laws and baseline opponent fractions.\n"
                    "Lines: seed means; bands: ±1 sample SD, not confidence intervals. Valid metric range shown.",
                    ha="center", fontsize=9, color="#444444")
        figure.subplots_adjust(left=0.09, right=0.978, bottom=0.112, top=0.806,
                               wspace=0.24, hspace=0.27)
        for extension in ("png", "pdf"):
            path = output / f"income_cash.{extension}"
            figure.savefig(path, dpi=160)
            saved.append(path)
        plt.close(figure)

        figure, axes = plt.subplots(2, len(fractions), squeeze=False,
                                    figsize=(6.5 * len(fractions), 8.5), sharex=True, sharey=True)
        for row_index, distribution in enumerate(DISTRIBUTIONS):
            for column, fraction in enumerate(fractions):
                axis = axes[row_index, column]
                for mode in PAYMENT_MODES:
                    for arm, style in (("tanh_only", "--"), ("both", "-")):
                        values = trajectories[f"{distribution}_{mode}_{arm}_effective_opponent_flip_fraction"][:, column, 1:]
                        means, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
                        axis.plot(rounds[1:], means, color=COLORS[mode], linestyle=style, linewidth=1.8)
                        axis.fill_between(rounds[1:], means - sd, means + sd,
                                          color=COLORS[mode], alpha=0.06, linewidth=0)
                axis.set(title=f"{distribution.capitalize()} wealth | {fraction:.0%} baseline opponents",
                         xlim=(0, rounds[-1]), ylim=(0, 1.025))
                axis.yaxis.set_major_formatter(PercentFormatter(1))
                axis.grid(alpha=0.18)
                if column == 0:
                    axis.set_ylabel("Opponent contacts becoming attractive")
                if row_index == 1:
                    axis.set_xlabel("Round")
        line_legend = [Line2D([], [], color="#222222", linestyle="--", label="Tanh only"),
                       Line2D([], [], color="#222222", label="Tanh + income")]
        figure.suptitle(f"Payments can turn opponent contacts into attraction\n{context}",
                       fontsize=15, y=0.985)
        figure.legend(handles=colors_legend, loc="upper center", bbox_to_anchor=(0.5, 0.912),
                      ncol=3, frameon=False, fontsize=10)
        figure.legend(handles=line_legend, loc="upper center", bbox_to_anchor=(0.5, 0.874),
                      ncol=2, frameon=False, fontsize=10)
        figure.text(0.5, 0.018,
                    "Fraction among scheduled baseline-opponent contacts with strictly positive influence; signs change per contact.\n"
                    "Lines: seed means; bands: ±1 sample SD, not confidence intervals. Valid fraction range shown.",
                    ha="center", fontsize=9, color="#444444")
        figure.subplots_adjust(left=0.095, right=0.978, bottom=0.112, top=0.806,
                               wspace=0.16, hspace=0.27)
        for extension in ("png", "pdf"):
            path = output / f"contact_signs.{extension}"
            figure.savefig(path, dpi=160)
            saved.append(path)
        plt.close(figure)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for path in plot_extensions(args.output):
        print(path.resolve())
