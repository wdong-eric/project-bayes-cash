"""Compare signed and allies-only networks using identical cash and evidence paths."""

import argparse
import csv
from dataclasses import asdict, replace
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from bayes_cash.experiment import COLORS, LABELS, diagnostics
from bayes_cash.model import PAYMENT_MODES, Config, make_inputs, simulate
from _legacy_scope import require_static_study


SCENARIOS = (*PAYMENT_MODES, "no_payment")
ARMS = ("allies", "opponents")
METRICS = ("mean_opinion", "opinion_disagreement", "belief_mse", "mean_posterior_sd", "wealth_gini")
PANELS = (("opinion_disagreement", "SD across agents' mean opinions"),
          ("belief_mse", "Mean belief squared error about true bias"))


def paired_results(config, inputs, mode):
    """Change edge signs alone and require the autonomous cash paths to match."""
    require_static_study(config)
    current = replace(config, fixed_payment=0.0) if mode == "no_payment" else config
    payment_mode = "fixed" if mode == "no_payment" else mode
    ally_inputs = replace(inputs, relationships=inputs.adjacency.astype(int))
    ally = simulate(replace(current, opponent_fraction=0.0), ally_inputs, payment_mode)
    mixed = simulate(current, inputs, payment_mode)
    np.testing.assert_array_equal(ally.wealth, mixed.wealth)
    np.testing.assert_array_equal(ally.payments, mixed.payments)
    return {"allies": ally, "opponents": mixed}


def run_comparison(config, seed_count, output):
    """Save paired outcomes, sample-SD comparisons, and replayable random inputs."""
    require_static_study(config)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if seed_count < 2:
        raise ValueError("at least two seeds are required for a sample SD")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seeds = list(range(config.seed, config.seed + seed_count))
    rows, histories, payload = [], {}, {}
    for distribution in ("loguniform", "uniform"):
        for seed in seeds:
            current = replace(config, seed=seed)
            inputs = make_inputs(current, distribution)
            opponent_fraction = np.count_nonzero(inputs.relationships < 0) / np.count_nonzero(inputs.adjacency)
            for name in ("adjacency", "relationships", "theta", "initial_beliefs", "observations", "recipients"):
                payload[f"seed_{seed}_{name}"] = getattr(inputs, name)
            payload[f"{distribution}_{seed}_initial_wealth"] = inputs.initial_wealth
            for mode in SCENARIOS:
                for arm, result in paired_results(current, inputs, mode).items():
                    metrics = diagnostics(result, inputs.theta, current.true_bias)
                    row = {
                        "distribution": distribution, "seed": seed, "arm": arm, "mode": mode,
                        "realized_opponent_fraction": opponent_fraction if arm == "opponents" else 0.0,
                        "cumulative_cash_transferred": float(result.payments.sum()),
                        "max_absolute_cash_error": float(np.max(np.abs(metrics["wealth_total"] - metrics["wealth_total"][0]))),
                    }
                    for key in METRICS:
                        values = metrics[key]
                        if not np.isfinite(values).all():
                            raise ValueError(f"Nonfinite {key}: {distribution}, seed {seed}, {arm}, {mode}")
                        histories.setdefault(f"{distribution}_{arm}_{mode}_{key}", []).append(values)
                        row[key] = float(values[-1])
                    rows.append(row)
            print(f"Finished {distribution}, seed {seed}", flush=True)
    with (output / "per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "config": asdict(config), "seeds": seeds,
        "versions": {name: version(name) for name in ("bayes-cash", "numpy", "networkx", "matplotlib")},
        "conventions": {
            "pairing": "identical topology, initial beliefs, initial wealth, evidence, and contacts across relationship arms and payment modes",
            "opponents": "independent negative undirected edges with probability opponent_fraction; realized fraction recorded per seed",
            "signed_weights": "ally: payment; opponent: -reference_payment / (1 + payment / reference_payment); denominator: self_weight + incoming absolute weights; positive-part renormalization",
            "no_payment": "allies: evidence only; opponents: evidence plus undamped repulsion on sampled opponent contacts",
            "cash": "wealth and payments verified bitwise identical across relationship arms for every mode and seed",
            "uncertainty": "sample SD across paired seeds, not confidence intervals",
            "belief_mse": "mean over agents of posterior variance plus squared posterior-mean error about the true coin bias",
            "limitations": "finite horizon on one network family, one opponent probability, and unequal spending between payment rules",
        },
        "groups": [], "paired_contrasts": [],
        "max_absolute_cash_error": max(row["max_absolute_cash_error"] for row in rows),
    }
    for distribution in ("loguniform", "uniform"):
        for mode in SCENARIOS:
            for arm in ARMS:
                subset = [row for row in rows if (row["distribution"], row["mode"], row["arm"]) == (distribution, mode, arm)]
                summary["groups"].append({
                    "distribution": distribution, "mode": mode, "arm": arm,
                    **{key: {"mean": float(np.mean([row[key] for row in subset])),
                             "sample_sd": float(np.std([row[key] for row in subset], ddof=1))}
                       for key in (*METRICS, "realized_opponent_fraction", "cumulative_cash_transferred")},
                })
            contrasts = {}
            for key in METRICS:
                difference = (np.asarray(histories[f"{distribution}_opponents_{mode}_{key}"])[:, -1]
                              - np.asarray(histories[f"{distribution}_allies_{mode}_{key}"])[:, -1])
                contrasts[key] = {"mean": float(difference.mean()), "sample_sd": float(difference.std(ddof=1))}
            summary["paired_contrasts"].append({"distribution": distribution, "mode": mode,
                                                "contrast": "opponents_minus_allies", **contrasts})

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    trajectory, time_axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for row_index, distribution in enumerate(("loguniform", "uniform")):
        for column, (key, label) in enumerate(PANELS):
            axis, time_axis = axes[row_index, column], time_axes[row_index, column]
            for index, mode in enumerate(SCENARIOS):
                for arm_index, arm in enumerate(ARMS):
                    values = np.asarray(histories[f"{distribution}_{arm}_{mode}_{key}"])
                    means, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
                    axis.errorbar(index + (arm_index - 0.5) * 0.2, means[-1], yerr=sd[-1],
                                  fmt="o", color=COLORS[mode], markerfacecolor="white" if arm == "allies" else COLORS[mode],
                                  capsize=4, markersize=7)
                    rounds = np.arange(config.rounds + 1)
                    time_axis.plot(rounds, means, color=COLORS[mode], linestyle="--" if arm == "allies" else "-",
                                   linewidth=1.5)
                    time_axis.fill_between(rounds, means - sd, means + sd, color=COLORS[mode], alpha=0.055)
            axis.set(xticks=np.arange(len(SCENARIOS)), xticklabels=("Fixed", "Fractional", "Max(c, fW)", "No payment"),
                     ylabel=label, title=f"{distribution.capitalize()} wealth | round {config.rounds}")
            time_axis.set(xlabel="Round", ylabel=label, title=f"{distribution.capitalize()} initial wealth")
            axis.grid(axis="y", alpha=0.2)
            time_axis.grid(alpha=0.2)
    legend = [Line2D([], [], color="black", marker="o", linestyle="none", markerfacecolor="white", label="All allies"),
              Line2D([], [], color="black", marker="o", linestyle="none", label="With opponents")]
    axes[0, 0].legend(handles=legend, fontsize=9)
    time_legend = [Line2D([], [], color=COLORS[mode], label=LABELS[mode]) for mode in SCENARIOS]
    time_legend += [Line2D([], [], color="black", linestyle="--", label="All allies"),
                   Line2D([], [], color="black", label="With opponents")]
    time_axes[0, 0].legend(handles=time_legend, fontsize=8, ncol=2)
    title = f"Opponent comparison | {seed_count} paired seeds, N={config.n_agents}; opponent probability={config.opponent_fraction:g}"
    figure.suptitle(title + "\nMarkers: mean; bars: sample SD; no-payment opponents retain repulsion")
    trajectory.suptitle(title + "\nLines: mean; ribbons: ±sample SD; no-payment opponents retain repulsion")
    for plot, name in ((figure, "opponent_comparison"), (trajectory, "opponent_trajectories")):
        plot.savefig(output / f"{name}.png", dpi=160)
        plot.savefig(output / f"{name}.pdf")
        plt.close(plot)
    payload.update({key: np.asarray(values) for key, values in histories.items()})
    payload["metadata_json"] = json.dumps(summary, allow_nan=False)
    np.savez_compressed(output / "opponent_comparison.npz", **payload)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/opponent_comparison"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--opponent-fraction", type=float, default=0.2)
    args = parser.parse_args()
    run_comparison(Config(seed=args.seed, rounds=args.rounds, opponent_fraction=args.opponent_fraction), args.seeds, args.output)
