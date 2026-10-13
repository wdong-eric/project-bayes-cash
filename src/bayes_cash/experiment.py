"""Run paired payment experiments and save replayable data and diagnostic figures."""

import argparse
import csv
from dataclasses import asdict, replace
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from .model import PAYMENT_MODES, Config, make_inputs, simulate


SOURCE_COMMIT = "be335d44c009add15388dbdc28b9fb2c42c5fb9e"
COLORS = {"fixed": "#2563a6", "fractional": "#d26416", "hybrid": "#23845b", "no_payment": "#666666"}
LABELS = {"fixed": "Fixed payment", "fractional": "Fraction of wealth", "hybrid": "Max(fixed, fractional)", "no_payment": "No payment"}


def diagnostics(result, theta, true_bias):
    """Measure learning, disagreement, inequality, and realized spending each round."""
    means = result.beliefs @ theta
    variances = np.sum(
        result.beliefs * (theta[None, None, :] - means[:, :, None]) ** 2, axis=2
    )
    wealth = result.wealth
    n = wealth.shape[1]
    # The sorted-sample formula includes agents with zero wealth.
    gini = (
        np.sort(wealth, axis=1) @ (2 * np.arange(1, n + 1) - n - 1)
        / (n * wealth.sum(axis=1))
    )
    return {
        "round": np.arange(len(wealth)),
        "mean_opinion": means.mean(axis=1),
        "opinion_disagreement": means.std(axis=1),
        "mean_posterior_sd": np.sqrt(variances).mean(axis=1),
        "belief_mse": (variances + (means - true_bias) ** 2).mean(axis=1),
        "wealth_gini": gini,
        "wealth_total": wealth.sum(axis=1),
        "wealth_min": wealth.min(axis=1),
        "wealth_median": np.median(wealth, axis=1),
        "wealth_max": wealth.max(axis=1),
        "total_paid": np.r_[0.0, result.payments.sum(axis=1)],
        "fraction_paying": np.r_[0.0, (result.payments > 0).mean(axis=1)],
    }


def run_experiment(config, output):
    """Save three paired payment scenarios for each wealth distribution, plus a control."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; choose a new --output.")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": asdict(config),
        "upstream_url": "https://github.com/YutongB/Opinion-Dynamics",
        "upstream_commit": SOURCE_COMMIT,
        "versions": {name: version(name) for name in ("bayes-cash", "numpy", "networkx", "matplotlib")},
        "conventions": {
            "network": "undirected Barabasi-Albert; fixed symmetric baseline ally/opponent labels; tanh effective signs can change each contact",
            "beliefs": "probability masses on an equally spaced theta grid, including endpoints",
            "cash": "nominal arbitrary units; income_rate * opening wealth injected per round; no normalization of wealth",
            "payments": "one uniformly sampled neighbor per payer; opening wealth; simultaneous income and transfer settlement; when income_rate > 0, skip requests above income",
            "fixed_insufficient_funds": "skip the payment, with no partial payment",
            "hybrid": "pay max(fixed_payment, wealth_fraction * opening_wealth) if affordable; otherwise skip",
            "self_weight": "degree * reference_payment * (1 - social_rate) / social_rate",
            "influence_rule": config.influence_rule,
            "opponent_weight": (
                "recipient opening W * tanh(payment/W - opponent_threshold); ally threshold zero; zero W gives zero weight; relationship sign can change each contact"
                if config.influence_rule == "tanh" else
                "-reference_payment / (1 + payment / reference_payment); unpaid scheduled contacts still repel"
            ),
            "signed_normalization": "positive part followed by probability-mass normalization for recipients of negative effective contact weights",
            "no_payment": "fixed_payment=0; no attraction if all allies, zero-payment repulsion if opponents are present; configured income still injected",
        },
        "results": {},
    }
    wealth_figure, wealth_axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for row, distribution in enumerate(("loguniform", "uniform")):
        inputs = make_inputs(config, distribution)
        results = {
            mode: simulate(config, inputs, mode) for mode in PAYMENT_MODES
        }
        results["no_payment"] = simulate(replace(config, fixed_payment=0.0), inputs, "fixed")
        metrics = {mode: diagnostics(result, inputs.theta, config.true_bias) for mode, result in results.items()}
        for data in metrics.values():
            data["total_income"] = np.r_[0.0, config.income_rate * data["wealth_total"][:-1]]
        payload = {
            "theta": inputs.theta,
            "adjacency": inputs.adjacency,
            "relationships": inputs.relationships,
            "initial_beliefs": inputs.initial_beliefs,
            "initial_wealth": inputs.initial_wealth,
            "observations": inputs.observations,
            "recipients": inputs.recipients,
            "config_json": json.dumps(asdict(config)),
            "upstream_commit": SOURCE_COMMIT,
        }
        for mode, result in results.items():
            payload.update({f"{mode}_beliefs": result.beliefs, f"{mode}_wealth": result.wealth, f"{mode}_payments": result.payments})
            if config.income_rate:
                payload[f"{mode}_income"] = config.income_rate * result.wealth[:-1]
        np.savez_compressed(output / f"{distribution}_trajectories.npz", **payload)
        fields = ["scenario", *metrics["fixed"]]
        with (output / f"{distribution}_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for mode, data in metrics.items():
                for t in range(config.rounds + 1):
                    writer.writerow({"scenario": mode, **{key: values[t] for key, values in data.items()}})

        figure, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout="constrained")
        panels = (
            ("mean_opinion", "Population mean opinion"),
            ("opinion_disagreement", "SD across agents' mean opinions"),
            ("mean_posterior_sd", "Mean within-agent belief SD"),
            ("wealth_gini", "Wealth Gini coefficient"),
            ("total_paid", "Cash transferred per round"),
            ("fraction_paying", "Fraction of agents paying"),
        )
        for axis, (key, label) in zip(axes.flat, panels):
            for mode, data in metrics.items():
                start = 1 if key in ("total_paid", "fraction_paying") else 0
                axis.plot(data["round"][start:], data[key][start:], color=COLORS[mode],
                          linestyle="--" if mode == "no_payment" else "-", label=LABELS[mode], linewidth=1.6)
            axis.set(xlabel="Round", ylabel=label)
            axis.grid(alpha=0.2)
        if config.public_evidence:
            axes[0, 0].axhline(config.true_bias, color="black", linestyle=":", linewidth=1, label="True bias")
        axes[0, 0].legend(fontsize=8)
        axes[1, 2].set_ylim(-0.03, 1.03)
        realized_opponents = float(np.count_nonzero(inputs.relationships < 0) / np.count_nonzero(inputs.adjacency))
        figure.suptitle(
            f"{distribution.capitalize()} initial wealth | N={config.n_agents}, seed={config.seed}\n"
            f"c={config.fixed_payment:g}; f={config.wealth_fraction:g}; opponents={realized_opponents:.1%}; "
            f"{'shared coin evidence' if config.public_evidence else 'no external evidence'}"
            + (f" | {config.influence_rule}; income={config.income_rate:.0%}" if config.influence_rule != "attenuation" or config.income_rate else "")
        )
        figure.savefig(output / f"{distribution}_comparison.png", dpi=160)
        figure.savefig(output / f"{distribution}_comparison.pdf")
        plt.close(figure)

        metadata["results"][distribution] = {
            "initial_total_wealth": float(inputs.initial_wealth.sum()),
            "initial_median_wealth": float(np.median(inputs.initial_wealth)),
            "sampled_heads_fraction": float(inputs.observations.mean()),
            "realized_opponent_edge_fraction": realized_opponents,
            "scenarios": {},
        }
        for mode, result in results.items():
            total = result.wealth.sum(axis=1)
            discounted_total = total * np.exp(-np.arange(total.size) * np.log1p(config.income_rate))
            metadata["results"][distribution]["scenarios"][mode] = {
                "final_mean_opinion": float(metrics[mode]["mean_opinion"][-1]),
                "final_opinion_disagreement": float(metrics[mode]["opinion_disagreement"][-1]),
                "final_belief_mse": float(metrics[mode]["belief_mse"][-1]),
                "initial_gini": float(metrics[mode]["wealth_gini"][0]),
                "final_gini": float(metrics[mode]["wealth_gini"][-1]),
                "cumulative_cash_transferred": float(result.payments.sum()),
                "fraction_of_contact_attempts_paid": float((result.payments > 0).mean()),
                "max_relative_cash_balance_error": float(np.max(np.abs(discounted_total / total[0] - 1.0))),
                "cumulative_nominal_income": float(config.income_rate * total[:-1].sum()),
                "max_belief_normalization_error": float(np.max(np.abs(result.beliefs.sum(axis=2) - 1))),
                "minimum_wealth": float(result.wealth.min()),
            }
            if config.income_rate == 0:
                metadata["results"][distribution]["scenarios"][mode]["max_absolute_cash_conservation_error"] = float(np.max(np.abs(total - total[0])))
        for column, mode in enumerate(PAYMENT_MODES):
            axis = wealth_axes[row, column]
            end = results[mode].wealth[-1]
            scatter = axis.scatter(inputs.initial_wealth, end, c=inputs.adjacency.sum(axis=1), cmap="viridis", s=26, alpha=0.85)
            upper = inputs.initial_wealth.max() * 1.04
            axis.plot([0, upper], [0, upper], linestyle="--", color="gray", linewidth=1)
            axis.set(xlabel="Initial wealth", ylabel="Final wealth", xlim=(0, upper),
                     title=f"{distribution.capitalize()} | {LABELS[mode]}")
            axis.grid(alpha=0.2)
            wealth_figure.colorbar(scatter, ax=axis, label="Network degree")
    wealth_figure.suptitle("Nominal wealth after income and transfers | dashed line: unchanged wealth" if config.income_rate else "Wealth redistribution | dashed line: unchanged wealth")
    wealth_figure.savefig(output / "wealth_redistribution.png", dpi=160)
    wealth_figure.savefig(output / "wealth_redistribution.pdf")
    plt.close(wealth_figure)
    (output / "summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    for name, kind in (("n_agents", int), ("rounds", int), ("attachment", int), ("grid_size", int),
                       ("seed", int), ("true_bias", float), ("social_rate", float), ("reference_payment", float),
                       ("wealth_min", float), ("wealth_max", float), ("fixed_payment", float), ("wealth_fraction", float),
                       ("opponent_fraction", float), ("opponent_threshold", float), ("income_rate", float)):
        parser.add_argument("--" + name.replace("_", "-"), type=kind, default=getattr(Config(), name))
    parser.add_argument("--influence-rule", choices=("attenuation", "tanh"), default=Config().influence_rule)
    parser.add_argument("--no-public-evidence", action="store_true", help="A separate social-only diagnostic; default keeps shared coin learning.")
    args = vars(parser.parse_args())
    output = args.pop("output")
    args["public_evidence"] = not args.pop("no_public_evidence")
    config = Config(**args)
    summary = run_experiment(config, output)
    print(f"Saved paired experiments to {output.resolve()}")
    for distribution, data in summary["results"].items():
        for mode in PAYMENT_MODES:
            item = data["scenarios"][mode]
            print(f"{distribution:10} {mode:10} mean opinion={item['final_mean_opinion']:.5f} "
                  f"Gini={item['final_gini']:.4f} "
                  f"relative cash balance error={item['max_relative_cash_balance_error']:.2e}")


if __name__ == "__main__":
    main()
