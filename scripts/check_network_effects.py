"""Separate payment-rule and topology effects with paired, cash-only simulations.

Cash is autonomous: E[delta wealth_i | wealth] = sum_j A_ij p_j/k_j - p_i.
Fractional payments have stationary mean wealth proportional to degree, whereas
fixed payments cap the rich agents' outflow. These are finite-horizon controls,
not a proof of stationarity or a perfectly isolated intervention on hubs.
"""

import argparse
import csv
from dataclasses import asdict, replace
from importlib.metadata import version
import json
from pathlib import Path

import networkx as nx
import numpy as np

from bayes_cash.experiment import COLORS, LABELS
from bayes_cash.model import PAYMENT_MODES, Config, make_inputs, payment_amounts
from _legacy_scope import require_static_study


TOPOLOGIES = ("barabasi_albert", "regular", "complete")
TOPOLOGY_LABELS = ("BA hubs", "Random regular", "Complete")


def control_inputs(config, distribution):
    """Keep the original 500-round BA experiment and append new random draws."""
    require_static_study(config)
    short_rounds = min(500, config.rounds)
    original = make_inputs(replace(config, rounds=short_rounds), distribution)
    # Cash-only controls use fixed contacts and forbid preferential giving.
    original = replace(original, contact_uniforms=None)
    degree = 2 * config.attachment
    if degree >= config.n_agents:
        raise ValueError("random-regular degree 2 * attachment must be less than n_agents")
    prepared = {}
    for index, topology in enumerate(TOPOLOGIES):
        graph_rng, contact_rng, evidence_rng = (
            np.random.default_rng(stream)
            for stream in np.random.SeedSequence([config.seed, 136045, index]).spawn(3)
        )
        if topology == "barabasi_albert":
            adjacency = original.adjacency
        elif topology == "regular":
            # Connectivity is part of the regular-network control's definition.
            while True:
                graph = nx.random_regular_graph(
                    degree, config.n_agents,
                    seed=int(graph_rng.integers(0, np.iinfo(np.int32).max)),
                )
                if nx.is_connected(graph):
                    break
            adjacency = nx.to_numpy_array(graph, nodelist=range(config.n_agents), dtype=bool)
        else:
            adjacency = ~np.eye(config.n_agents, dtype=bool)
        recipients = np.empty((config.rounds, config.n_agents), dtype=int)
        if topology == "barabasi_albert":
            recipients[:short_rounds] = original.recipients
        else:
            for payer in range(config.n_agents):
                recipients[:short_rounds, payer] = contact_rng.choice(
                    np.flatnonzero(adjacency[payer]), size=short_rounds,
                )
        for payer in range(config.n_agents):
            recipients[short_rounds:, payer] = contact_rng.choice(
                np.flatnonzero(adjacency[payer]), size=config.rounds - short_rounds,
            )
        observations = np.r_[
            original.observations,
            evidence_rng.random(config.rounds - short_rounds) < config.true_bias,
        ]
        prepared[topology] = replace(
            original, adjacency=adjacency, relationships=adjacency.astype(np.int8),
            recipients=recipients, observations=observations,
        )
    return prepared


def simulate_cash(config, inputs, mode):
    """Replay core synchronous transfers without allocating irrelevant beliefs."""
    require_static_study(config)
    wealth = np.empty((config.rounds + 1, config.n_agents))
    payments = np.empty((config.rounds, config.n_agents))
    wealth[0] = inputs.initial_wealth
    for step, recipients in enumerate(inputs.recipients):
        payments[step] = payment_amounts(config, wealth[step], mode)
        incoming = np.zeros(config.n_agents)
        np.add.at(incoming, recipients, payments[step])
        wealth[step + 1] = wealth[step] - payments[step] + incoming
    np.testing.assert_allclose(wealth.sum(axis=1), wealth[0].sum(), rtol=1e-12)
    assert np.all(wealth >= 0), "negative cash balance"
    return wealth, payments


def run_controls(config, seed_count, output):
    """Write per-seed outcomes, paired contrasts, reproducible inputs, and figures."""
    require_static_study(config)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if seed_count < 2:
        raise ValueError("at least two seeds are required for a sample SD")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    horizons = sorted({min(500, config.rounds), config.rounds})
    seeds = list(range(config.seed, config.seed + seed_count))
    rows, histories, payload = [], {}, {}
    ranks = 2 * np.arange(1, config.n_agents + 1) - config.n_agents - 1
    for distribution in ("loguniform", "uniform"):
        for seed in seeds:
            current = replace(config, seed=seed)
            inputs_by_topology = control_inputs(current, distribution)
            payload[f"{distribution}_{seed}_initial_wealth"] = inputs_by_topology["barabasi_albert"].initial_wealth
            for topology, inputs in inputs_by_topology.items():
                degree = inputs.adjacency.sum(axis=1)
                payload[f"{topology}_{seed}_adjacency"] = inputs.adjacency
                for mode in PAYMENT_MODES:
                    wealth, payments = simulate_cash(current, inputs, mode)
                    total = wealth.sum(axis=1)
                    gini = np.sort(wealth, axis=1) @ ranks / (config.n_agents * total)
                    histories.setdefault(f"{distribution}_{topology}_{mode}", []).append(gini)
                    for horizon in horizons:
                        final = wealth[horizon]
                        rows.append({
                            "distribution": distribution, "seed": seed, "topology": topology,
                            "mode": mode, "horizon": horizon, "initial_gini": float(gini[0]),
                            "gini": float(gini[horizon]), "mean_degree": float(degree.mean()),
                            "top_decile_share": float(np.sort(final)[-int(np.ceil(config.n_agents / 10)):].sum() / total[0]),
                            "degree_wealth_correlation": float(np.corrcoef(degree, final)[0, 1]) if np.ptp(degree) else None,
                            "mean_payment_per_agent_round": float(payments[:horizon].mean()),
                            "fraction_contact_attempts_paid": float((payments[:horizon] > 0).mean()),
                            "max_absolute_cash_error": float(np.max(np.abs(total[:horizon + 1] - total[0]))),
                        })
                        payload[f"{distribution}_{seed}_{topology}_{mode}_wealth_{horizon}"] = final
            print(f"Finished {distribution}, seed {seed}", flush=True)
    with (output / "per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "config": asdict(config), "seeds": seeds, "horizons": horizons,
        "versions": {name: version(name) for name in ("bayes-cash", "numpy", "networkx", "matplotlib")},
        "conventions": {
            "cash_only": "beliefs cannot change contacts or payments; exact cash subsystem",
            "pairing": "identical initial wealth across topology and mode; contacts shared across modes",
            "topologies": "BA attachment m; connected random regular degree 2m; complete graph",
            "randomness": "original make_inputs prefix through round 500; independent seeded continuation",
            "uncertainty": "sample SD across seeds, not confidence intervals",
            "limitations": "finite horizons; regular and BA mean degrees are close but unequal; complete graph also changes density",
            "correlation": "degree-wealth Pearson correlation undefined and omitted for regular/complete graphs",
        },
        "groups": [], "paired_contrasts": [],
    }
    figure, axes = plt.subplots(2, len(horizons), figsize=(6.3 * len(horizons), 8), squeeze=False, layout="constrained")
    for row_index, distribution in enumerate(("loguniform", "uniform")):
        for column, horizon in enumerate(horizons):
            axis = axes[row_index, column]
            paired = {}
            for mode_index, mode in enumerate(PAYMENT_MODES):
                means, deviations = [], []
                for topology in TOPOLOGIES:
                    subset = [row for row in rows if (row["distribution"], row["horizon"], row["mode"], row["topology"]) == (distribution, horizon, mode, topology)]
                    values = np.array([row["gini"] for row in subset])
                    paired[topology, mode] = values
                    metrics = {}
                    for key in ("initial_gini", "gini", "top_decile_share", "degree_wealth_correlation", "mean_payment_per_agent_round", "fraction_contact_attempts_paid"):
                        if subset[0][key] is not None:
                            data = [row[key] for row in subset]
                            metrics[key] = {"mean": float(np.mean(data)), "sample_sd": float(np.std(data, ddof=1))}
                    summary["groups"].append({"distribution": distribution, "horizon": horizon, "topology": topology, "mode": mode, **metrics})
                    means.append(values.mean())
                    deviations.append(values.std(ddof=1))
                axis.errorbar(np.arange(3) + (mode_index - 1) * 0.18, means, yerr=deviations,
                              fmt="o", capsize=4, color=COLORS[mode], label=LABELS[mode])
            for control in ("regular", "complete"):
                contrasts = {f"{mode}_BA_minus_control": paired["barabasi_albert", mode] - paired[control, mode] for mode in PAYMENT_MODES}
                contrasts["fixed_minus_fractional_interaction"] = contrasts["fixed_BA_minus_control"] - contrasts["fractional_BA_minus_control"]
                summary["paired_contrasts"].append({
                    "distribution": distribution, "horizon": horizon, "control": control,
                    **{key: {"mean": float(value.mean()), "sample_sd": float(value.std(ddof=1))} for key, value in contrasts.items()},
                })
            axis.set(xticks=np.arange(3), xticklabels=TOPOLOGY_LABELS, ylabel="Wealth Gini", ylim=(-0.02, 1.02),
                     title=f"{distribution.capitalize()} initial wealth | round {horizon}")
            axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=9, loc="lower left")
    figure.suptitle(f"Network controls | {seed_count} paired seeds, N={config.n_agents}\nMarkers: mean; bars: sample SD across seeds")
    figure.savefig(output / "network_controls.png", dpi=160)
    figure.savefig(output / "network_controls.pdf")
    plt.close(figure)
    for key, values in histories.items():
        payload[f"{key}_gini_mean"] = np.mean(values, axis=0)
        payload[f"{key}_gini_sample_sd"] = np.std(values, axis=0, ddof=1)
    summary["max_absolute_cash_error"] = max(row["max_absolute_cash_error"] for row in rows)
    payload["metadata_json"] = json.dumps(summary)
    np.savez_compressed(output / "network_controls.npz", **payload)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/network_controls"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5000)
    args = parser.parse_args()
    run_controls(Config(rounds=args.rounds), args.seeds, args.output)
