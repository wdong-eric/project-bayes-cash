"""Sweep signed-edge probabilities with paired, prefix-preserving long simulations."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from bayes_cash.experiment import diagnostics
from bayes_cash.model import PAYMENT_MODES, Config, make_inputs, simulate
from _legacy_scope import require_static_study


DISTRIBUTIONS = ("loguniform", "uniform")
METRICS = (
    "mean_opinion", "opinion_disagreement", "belief_mse", "mean_posterior_sd",
    "wealth_gini", "mean_posterior_variance", "posterior_mean_mse",
    "truth_excluded_fraction", "mean_truth_mass", "singleton_fraction",
    "frozen_fraction", "absorbing_disagreement_floor",
)
PREFIX_ROUNDS = 500
CONTINUATION_NAMESPACE = 0x53574545


def make_sweep_inputs(config, distribution):
    """Keep the old 500 rounds exactly, then append horizon-stable random streams.

    The old generator samples contacts payer by payer, so changing its horizon
    changes all but the first payer's schedule. Long runs therefore retain its
    500-round input verbatim and use separate evidence/contact seed namespaces
    afterwards. Continuation contacts use time-major uniform draws, making all
    prefixes beyond round 500 independent of the requested final horizon.
    """
    require_static_study(config)
    prefix = make_inputs(replace(config, rounds=min(PREFIX_ROUNDS, config.rounds)), distribution)
    extra = config.rounds - prefix.observations.size
    if extra == 0:
        return prefix
    evidence_rng = np.random.default_rng(np.random.SeedSequence([config.seed, CONTINUATION_NAMESPACE, 1]))
    contact_rng = np.random.default_rng(np.random.SeedSequence([config.seed, CONTINUATION_NAMESPACE, 2]))
    uniforms = contact_rng.random((extra, config.n_agents))
    contacts = np.empty((extra, config.n_agents), dtype=int)
    for payer in range(config.n_agents):
        neighbors = np.flatnonzero(prefix.adjacency[payer])
        contacts[:, payer] = neighbors[(uniforms[:, payer] * neighbors.size).astype(int)]
    return replace(
        prefix,
        observations=np.concatenate((prefix.observations, evidence_rng.random(extra) < config.true_bias)),
        recipients=np.concatenate((prefix.recipients, contacts)),
    )


def sweep_diagnostics(result, theta, true_bias, relationships):
    """Separate accuracy of posterior means, posterior width, and lost support."""
    truth = np.flatnonzero(np.isclose(theta, true_bias, rtol=0.0, atol=1e-14))
    if truth.size != 1:
        raise ValueError("truth-support diagnostics require true_bias on exactly one theta grid point")
    values = diagnostics(result, theta, true_bias)
    means = result.beliefs @ theta
    variances = np.sum(result.beliefs * (theta[None, None, :] - means[:, :, None]) ** 2, axis=2)
    truth_mass = result.beliefs[:, :, truth[0]]
    singleton = np.count_nonzero(result.beliefs, axis=2) == 1
    # Interior singleton beliefs cannot move under evidence or purely repulsive
    # contacts. Their mutual spread bounds all future population disagreement.
    frozen = singleton & ~np.any(relationships > 0, axis=1)[None, :] & (means > 0) & (means < 1)
    frozen_count = frozen.sum(axis=1)
    frozen_mean = (means * frozen).sum(axis=1) / np.maximum(frozen_count, 1)
    frozen_spread = (frozen * (means - frozen_mean[:, None]) ** 2).sum(axis=1)
    values.update({
        "mean_posterior_variance": variances.mean(axis=1),
        "posterior_mean_mse": ((means - true_bias) ** 2).mean(axis=1),
        "truth_excluded_fraction": (truth_mass == 0.0).mean(axis=1),
        "mean_truth_mass": truth_mass.mean(axis=1),
        "singleton_fraction": singleton.mean(axis=1),
        "frozen_fraction": frozen.mean(axis=1),
        "absorbing_disagreement_floor": np.sqrt(frozen_spread / means.shape[1]),
    })
    return values


def run_trajectory(config, inputs, mode, horizons, stride=50, chunk_size=500):
    """Run the existing model in bounded chunks, retaining exact late-round means."""
    require_static_study(config)
    if config.rounds < 4 or stride < 1 or chunk_size < 1:
        raise ValueError("rounds must be at least four; stride and chunk_size must be positive")
    horizons = tuple(sorted(set(int(t) for t in horizons)))
    if not horizons or horizons[0] < 1 or horizons[-1] > config.rounds:
        raise ValueError("horizons must be nonempty and lie between one and config.rounds")
    rounds = np.unique(np.r_[np.arange(0, config.rounds + 1, stride), horizons, config.rounds])
    histories = {key: np.empty(rounds.size) for key in METRICS}
    windows = ((config.rounds // 2 + 1, 3 * config.rounds // 4),
               (3 * config.rounds // 4 + 1, config.rounds))
    window_sums = [{key: 0.0 for key in METRICS} for _ in windows]
    checkpoint_rows = []
    wealth_hash, payments_hash = hashlib.sha256(), hashlib.sha256()
    initial_total = inputs.initial_wealth.sum()
    max_cash_error, cumulative_paid = 0.0, 0.0
    beliefs, wealth = inputs.initial_beliefs, inputs.initial_wealth
    for start in range(0, config.rounds, chunk_size):
        stop = min(start + chunk_size, config.rounds)
        current = replace(
            inputs, initial_beliefs=beliefs, initial_wealth=wealth,
            observations=inputs.observations[start:stop], recipients=inputs.recipients[start:stop],
        )
        result = simulate(replace(config, rounds=stop - start), current, mode)
        values = sweep_diagnostics(result, inputs.theta, config.true_bias, inputs.relationships)
        if any(not np.isfinite(values[key]).all() for key in METRICS):
            raise ValueError(f"nonfinite diagnostics during rounds {start + 1} to {stop}")
        local_rounds = np.arange(start, stop + 1)
        selected = np.flatnonzero((rounds >= start + (start > 0)) & (rounds <= stop))
        for key in METRICS:
            histories[key][selected] = values[key][rounds[selected] - start]
            for index, (first, last) in enumerate(windows):
                mask = (local_rounds >= max(first, start + 1)) & (local_rounds <= last)
                window_sums[index][key] += float(values[key][mask].sum())
        paid = np.r_[0.0, np.cumsum(result.payments.sum(axis=1))] + cumulative_paid
        for horizon in horizons:
            if start < horizon <= stop:
                checkpoint_rows.append({
                    "horizon": horizon,
                    **{key: float(values[key][horizon - start]) for key in METRICS},
                    "cumulative_cash_transferred": float(paid[horizon - start]),
                })
        cumulative_paid = float(paid[-1])
        max_cash_error = max(max_cash_error, float(np.max(np.abs(values["wealth_total"] - initial_total))))
        wealth_hash.update(result.wealth[1:].tobytes())
        payments_hash.update(result.payments.tobytes())
        beliefs, wealth = result.beliefs[-1].copy(), result.wealth[-1].copy()
    late_rows = [
        {"window_start": first, "window_end": last,
         **{key: sums[key] / (last - first + 1) for key in METRICS}}
        for (first, last), sums in zip(windows, window_sums)
    ]
    return {
        "rounds": rounds, "histories": histories, "checkpoint_rows": checkpoint_rows,
        "late_window_rows": late_rows,
        "cash_digest": {"wealth_sha256": wealth_hash.hexdigest(), "payments_sha256": payments_hash.hexdigest()},
        "max_absolute_cash_error": max_cash_error,
    }


def _run_job(job):
    config, distribution, mode, fractions, horizons, stride, chunk_size = job
    inputs = make_sweep_inputs(config, distribution)
    results, cash_digest = [], None
    for fraction in fractions:
        current = replace(config, opponent_fraction=fraction)
        signs = make_inputs(replace(current, rounds=min(PREFIX_ROUNDS, config.rounds)), distribution).relationships
        try:
            result = run_trajectory(current, replace(inputs, relationships=signs), mode, horizons, stride, chunk_size)
            if cash_digest is not None and result["cash_digest"] != cash_digest:
                raise ValueError("relationship signs changed the wealth or payment trajectory")
        except Exception as error:
            raise RuntimeError(f"{distribution}, seed={config.seed}, mode={mode}, opponent_fraction={fraction}: {error}") from error
        cash_digest = result["cash_digest"]
        result["opponent_fraction"] = fraction
        result["realized_opponent_fraction"] = float(np.count_nonzero(signs < 0) / np.count_nonzero(inputs.adjacency))
        results.append(result)
    return distribution, config.seed, mode, results


def _group_rows(rows, keys, metrics):
    """Summarize independent seeds; bars are sample SD, not standard errors."""
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    summaries = []
    for group, subset in groups.items():
        summary = dict(zip(keys, group))
        for metric in metrics:
            values = np.asarray([row[metric] for row in subset], dtype=float)
            scale = np.max(np.abs(values))
            if scale == 0:
                summary[metric] = {"mean": 0.0, "sample_sd": 0.0}
            else:
                # Income can make finite nominal cash exceed sqrt(float_max).
                # Scale before summing or squaring, preserving the original units.
                scaled = values / scale
                summary[metric] = {
                    "mean": float(scale * np.mean(scaled)),
                    "sample_sd": float(scale * np.std(scaled, ddof=1)),
                }
        summaries.append(summary)
    return summaries


def run_sweep(config, seed_count, output, probabilities=None, horizons=(500, 5000, 20000),
              stride=50, chunk_size=500, workers=4):
    """Save paired sweep metrics, replayable inputs, cash hashes, and run metadata."""
    require_static_study(config)
    fractions = np.linspace(0.0, 1.0, 11) if probabilities is None else np.asarray(probabilities, dtype=float)
    if fractions.ndim != 1 or fractions.size < 1 or not np.isfinite(fractions).all() or np.any((fractions < 0) | (fractions > 1)):
        raise ValueError("opponent fractions must be a finite one-dimensional sequence in [0, 1]")
    if not np.all(np.diff(fractions) > 0):
        raise ValueError("opponent fractions must be unique and increasing")
    if seed_count < 2 or workers < 1:
        raise ValueError("at least two seeds and one worker are required")
    horizons = tuple(sorted(set((*horizons, config.rounds))))
    if config.rounds < 4 or horizons[0] < 1 or horizons[-1] > config.rounds or stride < 1 or chunk_size < 1:
        raise ValueError("invalid horizon, stride, or chunk size")
    if np.count_nonzero(np.isclose(np.linspace(0, 1, config.grid_size), config.true_bias, rtol=0.0, atol=1e-14)) != 1:
        raise ValueError("truth-support diagnostics require true_bias on exactly one theta grid point")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seeds = list(range(config.seed, config.seed + seed_count))
    jobs = [(replace(config, seed=seed), distribution, mode, fractions.tolist(), horizons, stride, chunk_size)
            for distribution in DISTRIBUTIONS for seed in seeds for mode in PAYMENT_MODES]
    rows, late_rows, trajectories, hashes = [], [], {}, []
    max_cash_error = 0.0
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        completed = executor.map(_run_job, jobs) if executor is not None else map(_run_job, jobs)
        for distribution, seed, mode, results in completed:
            for result in results:
                identity = {"distribution": distribution, "seed": seed, "mode": mode,
                            "opponent_fraction": result["opponent_fraction"],
                            "realized_opponent_fraction": result["realized_opponent_fraction"]}
                rows.extend({**identity, **row} for row in result["checkpoint_rows"])
                late_rows.extend({**identity, **row} for row in result["late_window_rows"])
                max_cash_error = max(max_cash_error, result["max_absolute_cash_error"])
            hashes.append({"distribution": distribution, "seed": seed, "mode": mode, **results[0]["cash_digest"]})
            for key in METRICS:
                name = f"{distribution}_{mode}_{key}"
                trajectories.setdefault(name, []).append(np.asarray([result["histories"][key] for result in results]))
            print(f"Finished {distribution}, seed {seed}, {mode}; {len(fractions)} fractions", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)

    for name, table in (("per_seed", rows), ("late_windows", late_rows)):
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    summary = {
        "config": asdict(config), "seeds": seeds, "fractions": fractions.tolist(), "horizons": list(horizons),
        "stride": stride, "chunk_size": chunk_size,
        "versions": {name: version(name) for name in ("bayes-cash", "numpy", "networkx")},
        "conventions": {
            "pairing": "same initial beliefs, topology, wealth quantiles, evidence, and contacts across all fractions and payment modes; same nested signed-edge masks across wealth distributions",
            "legacy_prefix": "first min(500, rounds) inputs exactly from existing make_inputs; horizon stability guaranteed for final horizons at least 500",
            "continuation": f"after round 500: default_rng(SeedSequence([seed, {CONTINUATION_NAMESPACE}, stream])); stream=1 evidence, stream=2 contacts; time-major contact uniforms mapped uniformly to each payer's sorted neighbors",
            "opponents": "edge uniforms from the existing sixth SeedSequence(seed).spawn(6) stream; edge is negative iff uniform < opponent_fraction",
            "cash": "SHA256 of full wealth[1:] and payment byte streams required identical across fractions within each seed, wealth distribution, and payment rule",
            "uncertainty": "sample SD across seeds, not confidence intervals",
            "belief_mse": "mean posterior variance + mean squared error of posterior means relative to true_bias",
            "truth_support": "true_bias must occur on theta grid; truth_excluded_fraction counts exactly zero mass; singleton_fraction counts exactly one nonzero grid mass, without tolerance or probability floors",
            "absorbing_disagreement_floor": "agents at an interior singleton belief with no allied neighbors are permanently frozen; sqrt(sum of their squared deviations from their own mean / total number of agents) bounds all future population disagreement, conditional on updates remaining defined",
            "late_windows": "exact per-round means on inclusive [T//2+1, 3*T//4] and [3*T//4+1, T], independent of saved trajectory stride",
            "limitations": "fixed theta grid and signed update with support-erasing projection; finite-horizon averages do not establish an equilibrium or phase boundary; a positive absorbing-disagreement floor separately certifies nonvanishing disagreement under unchanged dynamics; no empirical calibration",
        },
        "groups": _group_rows(rows, ("distribution", "mode", "opponent_fraction", "horizon"),
                              (*METRICS, "realized_opponent_fraction", "cumulative_cash_transferred")),
        "late_window_groups": _group_rows(late_rows, ("distribution", "mode", "opponent_fraction", "window_start", "window_end"), METRICS),
        "cash_hashes": hashes, "max_absolute_cash_error": max_cash_error,
    }
    trajectories = {name: np.asarray(values) for name, values in trajectories.items()}
    trajectories.update(round=results[0]["rounds"], seeds=np.asarray(seeds), probabilities=fractions)
    np.savez_compressed(output / "trajectories.npz", **trajectories)
    payload = {}
    for seed in seeds:
        current = replace(config, seed=seed)
        inputs = make_sweep_inputs(current, "loguniform")
        for key in ("theta", "adjacency", "initial_beliefs", "observations", "recipients"):
            payload[f"seed_{seed}_{key}"] = getattr(inputs, key)
        payload[f"seed_{seed}_relationships"] = np.asarray([
            make_inputs(replace(current, rounds=min(PREFIX_ROUNDS, config.rounds), opponent_fraction=fraction), "loguniform").relationships
            for fraction in fractions
        ])
        payload[f"loguniform_{seed}_initial_wealth"] = inputs.initial_wealth
        payload[f"uniform_{seed}_initial_wealth"] = make_inputs(replace(current, rounds=1), "uniform").initial_wealth
    payload["metadata_json"] = json.dumps(summary, allow_nan=False)
    np.savez_compressed(output / "inputs.npz", **payload)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/opponent_sweep"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=20000)
    parser.add_argument("--fractions", type=float, nargs="+", default=np.linspace(0.0, 1.0, 11).tolist())
    parser.add_argument("--horizons", type=int, nargs="+", default=[500, 5000, 20000])
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_sweep(Config(seed=args.seed, rounds=args.rounds), args.seeds, args.output,
              args.fractions, args.horizons, args.stride, args.chunk_size, args.workers)
