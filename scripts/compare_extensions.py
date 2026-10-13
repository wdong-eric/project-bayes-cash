"""Compare wealth-scaled tanh influence and income budgets as separate additions."""

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
from bayes_cash.model import PAYMENT_MODES, Config, contact_weights, simulate
from sweep_opponents import make_sweep_inputs, _group_rows, CONTINUATION_NAMESPACE


DISTRIBUTIONS = ("loguniform", "uniform")
ARMS = ("baseline", "tanh_only", "income_only", "both")
METRICS = (
    "mean_opinion", "opinion_disagreement", "belief_mse", "mean_posterior_sd",
    "wealth_gini", "fraction_paying", "effective_opponent_flip_fraction",
)


def extension_config(config, arm, income_rate):
    """Keep the two interventions independently switchable in the factorial comparison."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    return replace(config, influence_rule="tanh" if arm in ("tanh_only", "both") else "attenuation",
                   income_rate=income_rate if arm in ("income_only", "both") else 0.0)


def run_trajectory(config, inputs, mode, horizons, stride=50, chunk_size=500):
    """Save bounded diagnostic histories and cash hashes without retaining full states."""
    horizons = tuple(sorted(set(horizons)))
    if config.rounds < 4 or stride < 1 or chunk_size < 1:
        raise ValueError("rounds must be at least four; stride and chunk_size must be positive")
    if not horizons or horizons[0] < 1 or horizons[-1] > config.rounds:
        raise ValueError("horizons must lie between one and rounds")
    if np.log(inputs.initial_wealth.sum()) + config.rounds * np.log1p(config.income_rate) >= np.log(np.finfo(float).max):
        raise ValueError("income growth exceeds finite nominal money; reduce rounds or income_rate")
    rounds = np.unique(np.r_[np.arange(0, config.rounds + 1, stride), horizons, config.rounds])
    histories = {key: np.empty(rounds.size) for key in METRICS}
    windows = ((config.rounds // 2 + 1, 3 * config.rounds // 4),
               (3 * config.rounds // 4 + 1, config.rounds))
    sums = [{key: 0.0 for key in METRICS} for _ in windows]
    checkpoint_rows = []
    wealth_hash, payment_hash = hashlib.sha256(), hashlib.sha256()
    beliefs, wealth = inputs.initial_beliefs, inputs.initial_wealth
    initial_total = wealth.sum()
    max_relative_cash_error = 0.0
    cumulative_paid = 0.0
    for start in range(0, config.rounds, chunk_size):
        stop = min(start + chunk_size, config.rounds)
        current = replace(inputs, initial_beliefs=beliefs, initial_wealth=wealth,
                          observations=inputs.observations[start:stop], recipients=inputs.recipients[start:stop])
        result = simulate(replace(config, rounds=stop - start), current, mode)
        values = diagnostics(result, inputs.theta, config.true_bias)
        flips = np.zeros(stop - start + 1)
        for t, recipients in enumerate(current.recipients):
            opponents = inputs.relationships[np.arange(config.n_agents), recipients] < 0
            if opponents.any():
                weights = contact_weights(recipients, result.payments[t], inputs.relationships,
                                          config.reference_payment, influence_rule=config.influence_rule,
                                          opening_wealth=result.wealth[t], opponent_threshold=config.opponent_threshold)
                flips[t + 1] = np.mean(weights[opponents] > 0)
        values["effective_opponent_flip_fraction"] = flips
        if any(not np.isfinite(values[key]).all() for key in METRICS):
            raise ValueError(f"nonfinite diagnostics during rounds {start + 1} to {stop}")
        local_rounds = np.arange(start, stop + 1)
        selected = np.flatnonzero((rounds >= start + (start > 0)) & (rounds <= stop))
        for key in METRICS:
            histories[key][selected] = values[key][rounds[selected] - start]
            for index, (first, last) in enumerate(windows):
                mask = (local_rounds >= max(first, start + 1)) & (local_rounds <= last)
                sums[index][key] += float(values[key][mask].sum())
        paid = np.r_[0.0, np.cumsum(result.payments.sum(axis=1))] + cumulative_paid
        for horizon in horizons:
            if start < horizon <= stop:
                checkpoint_rows.append({"horizon": horizon,
                                        **{key: float(values[key][horizon - start]) for key in METRICS},
                                        "cumulative_nominal_cash_transferred": float(paid[horizon - start])})
        cumulative_paid = float(paid[-1])
        relative_total = values["wealth_total"] / initial_total * np.exp(-local_rounds * np.log1p(config.income_rate))
        max_relative_cash_error = max(max_relative_cash_error, float(np.max(np.abs(relative_total - 1))))
        wealth_hash.update(result.wealth[1:].tobytes())
        payment_hash.update(result.payments.tobytes())
        beliefs, wealth = result.beliefs[-1].copy(), result.wealth[-1].copy()
    return {
        "rounds": rounds, "histories": histories, "checkpoint_rows": checkpoint_rows,
        "late_window_rows": [{"window_start": first, "window_end": last,
                              **{key: total[key] / (last - first + 1) for key in METRICS}}
                             for (first, last), total in zip(windows, sums)],
        "cash_digest": {"wealth_sha256": wealth_hash.hexdigest(), "payments_sha256": payment_hash.hexdigest()},
        "max_relative_cash_balance_error": max_relative_cash_error,
    }


def _run_job(job):
    config, distribution, mode, fractions, income_rate, horizons, stride, chunk_size = job
    results = []
    cash_by_income = {}
    for fraction in fractions:
        current = replace(config, opponent_fraction=fraction)
        inputs = make_sweep_inputs(current, distribution)
        if np.log(inputs.initial_wealth.sum()) + config.rounds * np.log1p(income_rate) >= np.log(np.finfo(float).max):
            raise ValueError("income growth exceeds finite nominal money; reduce rounds or income_rate")
        for arm in ARMS:
            arm_config = extension_config(current, arm, income_rate)
            try:
                result = run_trajectory(arm_config, inputs, mode, horizons, stride, chunk_size)
                previous = cash_by_income.setdefault(arm_config.income_rate, result["cash_digest"])
                if result["cash_digest"] != previous:
                    raise ValueError("relationship signs or influence rule changed the cash trajectory")
            except Exception as error:
                raise RuntimeError(f"{distribution}, seed={config.seed}, {mode}, opponents={fraction}, arm={arm}: {error}") from error
            result.update(arm=arm, opponent_fraction=fraction,
                          realized_opponent_fraction=float(np.count_nonzero(inputs.relationships < 0) / np.count_nonzero(inputs.adjacency)))
            results.append(result)
    return distribution, config.seed, mode, results


def run_comparison(config, seed_count, output, probabilities=(0.2, 1.0), income_rate=0.1,
                   horizons=(500, 5000), stride=50, chunk_size=500, workers=4):
    """Run all four interventions with shared random inputs and preserve seed statistics."""
    fractions = np.asarray(probabilities, dtype=float)
    if fractions.ndim != 1 or not fractions.size or not np.isfinite(fractions).all() or np.any((fractions < 0) | (fractions > 1)) or np.any(np.diff(fractions) <= 0):
        raise ValueError("fractions must be finite, unique, increasing probabilities")
    if seed_count < 2 or workers < 1:
        raise ValueError("at least two seeds and one worker are required")
    if not np.isfinite(income_rate) or income_rate <= 0:
        raise ValueError("comparison income_rate must be positive and finite")
    # Validate the income configuration; each job checks nominal cash range before its first arm.
    extension_config(config, "both", income_rate)
    horizons = tuple(sorted(set((*horizons, config.rounds))))
    if config.rounds < 4 or horizons[0] < 1 or horizons[-1] > config.rounds or stride < 1 or chunk_size < 1:
        raise ValueError("invalid horizon, stride, or chunk size")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seeds = list(range(config.seed, config.seed + seed_count))
    jobs = [(replace(config, seed=seed), distribution, mode, fractions.tolist(), income_rate, horizons, stride, chunk_size)
            for distribution in DISTRIBUTIONS for seed in seeds for mode in PAYMENT_MODES]
    rows, late_rows, trajectories, hashes = [], [], {}, []
    max_cash_error = 0.0
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        completed = executor.map(_run_job, jobs) if executor else map(_run_job, jobs)
        for distribution, seed, mode, results in completed:
            for result in results:
                identity = {"distribution": distribution, "seed": seed, "mode": mode,
                            "arm": result["arm"], "opponent_fraction": result["opponent_fraction"],
                            "realized_opponent_fraction": result["realized_opponent_fraction"]}
                rows.extend({**identity, **row} for row in result["checkpoint_rows"])
                late_rows.extend({**identity, **row} for row in result["late_window_rows"])
                max_cash_error = max(max_cash_error, result["max_relative_cash_balance_error"])
                hashes.append({**identity, **result["cash_digest"]})
            for arm in ARMS:
                arm_results = [result for result in results if result["arm"] == arm]
                for key in METRICS:
                    name = f"{distribution}_{mode}_{arm}_{key}"
                    trajectories.setdefault(name, []).append(np.asarray([result["histories"][key] for result in arm_results]))
            print(f"Finished {distribution}, seed {seed}, {mode}; {len(fractions)} fractions, four arms", flush=True)
    finally:
        if executor:
            executor.shutdown(cancel_futures=True)
    for name, table in (("per_seed", rows), ("late_windows", late_rows)):
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    keys = ("distribution", "mode", "opponent_fraction", "arm")
    summary = {
        "config": asdict(config), "income_rate": income_rate, "seeds": seeds,
        "fractions": fractions.tolist(), "horizons": list(horizons), "stride": stride, "chunk_size": chunk_size,
        "arms": {arm: asdict(extension_config(config, arm, income_rate)) for arm in ARMS},
        "versions": {name: version(name) for name in ("bayes-cash", "numpy", "networkx")},
        "conventions": {
            "pairing": "same initial beliefs, graph, wealth quantiles, evidence and contacts; cash hashes equal across influence rules and opponent probabilities at each income setting",
            "replay": f"make_sweep_inputs retains original first 500 rounds, then SeedSequence([seed, {CONTINUATION_NAMESPACE}, stream]); config, seed, distribution and opponent_fraction fully determine inputs",
            "tanh": "recipient opening wealth W_i * tanh(payment / W_i - kappa), kappa=opponent_threshold for baseline opponents and zero for allies; W_i=0 gives zero weight; signs recalculated each contact",
            "self_weight": "fixed degree * reference_payment * (1-social_rate)/social_rate in all arms",
            "income": "income_rate * opening wealth; requested payments must be affordable from opening wealth and at most income; skip, do not clip; simultaneous income and transfers",
            "cash_balance": "total wealth divided by initial total and (1+income_rate)^round must equal one; reported maximum relative error; cumulative cash is nominal, unadjusted for income growth",
            "effective_opponent_flip_fraction": "fraction of scheduled baseline-opponent contacts carrying strictly positive influence each round; zero at round zero and at rounds with no such contacts",
            "fraction_paying": "fraction of scheduled contact attempts with strictly positive actual payment; zero at round zero",
            "uncertainty": "sample SD across seeds, not confidence intervals",
            "belief_mse": "mean posterior variance plus mean squared error of posterior means about true_bias",
            "late_windows": "exact per-round means on inclusive [T//2+1,3*T//4] and [3*T//4+1,T]",
            "limitations": "finite grid and support-erasing signed projection; no frozen-agent certificate applies when baseline opponents can become attractive; fixed self-weight makes income-driven wealth growth increase tanh social strength; no equilibrium claim",
        },
        "groups": _group_rows(rows, (*keys, "horizon"), (*METRICS, "realized_opponent_fraction", "cumulative_nominal_cash_transferred")),
        "late_window_groups": _group_rows(late_rows, (*keys, "window_start", "window_end"), METRICS),
        "cash_hashes": hashes, "max_relative_cash_balance_error": max_cash_error,
    }
    trajectories = {name: np.asarray(values) for name, values in trajectories.items()}
    trajectories.update(round=results[0]["rounds"], seeds=np.asarray(seeds), probabilities=fractions)
    np.savez_compressed(output / "trajectories.npz", **trajectories)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/extensions"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=5000)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.2, 1.0])
    parser.add_argument("--horizons", type=int, nargs="+", default=[500, 5000])
    parser.add_argument("--income-rate", type=float, default=0.1)
    parser.add_argument("--opponent-threshold", type=float, default=0.05)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_comparison(Config(seed=args.seed, rounds=args.rounds, opponent_threshold=args.opponent_threshold),
                   args.seeds, args.output, args.fractions, args.income_rate, args.horizons,
                   args.stride, args.chunk_size, args.workers)
