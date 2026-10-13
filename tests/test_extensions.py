"""Protect independent additions, bounded replay, and cash-source bookkeeping."""

import csv
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from bayes_cash.experiment import diagnostics
from bayes_cash.model import Config, make_inputs, simulate


scripts = Path(__file__).parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("compare_extensions", scripts / "compare_extensions.py")
comparison = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(scripts))
try:
    spec.loader.exec_module(comparison)
finally:
    sys.path.remove(str(scripts))


@pytest.mark.parametrize("arm", comparison.ARMS)
def test_chunking_matches_full_simulation_and_income_ledger(arm):
    config = comparison.extension_config(
        Config(n_agents=8, attachment=2, rounds=23, opponent_fraction=0.7), arm, 0.1,
    )
    inputs = comparison.make_sweep_inputs(config, "loguniform")
    full = simulate(config, inputs, "hybrid")
    chunked = comparison.run_trajectory(config, inputs, "hybrid", (10, 23), stride=3, chunk_size=7)
    expected = diagnostics(full, inputs.theta, config.true_bias)
    for key in comparison.METRICS:
        if key != "effective_opponent_flip_fraction":
            assert_array_equal(chunked["histories"][key], expected[key][chunked["rounds"]])
    total = full.wealth.sum(axis=1)
    assert_allclose(total, total[0] * (1 + config.income_rate) ** np.arange(config.rounds + 1))
    assert chunked["max_relative_cash_balance_error"] < 1e-13
    for window in chunked["late_window_rows"]:
        for key in ("opinion_disagreement", "belief_mse"):
            values = expected[key][window["window_start"]:window["window_end"] + 1]
            assert_allclose(window[key], values.mean(), rtol=1e-14, atol=1e-16)
    if config.income_rate:
        assert np.all(full.payments <= config.income_rate * full.wealth[:-1])


def test_long_input_prefix_preserves_original_experiment_and_horizon():
    config = Config(n_agents=8, attachment=2, rounds=503, opponent_fraction=0.2)
    inputs = comparison.make_sweep_inputs(config, "uniform")
    old = make_inputs(replace(config, rounds=500), "uniform")
    longer = comparison.make_sweep_inputs(replace(config, rounds=507), "uniform")
    assert_array_equal(inputs.observations[:500], old.observations)
    assert_array_equal(inputs.recipients[:500], old.recipients)
    assert_array_equal(inputs.observations, longer.observations[:503])
    assert_array_equal(inputs.recipients, longer.recipients[:503])


def test_comparison_saves_paired_arms_statistics_and_compact_histories(tmp_path):
    config = Config(n_agents=6, attachment=2, rounds=12, seed=42)
    summary = comparison.run_comparison(config, 2, tmp_path, probabilities=(0.2, 1.0),
                                        horizons=(6, 12), stride=3, chunk_size=5, workers=1)
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    with (tmp_path / "per_seed.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 * 2 * 3 * 2 * 4 * 2
    for group in summary["groups"]:
        matched = [row for row in rows if all(str(group[key]) == row[key]
                   for key in ("distribution", "mode", "arm", "opponent_fraction", "horizon"))]
        for key in comparison.METRICS:
            values = [float(row[key]) for row in matched]
            assert_allclose(group[key]["mean"], np.mean(values))
            assert_allclose(group[key]["sample_sd"], np.std(values, ddof=1))
    for distribution in comparison.DISTRIBUTIONS:
        for seed in summary["seeds"]:
            for mode in comparison.PAYMENT_MODES:
                hashes = [row for row in summary["cash_hashes"]
                          if (row["distribution"], row["seed"], row["mode"]) == (distribution, seed, mode)]
                for paired in (("baseline", "tanh_only"), ("income_only", "both")):
                    assert len({row["wealth_sha256"] for row in hashes if row["arm"] in paired}) == 1
                    assert len({row["payments_sha256"] for row in hashes if row["arm"] in paired}) == 1
    with np.load(tmp_path / "trajectories.npz", allow_pickle=False) as data:
        assert data["loguniform_fixed_both_belief_mse"].shape == (2, 2, 5)
        assert not any("beliefs" in key or "recipients" in key for key in data.files)
        assert_array_equal(data["probabilities"], [0.2, 1.0])
    with pytest.raises(FileExistsError):
        comparison.run_comparison(config, 2, tmp_path, horizons=(12,), workers=1)


def test_baseline_opponents_can_have_positive_effective_influence():
    config = Config(n_agents=2, attachment=1, rounds=4, opponent_fraction=1.0,
                    influence_rule="tanh", fixed_payment=0.2, public_evidence=False)
    inputs = comparison.make_sweep_inputs(config, "uniform")
    inputs = replace(inputs, initial_wealth=np.array([1.0, 2.0]))
    result = comparison.run_trajectory(config, inputs, "fixed", (4,), stride=1, chunk_size=2)
    assert_array_equal(result["histories"]["effective_opponent_flip_fraction"], [0, 1, 1, 1, 1])
    baseline = comparison.run_trajectory(replace(config, influence_rule="attenuation"), inputs,
                                         "fixed", (4,), stride=1, chunk_size=2)
    assert_array_equal(baseline["histories"]["effective_opponent_flip_fraction"], np.zeros(5))


def test_chunking_does_not_bypass_full_horizon_income_overflow_check():
    config = Config(n_agents=4, attachment=1, rounds=20000, income_rate=0.1)
    inputs = comparison.make_sweep_inputs(config, "loguniform")
    with pytest.raises(ValueError, match="income growth exceeds finite"):
        comparison.run_trajectory(config, inputs, "fixed", (500, 20000))


def test_seed_statistics_preserve_huge_finite_nominal_cash():
    rows = [{"arm": "income", "cash": value, "zero": 0.0}
            for value in (1e210, 2e210, 3e210)]
    with np.errstate(over="raise", invalid="raise"):
        summary = comparison._group_rows(rows, ("arm",), ("cash", "zero"))[0]
    assert_allclose(summary["cash"]["mean"] / 1e210, 2.0)
    assert_allclose(summary["cash"]["sample_sd"] / 1e210, 1.0)
    assert summary["zero"] == {"mean": 0.0, "sample_sd": 0.0}
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary
