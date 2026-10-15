"""Protect long-run pairing, checkpoint replay, and persistence diagnostics."""

import csv
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from bayes_cash.model import Config, PAYMENT_MODES, PreparedInputs, Result, make_inputs, simulate


scripts = Path(__file__).parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("sweep_opponents", scripts / "sweep_opponents.py")
sweep = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(scripts))
try:
    spec.loader.exec_module(sweep)
finally:
    sys.path.remove(str(scripts))


def test_continuation_keeps_old_500_rounds_and_long_horizon_prefixes():
    config = Config(n_agents=8, attachment=2, rounds=523, opponent_fraction=0.4)
    old = make_inputs(replace(config, rounds=500), "loguniform")
    short = sweep.make_sweep_inputs(config, "loguniform")
    long = sweep.make_sweep_inputs(replace(config, rounds=541), "loguniform")
    assert_array_equal(short.observations[:500], old.observations)
    assert_array_equal(short.recipients[:500], old.recipients)
    assert_array_equal(long.observations[:523], short.observations)
    assert_array_equal(long.recipients[:523], short.recipients)
    for key in ("adjacency", "relationships", "theta", "initial_beliefs", "initial_wealth"):
        assert_array_equal(getattr(old, key), getattr(long, key))
    uniform = sweep.make_sweep_inputs(config, "uniform")
    assert_array_equal(uniform.observations, short.observations)
    assert_array_equal(uniform.recipients, short.recipients)
    assert not np.array_equal(uniform.initial_wealth, short.initial_wealth)


def test_sweep_masks_are_nested_and_leave_external_inputs_identical():
    config = Config(n_agents=8, attachment=2, rounds=12)
    previous = np.zeros((8, 8), dtype=bool)
    original = sweep.make_sweep_inputs(config, "uniform")
    for fraction in (0.0, 0.2, 0.5, 1.0):
        current = sweep.make_sweep_inputs(replace(config, opponent_fraction=fraction), "uniform")
        assert np.all(~previous | (current.relationships < 0))
        previous = current.relationships < 0
        for key in ("adjacency", "theta", "initial_beliefs", "initial_wealth", "observations", "recipients"):
            assert_array_equal(getattr(original, key), getattr(current, key))
    assert_array_equal(previous, original.adjacency)


@pytest.mark.parametrize("mode", PAYMENT_MODES)
@pytest.mark.parametrize("prepare", [make_inputs, sweep.make_sweep_inputs], ids=["canonical", "historical"])
def test_chunks_reproduce_full_simulation_and_exact_late_round_selection(mode, prepare):
    config = Config(n_agents=8, attachment=2, rounds=24, opponent_fraction=0.5)
    inputs = prepare(config, "loguniform")
    full = simulate(config, inputs, mode)
    expected = sweep.sweep_diagnostics(full, inputs.theta, config.true_bias, inputs.relationships)
    actual = sweep.run_trajectory(config, inputs, mode, horizons=(5, 24), stride=7, chunk_size=5)
    assert_array_equal(actual["rounds"], [0, 5, 7, 14, 21, 24])
    for key in sweep.METRICS:
        assert_array_equal(actual["histories"][key], expected[key][actual["rounds"]])
        for row in actual["checkpoint_rows"]:
            assert row[key] == expected[key][row["horizon"]]
        for row in actual["late_window_rows"]:
            assert_allclose(row[key], expected[key][row["window_start"]:row["window_end"] + 1].mean(), atol=1e-16)
    assert actual["cash_digest"] == {
        "wealth_sha256": hashlib.sha256(full.wealth[1:].tobytes()).hexdigest(),
        "payments_sha256": hashlib.sha256(full.payments.tobytes()).hexdigest(),
    }
    assert_allclose(expected["belief_mse"], expected["mean_posterior_variance"] + expected["posterior_mean_mse"], rtol=1e-14)


def test_interior_singletons_without_allied_neighbors_certify_disagreement():
    theta = np.linspace(0, 1, 5)
    beliefs = np.zeros((1, 4, 5))
    beliefs[0, 0, 1] = 1.0
    beliefs[0, 1, 3] = 1.0
    beliefs[0, 2, 0] = 1.0  # Endpoint cannot survive both possible coin outcomes.
    beliefs[0, 3, 2] = 1.0
    signs = np.array([[0, -1, -1, -1], [-1, 0, -1, -1], [-1, -1, 0, 1], [-1, -1, 1, 0]])
    result = Result(beliefs, np.ones((1, 4)), np.empty((0, 4)), "fixed")
    values = sweep.sweep_diagnostics(result, theta, 0.5, signs)
    assert_array_equal(values["frozen_fraction"], [0.5])
    assert_allclose(values["absorbing_disagreement_floor"], [np.sqrt((0.25**2 + 0.25**2) / 4)])
    assert values["absorbing_disagreement_floor"][0] <= values["opinion_disagreement"][0]
    assert_array_equal(values["truth_excluded_fraction"], [0.75])
    assert_array_equal(values["singleton_fraction"], [1.0])


def test_truth_diagnostics_reject_an_off_grid_truth():
    config = Config(n_agents=6, attachment=2, rounds=4, true_bias=0.61)
    inputs = make_inputs(config, "uniform")
    with pytest.raises(ValueError, match="true_bias"):
        sweep.sweep_diagnostics(simulate(config, inputs, "fixed"), inputs.theta, config.true_bias, inputs.relationships)


def test_sweep_exports_replayable_inputs_statistics_and_cash_pairing(tmp_path):
    config = Config(n_agents=6, attachment=2, rounds=12)
    fractions = [0.0, 0.5, 1.0]
    summary = sweep.run_sweep(config, 2, tmp_path, fractions, horizons=(5, 12), stride=5, chunk_size=4, workers=1)
    with (tmp_path / "per_seed.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with (tmp_path / "late_windows.csv").open() as stream:
        late_rows = list(csv.DictReader(stream))
    assert len(rows) == len(late_rows) == 2 * 2 * 3 * 3 * 2
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert len(summary["cash_hashes"]) == 12
    assert summary["max_absolute_cash_error"] < 1e-10
    with np.load(tmp_path / "inputs.npz") as raw, np.load(tmp_path / "trajectories.npz") as saved:
        replay = PreparedInputs(
            theta=raw["seed_42_theta"], adjacency=raw["seed_42_adjacency"],
            relationships=raw["seed_42_relationships"][1], initial_beliefs=raw["seed_42_initial_beliefs"],
            initial_wealth=raw["uniform_42_initial_wealth"], observations=raw["seed_42_observations"],
            recipients=raw["seed_42_recipients"], wealth_distribution="uniform",
        )
        expected = sweep.sweep_diagnostics(simulate(config, replay, "fixed"), replay.theta, config.true_bias, replay.relationships)
        assert_array_equal(saved["seeds"], [42, 43])
        assert_array_equal(saved["probabilities"], fractions)
        for key in sweep.METRICS:
            assert saved[f"uniform_fixed_{key}"].shape == (2, 3, saved["round"].size)
            assert_array_equal(saved[f"uniform_fixed_{key}"][0, 1], expected[key][saved["round"]])
        for group in summary["groups"]:
            subset = [row for row in rows if row["distribution"] == group["distribution"]
                      and row["mode"] == group["mode"] and float(row["opponent_fraction"]) == group["opponent_fraction"]
                      and int(row["horizon"]) == group["horizon"]]
            for key in sweep.METRICS:
                values = [float(row[key]) for row in subset]
                assert_allclose(group[key]["mean"], np.mean(values))
                assert_allclose(group[key]["sample_sd"], np.std(values, ddof=1))
    with pytest.raises(FileExistsError):
        sweep.run_sweep(config, 2, tmp_path, fractions, horizons=(12,), workers=1)


def test_worker_failure_has_run_context(monkeypatch):
    config = Config(n_agents=6, attachment=2, rounds=4)
    def fail(*args, **kwargs):
        raise ValueError("scientific failure")
    monkeypatch.setattr(sweep, "run_trajectory", fail)
    with pytest.raises(RuntimeError, match="uniform, seed=42, mode=hybrid, opponent_fraction=0.5: scientific failure"):
        sweep._run_job((config, "uniform", "hybrid", [0.5], (4,), 2, 2))
