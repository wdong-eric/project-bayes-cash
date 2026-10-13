"""Protect relationship-only pairing and saved sample statistics."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from bayes_cash.model import Config, make_inputs


spec = importlib.util.spec_from_file_location(
    "compare_opponents", Path(__file__).parents[1] / "scripts" / "compare_opponents.py",
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


@pytest.mark.parametrize("mode", comparison.SCENARIOS)
def test_opponents_change_beliefs_without_changing_cash(mode):
    config = Config(n_agents=8, attachment=2, rounds=10, opponent_fraction=1.0)
    results = comparison.paired_results(config, make_inputs(config, "uniform"), mode)
    assert_array_equal(results["allies"].wealth, results["opponents"].wealth)
    assert_array_equal(results["allies"].payments, results["opponents"].payments)
    assert not np.allclose(results["allies"].beliefs, results["opponents"].beliefs)


def test_comparison_records_paired_differences_and_replayable_inputs(tmp_path):
    config = Config(n_agents=8, attachment=2, rounds=10, opponent_fraction=0.5)
    summary = comparison.run_comparison(config, 2, tmp_path)
    with (tmp_path / "per_seed.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 32
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    with np.load(tmp_path / "opponent_comparison.npz") as archive:
        inputs = make_inputs(config, "loguniform")
        assert_array_equal(archive["seed_42_relationships"], inputs.relationships)
        assert_array_equal(archive["seed_42_observations"], inputs.observations)
        assert_array_equal(archive["seed_42_recipients"], inputs.recipients)
        assert_array_equal(archive["loguniform_42_initial_wealth"], inputs.initial_wealth)
        for contrast in summary["paired_contrasts"]:
            distribution, mode = contrast["distribution"], contrast["mode"]
            for key in comparison.METRICS:
                values = {
                    (int(row["seed"]), row["arm"]): float(row[key]) for row in rows
                    if (row["distribution"], row["mode"]) == (distribution, mode)
                }
                differences = [values[seed, "opponents"] - values[seed, "allies"] for seed in (42, 43)]
                assert_allclose(contrast[key]["mean"], sum(differences) / 2)
                assert_allclose(contrast[key]["sample_sd"], abs(differences[1] - differences[0]) / np.sqrt(2))
                for arm in comparison.ARMS:
                    history = archive[f"{distribution}_{arm}_{mode}_{key}"]
                    assert history.shape == (2, config.rounds + 1)
                    assert_allclose(history[:, -1], [values[seed, arm] for seed in (42, 43)])
            assert contrast["wealth_gini"] == {"mean": 0.0, "sample_sd": 0.0}
    for name in ("opponent_comparison", "opponent_trajectories"):
        for extension in ("png", "pdf"):
            assert (tmp_path / f"{name}.{extension}").stat().st_size > 1000
    with pytest.raises(FileExistsError):
        comparison.run_comparison(config, 2, tmp_path)
