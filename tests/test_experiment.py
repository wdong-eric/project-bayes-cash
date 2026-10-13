import csv
import json

import numpy as np
import pytest

from bayes_cash.experiment import diagnostics, run_experiment
from bayes_cash.model import Config, Result


def test_diagnostics_distinguish_disagreement_uncertainty_and_inequality():
    result = Result(
        beliefs=np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]),
        wealth=np.array([[0.0, 2.0], [1.0, 1.0]]),
        payments=np.array([[0.0, 1.0]]),
        payment_mode="fixed",
    )
    measured = diagnostics(result, np.array([0.0, 1.0]), true_bias=0.5)
    np.testing.assert_allclose(measured["opinion_disagreement"], [0.5, 0.0])
    np.testing.assert_allclose(measured["mean_posterior_sd"], [0.0, 0.5])
    np.testing.assert_allclose(measured["belief_mse"], [0.25, 0.25])
    np.testing.assert_allclose(measured["wealth_gini"], [0.5, 0.0])


def test_export_contains_replayable_ledger_and_refuses_overwrite(tmp_path):
    config = Config(n_agents=4, attachment=1, rounds=3, grid_size=5)
    output = tmp_path / "experiment"
    summary = run_experiment(config, output)
    saved_summary = (output / "summary.json").read_bytes()
    assert json.loads(saved_summary) == summary
    for distribution in ("uniform", "loguniform"):
        with np.load(output / f"{distribution}_trajectories.npz", allow_pickle=False) as data:
            assert data["fixed_beliefs"].shape == (4, 4, 5)
            np.testing.assert_array_equal(data["fixed_beliefs"][0], data["initial_beliefs"])
            for mode in ("fixed", "fractional", "hybrid", "no_payment"):
                reconstructed = data["initial_wealth"].copy()
                for t in range(config.rounds):
                    for payer, receiver in enumerate(data["recipients"][t]):
                        amount = data[f"{mode}_payments"][t, payer]
                        reconstructed[payer] -= amount
                        reconstructed[receiver] += amount
                    np.testing.assert_allclose(reconstructed, data[f"{mode}_wealth"][t + 1])
        with (output / f"{distribution}_metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 4 * (config.rounds + 1)
        assert {row["scenario"] for row in rows} == {"fixed", "fractional", "hybrid", "no_payment"}
    with pytest.raises(FileExistsError, match="not empty"):
        run_experiment(config, output)
    assert (output / "summary.json").read_bytes() == saved_summary
