"""Guard the cash-only shortcut and pairing in the network experiment."""

from dataclasses import replace
import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from bayes_cash.model import Config, make_inputs, simulate


spec = importlib.util.spec_from_file_location(
    "check_network_effects", Path(__file__).parents[1] / "scripts" / "check_network_effects.py",
)
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)


@pytest.mark.parametrize("mode", ["fixed", "fractional", "hybrid"])
def test_cash_shortcut_exactly_matches_full_belief_simulation(mode):
    config = Config(n_agents=12, attachment=2, rounds=20)
    for inputs in controls.control_inputs(config, "loguniform").values():
        wealth, payments = controls.simulate_cash(config, inputs, mode)
        full = simulate(config, inputs, mode)
        assert_array_equal(wealth, full.wealth)
        assert_array_equal(payments, full.payments)


def test_network_controls_preserve_initial_conditions_and_original_500_rounds():
    config = Config(n_agents=12, attachment=2, rounds=505)
    original = make_inputs(replace(config, rounds=500), "uniform")
    inputs = controls.control_inputs(config, "uniform")
    for topology, current in inputs.items():
        assert_array_equal(current.initial_wealth, original.initial_wealth)
        assert_array_equal(current.initial_beliefs, original.initial_beliefs)
        assert nx.is_connected(nx.from_numpy_array(current.adjacency))
        assert current.adjacency[np.arange(config.n_agents), current.recipients].all()
        short = controls.control_inputs(replace(config, rounds=500), "uniform")[topology]
        assert_array_equal(current.recipients[:500], short.recipients)
    assert_array_equal(inputs["barabasi_albert"].recipients[:500], original.recipients)
    assert_array_equal(inputs["regular"].adjacency.sum(axis=1), np.full(config.n_agents, 4))
    assert_array_equal(inputs["complete"].adjacency.sum(axis=1), np.full(config.n_agents, 11))
