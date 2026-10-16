from dataclasses import replace
from statistics import NormalDist

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from bayes_cash.model import Config, Intervention, PreparedInputs
from bayes_cash.strategy import GaussianComponent, StrategySettings, initialize_beliefs, simulate_strategy


@pytest.fixture
def pair():
    config = Config(
        n_agents=2, rounds=4, attachment=1, grid_size=3, public_evidence=False,
        social_rate=0.5, opponent_threshold=0.1, fixed_payment=1.0,
        preference_probability=1.0, loyalty_rounds=1,
    )
    inputs = PreparedInputs(
        theta=np.array([0.0, 0.5, 1.0]),
        adjacency=np.array([[False, True], [True, False]]),
        relationships=np.array([[0, -1], [-1, 0]], dtype=np.int8),
        initial_beliefs=np.array([[0.8, 0.2, 0.0], [0.1, 0.2, 0.7]]),
        initial_wealth=np.array([20.0, 5.0]),
        observations=np.ones(4, dtype=int),
        recipients=np.tile([1, 0], (4, 1)),
        wealth_distribution="uniform",
        giving_preferences=np.zeros(2, dtype=np.int8),
        conditional_allies=np.zeros(2, dtype=bool),
        contact_uniforms=np.zeros((4, 2)),
    )
    return config, inputs


def test_uniform_population_is_evenly_spaced_and_seeded_with_distinct_normalized_beliefs():
    config = Config(n_agents=8, attachment=1, grid_size=31)
    centers, beliefs = initialize_beliefs(config, StrategySettings())
    assert_allclose(np.sort(centers), np.linspace(0, 1, 8))
    assert not np.array_equal(centers, np.sort(centers))
    assert_allclose(beliefs.sum(axis=1), 1.0)
    assert np.unique(beliefs, axis=0).shape[0] == config.n_agents
    repeated_centers, repeated_beliefs = initialize_beliefs(config, StrategySettings())
    assert_array_equal(centers, repeated_centers)
    assert_array_equal(beliefs, repeated_beliefs)
    changed_centers, _ = initialize_beliefs(replace(config, seed=config.seed + 1), StrategySettings())
    assert not np.array_equal(centers, changed_centers)


@pytest.mark.parametrize("components", [
    (GaussianComponent(0.5, 0.04),),
    (GaussianComponent(0.05, 0.01, 1), GaussianComponent(0.6, 0.02, 2)),
    (GaussianComponent(0.1, 0.01), GaussianComponent(0.5, 0.02), GaussianComponent(0.9, 0.01)),
])
def test_gaussian_population_quantiles_follow_conditioned_mixture(components):
    config = Config(n_agents=15, attachment=1)
    settings = StrategySettings(belief_distribution="gaussian_mixture", components=components)
    centers, beliefs = initialize_beliefs(config, settings)
    normals = [NormalDist(component.mean, np.sqrt(component.variance)) for component in components]
    retained = sum(component.weight * (normal.cdf(1) - normal.cdf(0))
                   for component, normal in zip(components, normals))
    cumulative = np.array([
        sum(component.weight * (normal.cdf(center) - normal.cdf(0))
            for component, normal in zip(components, normals)) / retained
        for center in np.sort(centers)
    ])
    assert_allclose(cumulative, (np.arange(config.n_agents) + 0.5) / config.n_agents, atol=1e-13)
    assert_allclose(beliefs.sum(axis=1), 1)
    assert np.all(np.diff(np.sort(centers)) > 0)


@pytest.mark.parametrize("settings", [
    {"components": (GaussianComponent(),) * 4},
    {"components": (GaussianComponent(weight=0),)},
    {"within_agent_variance": 0},
    {"incitation_cost": float("inf")},
    {"reinforcement_scale": -1},
    {"belief_distribution": "clipped_gaussian"},
])
def test_invalid_strategy_settings_fail_explicitly(settings):
    with pytest.raises(ValueError):
        StrategySettings(**settings)


def test_unresolvable_gaussian_population_is_not_clipped_or_substituted():
    config = Config(n_agents=4, attachment=1)
    settings = StrategySettings(belief_distribution="gaussian_mixture", components=(GaussianComponent(0.5, 1e308),))
    with pytest.raises(ValueError, match="unresolvable mass"):
        initialize_beliefs(config, settings)
    settings = replace(settings, components=(GaussianComponent(0.5, 1e-300),))
    with pytest.raises(ValueError, match="indistinguishable"):
        initialize_beliefs(config, settings)


@pytest.mark.parametrize("variance", [1e-300, 1e308])
def test_unresolvable_agent_beliefs_are_rejected(variance):
    config = Config(n_agents=20, attachment=1, grid_size=3)
    with pytest.raises(ValueError, match="indistinguishable"):
        initialize_beliefs(config, StrategySettings(within_agent_variance=variance))


def test_crossing_creates_mutual_alliance_without_changing_reverse_current_weight(pair):
    config, inputs = pair
    result = simulate_strategy(config, inputs, StrategySettings(payment_mode="fixed"))
    assert result.relationships[1, 0, 1] == result.relationships[1, 1, 0] == 1
    forward = 5.0 * np.tanh(1.0 / 5.0 - 0.1)
    reverse = 20.0 * np.tanh(1.0 / 20.0 - 0.1)
    expected = inputs.initial_beliefs.copy()
    for recipient, payer, weight in [(0, 1, reverse), (1, 0, forward)]:
        expected[recipient] += weight * (inputs.initial_beliefs[payer] - inputs.initial_beliefs[recipient]) / (1 + abs(weight))
        expected[recipient] = np.maximum(expected[recipient], 0)
        expected[recipient] /= expected[recipient].sum()
    assert_allclose(result.beliefs[1], expected)
    assert_allclose(result.wealth.sum(axis=1), inputs.initial_wealth.sum())
    assert_array_equal(result.relationships, result.relationships.transpose(0, 2, 1))
    assert_array_equal(inputs.relationships, [[0, -1], [-1, 0]])


@pytest.mark.parametrize("conditional_setting", [None, False])
def test_alliance_persists_through_unpaid_rounds_and_disabling_conditional_trait(pair, conditional_setting):
    config, inputs = pair
    inputs = replace(inputs, conditional_allies=np.ones(2, dtype=bool))
    events = tuple(Intervention(agent=agent, round=2, giving_preference=-1, conditional_ally=conditional_setting) for agent in range(2))
    result = simulate_strategy(config, inputs, StrategySettings(payment_mode="fixed"), interventions=events)
    assert_array_equal(result.payments[1:], np.zeros((3, 2)))
    assert_array_equal(result.recipients[1:], np.full((3, 2), -1))
    assert_array_equal(result.relationships[1:, 0, 1], np.ones(4))
    assert_allclose(result.beliefs[1:], np.broadcast_to(result.beliefs[1], result.beliefs[1:].shape))


def test_conditional_positive_payment_converts_below_tanh_threshold_and_at_zero_recipient_wealth(pair):
    config, inputs = pair
    config = replace(config, opponent_threshold=100.0)
    inputs = replace(inputs, conditional_allies=np.array([False, True]))
    result = simulate_strategy(config, inputs, StrategySettings(payment_mode="fixed"))
    assert result.relationships[1, 0, 1] == result.relationships[1, 1, 0] == 1
    assert result.beliefs[1, 1, 0] > inputs.initial_beliefs[1, 0]
    inputs = replace(inputs, initial_wealth=np.array([20.0, 0.0]))
    result = simulate_strategy(config, inputs, StrategySettings(payment_mode="fixed"))
    assert result.payments[0, 0] == 1.0
    assert result.relationships[1, 0, 1] == result.relationships[1, 1, 0] == 1
    assert_array_equal(result.beliefs[1, 1], inputs.initial_beliefs[1])


@pytest.mark.parametrize("fixed_payment", [0.0, 0.5])
def test_zero_and_exact_threshold_payments_do_not_convert_nonconditional_opponents(pair, fixed_payment):
    config, inputs = pair
    config = replace(config, fixed_payment=fixed_payment)
    inputs = replace(inputs, initial_wealth=np.full(2, 5.0))
    result = simulate_strategy(config, inputs, StrategySettings(payment_mode="fixed"))
    assert_array_equal(result.relationships[:, 0, 1], -np.ones(5))


def test_no_payment_does_not_convert_conditional_recipients(pair):
    config, inputs = pair
    inputs = replace(inputs, conditional_allies=np.ones(2, dtype=bool))
    result = simulate_strategy(config, inputs, StrategySettings(payment_mode="no_payment"))
    assert_array_equal(result.payments, np.zeros((4, 2)))
    assert_array_equal(result.relationships[:, 0, 1], -np.ones(5))
    assert not np.array_equal(result.beliefs[1], result.beliefs[0])


def test_intervention_replay_preserves_prefix_and_cash_accounting(pair):
    config, inputs = pair
    settings = StrategySettings(payment_mode="no_payment")
    baseline = simulate_strategy(config, inputs, settings)
    events = (
        Intervention(agent=0, round=3, cash_delta=3),
        Intervention(agent=0, round=3, belief_center=0.5, belief_width=0.1, belief_strength=1),
        Intervention(agent=0, round=3, neighbor=1, connection=1),
    )
    changed = simulate_strategy(config, inputs, settings, interventions=events)
    assert_array_equal(changed.beliefs[:3], baseline.beliefs[:3])
    assert_array_equal(changed.wealth[:3], baseline.wealth[:3])
    assert_array_equal(changed.relationships[:3], baseline.relationships[:3])
    assert_allclose(changed.wealth.sum(axis=1), [25, 25, 25, 28, 28])
    assert_allclose(changed.external_cash.sum(axis=1), [0, 0, 3, 0])
    assert changed.relationships[3, 0, 1] == 1
    assert changed.beliefs[3, 0, 1] > 0.999


def test_manual_opponent_can_be_reconverted_by_same_round_payment(pair):
    config, inputs = pair
    event = Intervention(agent=0, round=2, neighbor=1, connection=-1)
    result = simulate_strategy(config, inputs, StrategySettings(payment_mode="fixed"), interventions=(event,))
    assert result.relationships[2, 0, 1] == 1


def test_future_cost_proposals_and_loyalty_duration_have_no_effect(pair):
    config, inputs = pair
    settings = StrategySettings(payment_mode="fixed")
    initial = simulate_strategy(config, inputs, settings)
    changed = simulate_strategy(
        replace(config, loyalty_rounds=100), inputs,
        replace(settings, incitation_cost=900, reinforcement_scale=700),
    )
    assert_array_equal(initial.beliefs, changed.beliefs)
    assert_array_equal(initial.wealth, changed.wealth)
    assert_array_equal(initial.relationships, changed.relationships)


def test_removing_connection_is_rejected(pair):
    config, inputs = pair
    with pytest.raises(ValueError, match="cannot be removed"):
        simulate_strategy(config, inputs, StrategySettings(), interventions=(Intervention(agent=0, round=1, neighbor=1, connection=0),))
