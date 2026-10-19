from dataclasses import replace
from statistics import NormalDist

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from bayes_cash.model import Config, PreparedInputs, bayesian_update, social_update
from bayes_cash.strategy import (
    GaussianComponent, StrategyIntervention, StrategySettings, initialize_beliefs,
    initialize_person_types, simulate_strategy, wasserstein_distance,
)


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


@pytest.fixture
def four_agents():
    config = Config(n_agents=4, rounds=1, attachment=1, grid_size=3,
                    public_evidence=False, wealth_fraction=0.2, social_rate=0.5)
    adjacency = ~np.eye(4, dtype=bool)
    inputs = PreparedInputs(
        theta=np.array([0.0, 0.5, 1.0]), adjacency=adjacency,
        relationships=-adjacency.astype(np.int8),
        initial_beliefs=np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 1., 0.]]),
        initial_wealth=np.array([10., 20., 30., 40.]), observations=np.ones(1, dtype=int),
        recipients=np.array([[1, 0, 0, 0]]), wealth_distribution="uniform",
        contact_uniforms=np.array([[0.0, 0.5, 0.999, 0.25]]),
    )
    return config, inputs


def test_person_types_are_exclusive_reproducible_and_not_the_old_boolean_trait(pair):
    config, inputs = pair
    assert initialize_person_types(config, StrategySettings(junzi_fraction=1)) == ("junzi",) * 2
    assert initialize_person_types(replace(config, conditional_ally_fraction=1), StrategySettings()) == ("payment_dependent",) * 2
    with pytest.raises(ValueError, match="sum to at most 1"):
        initialize_person_types(replace(config, conditional_ally_fraction=0.7), StrategySettings(junzi_fraction=0.4))
    config = replace(config, n_agents=100, conditional_ally_fraction=0.4)
    first = initialize_person_types(config, StrategySettings(junzi_fraction=0.3))
    assert set(first) == {"ordinary", "payment_dependent", "junzi"}
    assert first == initialize_person_types(config, StrategySettings(junzi_fraction=0.3))
    assert first != initialize_person_types(replace(config, seed=99), StrategySettings(junzi_fraction=0.3))
    config, inputs = pair
    old_traits = replace(inputs, giving_preferences=np.ones(2, dtype=np.int8), conditional_allies=np.ones(2, dtype=bool))
    ordinary = simulate_strategy(config, inputs, StrategySettings())
    ignored_old_traits = simulate_strategy(config, old_traits, StrategySettings())
    assert_array_equal(ordinary.beliefs, ignored_old_traits.beliefs)
    assert_array_equal(ordinary.relationships, ignored_old_traits.relationships)


@pytest.mark.parametrize("strategy,expected", [
    ("random_one", [{1}, {2}, {3}, {0}]),
    ("all", [{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}]),
    ("farthest", [{2}, {0}, {0}, {0}]),
    ("farthest_random", [{1, 2}, {0, 3}, {0, 3}, {0, 1}]),
])
def test_targeting_and_equal_split_keep_one_total_budget(four_agents, strategy, expected):
    config, inputs = four_agents
    result = simulate_strategy(config, inputs, StrategySettings(payment_strategy=strategy))
    for payer, recipients in enumerate(expected):
        assert set(np.flatnonzero(result.contacts[0, payer])) == recipients
        assert_allclose(result.transfers[0, payer, list(recipients)], result.payments[0, payer] / len(recipients))
    assert_array_equal(np.diag(result.contacts[0]), False)
    assert_allclose(result.payments[0], [2, 4, 6, 8])
    assert_allclose(result.transfers.sum(axis=2), result.payments)
    assert_allclose(result.wealth[1], inputs.initial_wealth - result.payments[0] + result.transfers[0].sum(axis=0))
    assert_allclose(result.wealth.sum(axis=1), 100)


def test_random_contacts_use_saved_uniforms_and_not_legacy_recipients(four_agents):
    config, inputs = four_agents
    first = simulate_strategy(config, inputs, StrategySettings())
    changed = simulate_strategy(config, replace(inputs, recipients=np.array([[3, 3, 3, 2]])), StrategySettings())
    assert_array_equal(first.contacts, changed.contacts)
    with pytest.raises(ValueError, match="contact_uniforms"):
        simulate_strategy(config, replace(inputs, contact_uniforms=None), StrategySettings())


def test_farthest_plus_random_has_only_one_recipient_when_n_is_two(pair):
    config, inputs = pair
    result = simulate_strategy(config, inputs, StrategySettings(payment_strategy="farthest_random"))
    assert_array_equal(result.contacts.sum(axis=2), np.ones((4, 2)))
    assert_allclose(result.transfers[:, 0, 1], result.payments[:, 0])


def test_farthest_uses_post_evidence_pmfs_and_lowest_id_ties(four_agents):
    config, inputs = four_agents
    priors = np.array([[.8, .1, .1], [.1, .7, .2], [.1, .2, .7], [.8, .1, .1]])
    inputs = replace(inputs, initial_beliefs=priors)
    result = simulate_strategy(replace(config, public_evidence=True), inputs,
                               StrategySettings(payment_strategy="farthest", payment_mode="no_payment"))
    posterior = bayesian_update(priors, inputs.theta, 1)
    before = wasserstein_distance(priors[:, None, :], priors[None, :, :], inputs.theta)
    after = wasserstein_distance(posterior[:, None, :], posterior[None, :, :], inputs.theta)
    np.fill_diagonal(before, -np.inf)
    np.fill_diagonal(after, -np.inf)
    assert not np.array_equal(before.argmax(axis=1), after.argmax(axis=1))
    assert_array_equal(result.contacts[0].argmax(axis=1), after.argmax(axis=1))


def test_farthest_uses_wasserstein_two_instead_of_one(four_agents):
    config, inputs = four_agents
    config = replace(config, grid_size=5)
    inputs = replace(inputs, theta=np.linspace(0, 1, 5), initial_beliefs=np.array([
        [1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0.],
        [.9, 0., 0., 0., .1],
        [1., 0., 0., 0., 0.],
    ]))
    result = simulate_strategy(config, inputs, StrategySettings(payment_strategy="farthest"))
    # From agent 0, W1 prefers agent 1 (.25 > .1), W2 prefers 2 (sqrt(.1) > .25).
    assert_array_equal(np.flatnonzero(result.contacts[0, 0]), [2])


def test_zero_budget_still_contacts_and_can_repel(four_agents):
    config, inputs = four_agents
    result = simulate_strategy(config, inputs, StrategySettings(payment_mode="no_payment", payment_strategy="all"))
    assert_array_equal(result.contacts[0], inputs.adjacency)
    assert_array_equal(result.transfers, np.zeros((1, 4, 4)))
    assert_array_equal(result.relationships[1], inputs.relationships)
    # Start with overlapping PMFs so repulsion need not project back to the same delta.
    mixed = replace(inputs, initial_beliefs=np.array([[.7, .2, .1], [.2, .5, .3], [.1, .2, .7], [.3, .3, .4]]))
    result = simulate_strategy(config, mixed, StrategySettings(payment_mode="no_payment", payment_strategy="all"))
    assert not np.allclose(result.beliefs[1], mixed.initial_beliefs)


def test_matrix_pooling_agrees_with_original_for_one_contact_per_payer(four_agents):
    config, inputs = four_agents
    result = simulate_strategy(config, inputs, StrategySettings())
    recipients = result.contacts[0].argmax(axis=1)
    expected = social_update(
        inputs.initial_beliefs, recipients, result.payments[0], np.full(4, 3.0),
        inputs.relationships, config.reference_payment, influence_rule="tanh",
        opening_wealth=inputs.initial_wealth, opponent_threshold=config.opponent_threshold,
    )
    assert_allclose(result.beliefs[1], expected)


def test_all_recipient_pooling_is_equivariant_under_agent_permutation(four_agents):
    config, inputs = four_agents
    order = np.array([2, 0, 3, 1])
    shuffled = replace(inputs, initial_beliefs=inputs.initial_beliefs[order], initial_wealth=inputs.initial_wealth[order])
    settings = StrategySettings(payment_strategy="all")
    original = simulate_strategy(config, inputs, settings)
    permuted = simulate_strategy(config, shuffled, settings)
    assert_allclose(permuted.beliefs, original.beliefs[:, order])
    assert_allclose(permuted.wealth, original.wealth[:, order])
    assert_array_equal(permuted.relationships, original.relationships[:, order][:, :, order])


def test_crossing_creates_mutual_base_without_changing_reverse_current_weight(pair):
    config, inputs = pair
    result = simulate_strategy(config, inputs, StrategySettings(payment_mode="fixed"))
    assert result.persistent_relationships[1, 0, 1] == result.persistent_relationships[1, 1, 0] == 1
    forward = 5.0 * np.tanh(1.0 / 5.0 - 0.1)
    reverse = 20.0 * np.tanh(1.0 / 20.0 - 0.1)
    expected = inputs.initial_beliefs.copy()
    for recipient, payer, weight in [(0, 1, reverse), (1, 0, forward)]:
        expected[recipient] += weight * (inputs.initial_beliefs[payer] - inputs.initial_beliefs[recipient]) / (1 + abs(weight))
        expected[recipient] = np.maximum(expected[recipient], 0)
        expected[recipient] /= expected[recipient].sum()
    assert_allclose(result.beliefs[1], expected)
    assert_array_equal(result.persistent_relationships, result.persistent_relationships.transpose(0, 2, 1))
    assert_array_equal(inputs.relationships, [[0, -1], [-1, 0]])


@pytest.mark.parametrize("fixed_payment", [0.0, 0.5])
def test_zero_and_exact_threshold_payments_do_not_convert_base(pair, fixed_payment):
    config, inputs = pair
    config = replace(config, fixed_payment=fixed_payment)
    inputs = replace(inputs, initial_wealth=np.full(2, 5.0))
    result = simulate_strategy(config, inputs, StrategySettings(payment_mode="fixed"))
    assert_array_equal(result.persistent_relationships[:, 0, 1], -np.ones(5))


@pytest.mark.parametrize("threshold", [0.1, 100.0])
def test_temporary_loyalty_expires_on_exact_unpaid_round_without_erasing_base(pair, threshold):
    config, inputs = pair
    config = replace(config, fixed_payment=2, opponent_threshold=threshold, loyalty_rounds=2)
    inputs = replace(inputs, initial_wealth=np.array([20., 1.]))
    events = (StrategyIntervention(0, 2, cash_delta=-18), StrategyIntervention(0, 3, cash_delta=-2))
    result = simulate_strategy(config, inputs, StrategySettings(payment_mode="fixed"), interventions=events,
                               person_types=("ordinary", "payment_dependent"))
    assert_array_equal(result.payments[:, 0], [2, 0, 0, 0])
    assert_array_equal(result.relationships[:, 0, 1], [-1, 1, 1, -1, -1])
    expected_base = 1 if threshold == 0.1 else -1
    assert_array_equal(result.persistent_relationships[1:, 0, 1], expected_base)
    assert_array_equal(result.relationships[1:, 1, 0], expected_base)
    longer = simulate_strategy(replace(config, loyalty_rounds=3), inputs, StrategySettings(payment_mode="fixed"),
                               interventions=events, person_types=("ordinary", "payment_dependent"))
    assert_allclose(longer.beliefs[3, 1], longer.beliefs[2, 1])
    assert not np.allclose(result.beliefs[3, 1], result.beliefs[2, 1])


def test_positive_payment_to_zero_wealth_conditional_recipient_sets_only_temporary_sign(pair):
    config, inputs = pair
    inputs = replace(inputs, initial_wealth=np.array([20., 0.]))
    result = simulate_strategy(config, inputs, StrategySettings(payment_mode="fixed"),
                               person_types=("ordinary", "payment_dependent"))
    assert result.relationships[1, 0, 1] == 1
    assert result.persistent_relationships[1, 0, 1] == -1
    assert_array_equal(result.beliefs[1, 1], inputs.initial_beliefs[1])


def test_junzi_ignores_incoming_beliefs_and_blocks_mutual_conversion_but_exchanges_cash(pair):
    config, inputs = pair
    result = simulate_strategy(config, inputs, StrategySettings(payment_mode="fixed"), person_types=("junzi", "ordinary"))
    assert_array_equal(result.beliefs[:, 0], np.broadcast_to(inputs.initial_beliefs[0], (5, 3)))
    assert not np.allclose(result.beliefs[1, 1], inputs.initial_beliefs[1])
    assert_array_equal(result.relationships[:, 0, 1], -1)
    assert_array_equal(result.relationships[:, 1, 0], -1)
    assert_array_equal(result.persistent_relationships[:, 0, 1], -1)
    assert_array_equal(result.transfers[:, 0, 1], 1)
    assert_array_equal(result.transfers[:, 1, 0], 1)
    assert_allclose(result.wealth.sum(axis=1), 25)


def test_junzi_retains_evidence_and_manual_belief_and_connection_edits(pair):
    config, inputs = pair
    config = replace(config, public_evidence=True)
    event = StrategyIntervention(0, 1, belief_center=.6, belief_width=.2, belief_strength=1, neighbor=1, connection=1)
    result = simulate_strategy(config, inputs, StrategySettings(payment_mode="fixed"), interventions=(event,),
                               person_types=("junzi", "ordinary"))
    prior = np.exp(-.5 * ((inputs.theta - .6) / .2) ** 2)
    prior /= prior.sum()
    expected = bayesian_update(prior[None, :], inputs.theta, 1)[0]
    assert_allclose(result.beliefs[1, 0], expected)
    assert_array_equal(result.relationships[1:, 1, 0], 1)
    assert_array_equal(result.persistent_relationships[1:, 0, 1], 1)


def test_type_changes_keep_layers_distinct_and_junzi_freezes_current_response(pair):
    config, inputs = pair
    events = (
        StrategyIntervention(0, 1, neighbor=1, connection=1),
        StrategyIntervention(1, 2, person_type="payment_dependent"),
        StrategyIntervention(1, 3, person_type="junzi"),
        StrategyIntervention(1, 4, person_type="ordinary"),
    )
    result = simulate_strategy(config, inputs, StrategySettings(payment_mode="no_payment"), interventions=events)
    assert_array_equal(result.persistent_relationships[:, 0, 1], [-1, 1, 1, 1, 1])
    assert_array_equal(result.relationships[:, 0, 1], [-1, 1, -1, -1, 1])
    assert_allclose(result.beliefs[3, 1], result.beliefs[2, 1])


def test_reapplying_same_type_does_not_reset_unpaid_clock(pair):
    config, inputs = pair
    events = (
        StrategyIntervention(0, 1, neighbor=1, connection=1),
        StrategyIntervention(1, 2, person_type="payment_dependent"),
        StrategyIntervention(1, 3, person_type="payment_dependent"),
    )
    result = simulate_strategy(replace(config, loyalty_rounds=2), inputs, StrategySettings(payment_mode="no_payment"), interventions=events)
    assert_array_equal(result.relationships[:, 0, 1], [-1, 1, 1, -1, -1])


def test_strategy_intervention_preserves_replay_prefix_and_only_changes_target_agent(four_agents):
    config, inputs = four_agents
    config = replace(config, rounds=3)
    inputs = replace(inputs, observations=np.ones(3, dtype=int), recipients=np.tile(inputs.recipients, (3, 1)),
                     contact_uniforms=np.tile(inputs.contact_uniforms, (3, 1)))
    settings = StrategySettings()
    baseline = simulate_strategy(config, inputs, settings)
    edited = simulate_strategy(config, inputs, settings, interventions=(StrategyIntervention(0, 2, payment_strategy="all"),))
    assert_array_equal(edited.beliefs[:2], baseline.beliefs[:2])
    assert_array_equal(edited.wealth[:2], baseline.wealth[:2])
    assert_array_equal(edited.contacts[:, 1:], baseline.contacts[:, 1:])
    assert_array_equal(edited.contacts[1:, 0].sum(axis=1), [3, 3])
    assert_allclose(edited.wealth.sum(axis=1), 100)


def test_uncontacted_conditional_direction_still_ages_each_round(four_agents):
    config, inputs = four_agents
    config = replace(config, rounds=3, loyalty_rounds=2)
    inputs = replace(inputs, observations=np.ones(3, dtype=int), recipients=np.tile(inputs.recipients, (3, 1)),
                     contact_uniforms=np.tile(inputs.contact_uniforms, (3, 1)))
    result = simulate_strategy(
        config, inputs, StrategySettings(), person_types=("ordinary", "payment_dependent", "ordinary", "ordinary"),
        interventions=(StrategyIntervention(2, 1, neighbor=1, connection=1),),
    )
    assert_array_equal(result.contacts[:, 2, 1], False)
    assert_array_equal(result.relationships[:, 2, 1], [-1, 1, -1, -1])
    assert_array_equal(result.persistent_relationships[1:, 2, 1], 1)


def test_equal_split_of_entire_wealth_keeps_empty_payer_nonnegative():
    config = Config(n_agents=100, rounds=2, attachment=1, grid_size=3,
                    public_evidence=False, wealth_fraction=1)
    adjacency = ~np.eye(100, dtype=bool)
    initial_wealth = np.zeros(100)
    initial_wealth[0] = 10.0
    inputs = PreparedInputs(
        theta=np.array([0., .5, 1.]), adjacency=adjacency, relationships=-adjacency.astype(np.int8),
        initial_beliefs=np.tile([.2, .5, .3], (100, 1)), initial_wealth=initial_wealth,
        observations=np.ones(2, dtype=int), recipients=np.tile(np.r_[1, np.zeros(99, dtype=int)], (2, 1)),
        wealth_distribution="uniform", contact_uniforms=np.zeros((2, 100)),
    )
    result = simulate_strategy(config, inputs, StrategySettings(payment_strategy="all"))
    assert result.wealth[1, 0] == 0.0
    assert np.all(result.wealth >= 0)
    assert_allclose(result.transfers.sum(axis=2), result.payments, rtol=1e-14, atol=0)
    assert_allclose(result.wealth.sum(axis=1), 10.0, rtol=1e-14, atol=0)


def test_manual_cash_accounting_and_inactive_cost_proposals(pair):
    config, inputs = pair
    settings = StrategySettings(payment_mode="no_payment")
    baseline = simulate_strategy(config, inputs, settings)
    events = (StrategyIntervention(0, 3, cash_delta=3), StrategyIntervention(0, 3, neighbor=1, connection=1))
    changed = simulate_strategy(config, inputs, settings, interventions=events)
    assert_array_equal(changed.beliefs[:3], baseline.beliefs[:3])
    assert_allclose(changed.wealth.sum(axis=1), [25, 25, 25, 28, 28])
    assert_allclose(changed.external_cash.sum(axis=1), [0, 0, 3, 0])
    proposals = simulate_strategy(config, inputs, replace(settings, incitation_cost=900, reinforcement_scale=700), interventions=events)
    assert_array_equal(changed.beliefs, proposals.beliefs)
    assert_array_equal(changed.wealth, proposals.wealth)


@pytest.mark.parametrize("changes", [
    {}, {"person_type": "unknown"}, {"payment_strategy": "old_preference"},
    {"neighbor": 1, "connection": 0}, {"belief_center": .5}, {"cash_delta": float("nan")},
])
def test_standalone_intervention_rejects_invalid_changes(changes):
    with pytest.raises(ValueError):
        StrategyIntervention(0, 1, **changes)


def test_wasserstein_respects_grid_spacing_shape_and_broadcasting():
    theta = np.array([0., .2, 1.])
    beliefs = np.eye(3)
    distances = wasserstein_distance(beliefs[:, None, :], beliefs[None, :, :], theta)
    assert_allclose(distances, [[0, .2, 1], [.2, 0, .8], [1, .8, 0]])
    assert_allclose(wasserstein_distance(np.array([.5, 0., .5]), np.array([0., 1., 0.]), theta), np.sqrt(.34))
