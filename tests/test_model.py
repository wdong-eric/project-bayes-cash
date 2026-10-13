from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from bayes_cash.model import (
    Config,
    PreparedInputs,
    bayesian_update,
    make_inputs,
    payment_amounts,
    simulate,
    social_update,
)


@pytest.fixture
def directed_round():
    """Only agent 0 can afford to pay, and its recipient is agent 1."""
    config = Config(
        n_agents=3,
        rounds=1,
        attachment=1,
        grid_size=3,
        public_evidence=False,
    )
    inputs = PreparedInputs(
        theta=np.array([0.0, 0.5, 1.0]),
        adjacency=np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool),
        relationships=np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
        initial_beliefs=np.eye(3),
        initial_wealth=np.array([1.0, 0.4, 0.2]),
        observations=np.array([1]),
        recipients=np.array([[1, 0, 1]]),
        wealth_distribution="uniform",
    )
    return config, inputs


@pytest.mark.parametrize(
    ("observation", "expected"),
    [
        (1, [[0.0, 3 / 13, 10 / 13], [0.0, 0.4, 0.6]]),
        (0, [[4 / 7, 3 / 7, 0.0], [0.6, 0.4, 0.0]]),
    ],
)
def test_bayesian_update_matches_analytic_coin_posterior(observation, expected):
    beliefs = np.array([[0.2, 0.3, 0.5], [0.3, 0.4, 0.3]])
    posterior = bayesian_update(beliefs, np.array([0.0, 0.5, 1.0]), observation)
    assert_allclose(posterior, expected)
    assert_array_equal(beliefs, [[0.2, 0.3, 0.5], [0.3, 0.4, 0.3]])


def test_payer_loses_cash_and_influences_the_recipient(directed_round):
    config, inputs = directed_round
    result = simulate(config, inputs, "fixed")

    # Agent 0 spends its entire opening balance. Agent 1 receives its opinion.
    assert_array_equal(result.payments, [[1.0, 0.0, 0.0]])
    assert_allclose(result.wealth[1], [0.0, 1.4, 0.2])
    assert_allclose(
        result.beliefs[1],
        [[1.0, 0.0, 0.0], [1 / 7, 6 / 7, 0.0], [0.0, 0.0, 1.0]],
    )


@pytest.mark.parametrize("mode", ["fixed", "hybrid"])
def test_received_cash_becomes_spendable_only_next_round(directed_round, mode):
    config, inputs = directed_round
    config = replace(config, rounds=2)
    inputs = replace(
        inputs,
        observations=np.array([1, 1]),
        recipients=np.array([[1, 0, 1], [1, 0, 1]]),
    )
    result = simulate(config, inputs, mode)

    assert_array_equal(result.payments, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert_allclose(result.wealth, [[1.0, 0.4, 0.2], [0.0, 1.4, 0.2], [1.0, 0.4, 0.2]])


def test_fractional_payment_uses_opening_payer_wealth(directed_round):
    config, inputs = directed_round
    result = simulate(replace(config, wealth_fraction=0.5), inputs, "fractional")

    assert_allclose(result.payments, [[0.5, 0.2, 0.1]])
    assert_allclose(result.wealth[1], [0.7, 0.8, 0.1])
    # Both incoming payer opinions contribute using their actual payments.
    assert_allclose(result.beliefs[1, 1], np.array([0.5, 6.0, 0.1]) / 6.6)


def test_hybrid_payments_have_affordability_threshold_and_fractional_crossover():
    config = Config(fixed_payment=1.0, wealth_fraction=0.05)
    opening_wealth = np.array([0.0, 0.9, 1.0, 5.0, 20.0, 40.0])

    assert_array_equal(
        payment_amounts(config, opening_wealth, "hybrid"),
        [0.0, 0.0, 1.0, 1.0, 1.0, 2.0],
    )


@pytest.mark.parametrize(
    ("overrides", "equivalent_mode"),
    [({"wealth_fraction": 0.0}, "fixed"), ({"fixed_payment": 0.0}, "fractional")],
)
def test_hybrid_limits_recover_existing_simulations(overrides, equivalent_mode):
    config = replace(Config(n_agents=8, rounds=12, attachment=2), **overrides)
    inputs = make_inputs(config, "loguniform")
    hybrid = simulate(config, inputs, "hybrid")
    equivalent = simulate(config, inputs, equivalent_mode)

    assert_array_equal(hybrid.payments, equivalent.payments)
    assert_array_equal(hybrid.wealth, equivalent.wealth)
    assert_array_equal(hybrid.beliefs, equivalent.beliefs)


def test_hybrid_full_fraction_spends_all_wealth_only_above_threshold():
    config = Config(fixed_payment=1.0, wealth_fraction=1.0)

    assert_array_equal(
        payment_amounts(config, np.array([0.0, 0.9, 1.0, 5.0]), "hybrid"),
        [0.0, 0.0, 1.0, 5.0],
    )


def test_all_neighbor_reference_payments_recover_original_ally_update():
    config = Config(
        n_agents=2,
        rounds=1,
        attachment=1,
        grid_size=3,
        social_rate=0.25,
        reference_payment=2.0,
        fixed_payment=2.0,
        public_evidence=False,
    )
    inputs = PreparedInputs(
        theta=np.array([0.0, 0.5, 1.0]),
        adjacency=np.array([[0, 1], [1, 0]], dtype=bool),
        relationships=np.array([[0, 1], [1, 0]]),
        initial_beliefs=np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]]),
        initial_wealth=np.array([2.0, 2.0]),
        observations=np.array([1]),
        recipients=np.array([[1, 0]]),
        wealth_distribution="uniform",
    )
    result = simulate(config, inputs, "fixed")
    upstream_ally_step = inputs.initial_beliefs + config.social_rate * (
        inputs.initial_beliefs[::-1] - inputs.initial_beliefs
    )
    assert_allclose(result.beliefs[1], upstream_ally_step)
    assert_array_equal(result.wealth[1], inputs.initial_wealth)


def test_larger_payment_has_stronger_effect_on_recipient():
    beliefs = np.array([[1.0, 0.0], [0.0, 1.0]])
    recipients = np.array([1, 0])
    self_weight = np.array([3.0, 3.0])
    relationships = np.array([[0, 1], [1, 0]])
    small = social_update(beliefs, recipients, np.array([1.0, 0.0]), self_weight, relationships, 1.0)
    large = social_update(beliefs, recipients, np.array([3.0, 0.0]), self_weight, relationships, 1.0)

    assert_allclose(small[1], [0.25, 0.75])
    assert_allclose(large[1], [0.5, 0.5])
    assert_array_equal(small[0], beliefs[0])
    assert_array_equal(large[0], beliefs[0])


def test_unpaid_opponents_repel_and_reference_payment_halves_repulsive_weight():
    beliefs = np.array([[0.4, 0.6], [0.6, 0.4]])
    recipients = np.array([1, 0])
    self_weight = np.array([3.0, 3.0])
    relationships = np.array([[0, -1], [-1, 0]])
    unpaid = social_update(beliefs, recipients, np.zeros(2), self_weight, relationships, 1.0)
    reference = social_update(beliefs, recipients, np.ones(2), self_weight, relationships, 1.0)

    # Unpaid repulsion has magnitude 1; a reference-sized payment reduces it to 1/2.
    assert_allclose(unpaid, [[0.35, 0.65], [0.65, 0.35]])
    assert_allclose(reference[1], beliefs[1] + 0.5 / 3.5 * (beliefs[1] - beliefs[0]))


def test_increasing_payment_monotonically_weakens_pair_repulsion():
    beliefs = np.array([[0.4, 0.6], [0.6, 0.4]])
    shifts = []
    for payment in (0.0, 0.1, 1.0, 10.0, 1e12):
        updated = social_update(
            beliefs, np.array([1, 0]), np.full(2, payment), np.full(2, 3.0),
            np.array([[0, -1], [-1, 0]]), 1.0,
        )
        shifts.append(updated[1, 0] - beliefs[1, 0])
    assert np.all(np.diff(shifts) < 0)
    assert shifts[-1] < 1e-12


def test_zero_payment_allies_leave_beliefs_exactly_unchanged():
    beliefs = np.array([[0.4, 0.6], [0.6, 0.4]])
    updated = social_update(
        beliefs, np.array([1, 0]), np.zeros(2), np.full(2, 3.0),
        np.array([[0, 1], [1, 0]]), 1.0,
    )
    assert_array_equal(updated, beliefs)


def test_signed_projection_can_erase_the_true_hypothesis_without_a_floor():
    theta = np.array([0.0, 0.5, 1.0])
    beliefs = np.array([[0.05, 0.9, 0.05], [0.45, 0.1, 0.45]])
    updated = social_update(
        beliefs, np.array([1, 0]), np.zeros(2), np.full(2, 3.0),
        np.array([[0, -1], [-1, 0]]), 1.0,
    )
    assert_array_equal(updated[1], [0.5, 0.0, 0.5])
    assert bayesian_update(updated, theta, 1)[1, 1] == 0.0
    assert_allclose(updated.sum(axis=1), 1.0)


def test_allies_preserve_original_pooling_bitwise_with_multiple_incoming_payments():
    beliefs = np.array([[0.13, 0.87], [0.57, 0.43], [0.21, 0.79]])
    recipients = np.array([1, 0, 1])
    payments = np.array([0.17, 1.23, 2.41])
    self_weight = np.array([3.0, 6.0, 3.0])
    relationships = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    expected = beliefs.copy()
    expected[0] = (3.0 * beliefs[0] + 1.23 * beliefs[1]) / (3.0 + 1.23)
    expected[1] = (6.0 * beliefs[1] + (0.17 * beliefs[0] + 2.41 * beliefs[2])) / (
        6.0 + (0.17 + 2.41)
    )
    actual = social_update(beliefs, recipients, payments, self_weight, relationships, 1.0)
    assert_array_equal(actual, expected)


def test_mixed_incoming_signs_use_total_absolute_weight_in_the_denominator():
    beliefs = np.array([[0.4, 0.6], [0.5, 0.5], [0.7, 0.3]])
    # Agent 0 attracts agent 1 with weight 2; agent 2 repels it with weight 1/2.
    actual = social_update(
        beliefs, np.array([1, 0, 1]), np.array([2.0, 0.0, 1.0]), np.array([3.0, 6.0, 3.0]),
        np.array([[0, 1, 0], [1, 0, -1], [0, -1, 0]]), 1.0,
    )
    expected = beliefs[1] + (
        2.0 * (beliefs[0] - beliefs[1]) - 0.5 * (beliefs[2] - beliefs[1])
    ) / 8.5
    assert_allclose(actual[1], expected)
    assert_array_equal(actual[[0, 2]], beliefs[[0, 2]])


@pytest.mark.parametrize("mode", ["fixed", "fractional", "hybrid"])
def test_edge_signs_leave_all_wealth_and_payment_trajectories_exactly_unchanged(mode):
    ally_config = Config(n_agents=18, rounds=30, attachment=2, seed=129)
    signed_config = replace(ally_config, opponent_fraction=0.5)
    allies = simulate(ally_config, make_inputs(ally_config, "loguniform"), mode)
    signed = simulate(signed_config, make_inputs(signed_config, "loguniform"), mode)
    assert_array_equal(signed.wealth, allies.wealth)
    assert_array_equal(signed.payments, allies.payments)
    assert not np.array_equal(signed.beliefs, allies.beliefs)


def test_edge_sign_draws_are_nested_and_leave_other_random_streams_unchanged():
    config = Config(n_agents=18, rounds=10, attachment=2, seed=129)
    ally = make_inputs(config, "uniform")
    quarter = make_inputs(replace(config, opponent_fraction=0.25), "uniform")
    half = make_inputs(replace(config, opponent_fraction=0.5), "uniform")
    all_opponents = make_inputs(replace(config, opponent_fraction=1.0), "uniform")
    for name in ("theta", "adjacency", "initial_beliefs", "initial_wealth", "observations", "recipients"):
        assert_array_equal(getattr(ally, name), getattr(half, name))
        assert_array_equal(getattr(ally, name), getattr(all_opponents, name))
    assert_array_equal(ally.relationships, ally.adjacency.astype(int))
    assert_array_equal(all_opponents.relationships, -ally.adjacency.astype(int))
    assert np.any(half.relationships == -1)
    assert np.any(half.relationships == 1)
    assert np.any(quarter.relationships == -1)
    assert np.all(half.relationships[quarter.relationships == -1] == -1)
    assert np.any((quarter.relationships == 1) & (half.relationships == -1))
    assert np.all(all_opponents.relationships[half.relationships == -1] == -1)


def test_unaffordable_payments_leave_only_public_bayesian_learning():
    config = Config(n_agents=8, rounds=10, attachment=2, fixed_payment=101.0)
    inputs = make_inputs(config, "uniform")
    result = simulate(config, inputs, "fixed")
    expected = inputs.initial_beliefs.copy()
    for step, observation in enumerate(inputs.observations, start=1):
        expected = bayesian_update(expected, inputs.theta, observation)
        assert_allclose(result.beliefs[step], expected)
    assert_array_equal(result.payments, np.zeros((config.rounds, config.n_agents)))
    assert_allclose(result.wealth, np.broadcast_to(inputs.initial_wealth, result.wealth.shape))


@pytest.mark.parametrize("mode", ["fixed", "fractional", "hybrid"])
@pytest.mark.parametrize("distribution", ["uniform", "loguniform"])
@pytest.mark.parametrize("opponent_fraction", [0.0, 0.25, 1.0])
def test_simulation_preserves_probability_and_conserves_cash(mode, distribution, opponent_fraction):
    config = Config(n_agents=18, rounds=30, attachment=2, seed=129, opponent_fraction=opponent_fraction)
    inputs = make_inputs(config, distribution)
    result = simulate(config, inputs, mode)

    assert np.isfinite(result.beliefs).all()
    assert np.isfinite(result.wealth).all()
    assert (result.beliefs >= 0).all()
    assert (result.wealth >= 0).all()
    assert_allclose(result.beliefs.sum(axis=2), 1.0, rtol=1e-12, atol=1e-12)
    assert_allclose(result.wealth.sum(axis=1), inputs.initial_wealth.sum(), rtol=1e-12)
    assert (result.payments <= result.wealth[:-1] + 1e-12).all()


def test_seeded_inputs_share_exogenous_draws_across_wealth_distributions():
    config = Config(n_agents=12, rounds=10, attachment=2, seed=91)
    uniform = make_inputs(config, "uniform")
    repeated = make_inputs(config, "uniform")
    loguniform = make_inputs(config, "loguniform")
    for name in ("theta", "adjacency", "relationships", "initial_beliefs", "observations", "recipients"):
        assert_array_equal(getattr(uniform, name), getattr(repeated, name))
        assert_array_equal(getattr(uniform, name), getattr(loguniform, name))
    assert_array_equal(uniform.initial_wealth, repeated.initial_wealth)
    uniform_quantiles = (uniform.initial_wealth - config.wealth_min) / (
        config.wealth_max - config.wealth_min
    )
    loguniform_quantiles = np.log(loguniform.initial_wealth / config.wealth_min) / np.log(
        config.wealth_max / config.wealth_min
    )
    assert_allclose(uniform_quantiles, loguniform_quantiles, atol=1e-14)


def test_payment_scenario_order_does_not_mutate_prepared_inputs():
    config = Config(n_agents=8, rounds=12, attachment=2)
    inputs = make_inputs(config, "loguniform")
    prior_copy = inputs.initial_beliefs.copy()
    wealth_copy = inputs.initial_wealth.copy()
    modes = ("fixed", "fractional", "hybrid")
    first_results = {mode: simulate(config, inputs, mode) for mode in modes}
    second_results = {mode: simulate(config, inputs, mode) for mode in reversed(modes)}

    for mode in modes:
        first, second = first_results[mode], second_results[mode]
        assert_array_equal(first.beliefs, second.beliefs)
        assert_array_equal(first.wealth, second.wealth)
        assert_array_equal(first.payments, second.payments)
    assert_array_equal(inputs.initial_beliefs, prior_copy)
    assert_array_equal(inputs.initial_wealth, wealth_copy)


@pytest.mark.parametrize(
    "invalid",
    [
        {"n_agents": 1},
        {"social_rate": 0.0},
        {"social_rate": 1.0},
        {"reference_payment": 0.0},
        {"wealth_min": 0.0},
        {"wealth_min": 101.0},
        {"fixed_payment": -1.0},
        {"wealth_fraction": -0.1},
        {"wealth_fraction": 1.1},
        {"opponent_fraction": -0.1},
        {"opponent_fraction": 1.1},
        {"opponent_fraction": True},
        {"opponent_fraction": np.nan},
    ],
)
def test_invalid_economic_conventions_are_rejected(invalid):
    with pytest.raises(ValueError):
        Config(**invalid)


@pytest.mark.parametrize(
    "invalid",
    [
        {"initial_wealth": np.array([1.0, -0.4, 0.2])},
        {"initial_beliefs": np.array([[1.1, -0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])},
        {"recipients": np.array([[2, 0, 1]])},
        {"observations": np.array([2])},
        {"relationships": np.array([[0, 1], [1, 0]])},
        {"relationships": np.array([[0, -1, 0], [1, 0, 1], [0, 1, 0]])},
        {"relationships": np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])},
        {"relationships": np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])},
        {"relationships": np.array([[0, 2, 0], [2, 0, 1], [0, 1, 0]])},
        {"relationships": np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)},
    ],
)
def test_invalid_prepared_inputs_are_rejected(directed_round, invalid):
    _, inputs = directed_round
    with pytest.raises(ValueError):
        replace(inputs, **invalid)


def test_impossible_evidence_is_reported_instead_of_substituting_a_prior():
    with pytest.raises(ValueError):
        bayesian_update(np.array([[1.0, 0.0, 0.0]]), np.array([0.0, 0.5, 1.0]), 1)


def test_unknown_payment_mode_is_rejected(directed_round):
    config, inputs = directed_round
    with pytest.raises(ValueError, match="payment_mode"):
        simulate(config, inputs, "unknown")
