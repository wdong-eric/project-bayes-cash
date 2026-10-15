from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from bayes_cash.model import Config, Intervention, PreparedInputs, make_inputs, simulate


@pytest.fixture
def intervention_network():
    config = Config(
        n_agents=3, rounds=8, attachment=1, grid_size=3, public_evidence=False,
        preference_probability=1.0, loyalty_rounds=3,
    )
    adjacency = ~np.eye(3, dtype=bool)
    inputs = PreparedInputs(
        theta=np.array([0.0, 0.5, 1.0]), adjacency=adjacency,
        relationships=adjacency.astype(np.int8),
        initial_beliefs=np.array([[0.7, 0.2, 0.1], [0.2, 0.2, 0.6], [0.6, 0.3, 0.1]]),
        initial_wealth=np.full(3, 10.0), observations=np.ones(8, dtype=int),
        recipients=np.tile([1, 0, 0], (8, 1)), wealth_distribution="uniform",
        giving_preferences=np.zeros(3, dtype=np.int8),
        conditional_allies=np.zeros(3, dtype=bool), contact_uniforms=np.zeros((8, 3)),
    )
    return config, inputs


@pytest.mark.parametrize("round_number", [1, 8])
def test_effective_round_is_one_based_and_change_persists(intervention_network, round_number):
    config, inputs = intervention_network
    baseline = simulate(config, inputs, "fixed")
    result = simulate(
        config, inputs, "fixed",
        interventions=(Intervention(agent=0, round=round_number, giving_preference=-1),),
    )
    for name in ("beliefs", "wealth", "relationships"):
        assert_array_equal(getattr(result, name)[:round_number], getattr(baseline, name)[:round_number])
    assert_array_equal(result.recipients[:round_number - 1], baseline.recipients[:round_number - 1])
    assert_array_equal(result.payments[:round_number - 1], baseline.payments[:round_number - 1])
    # An exclusive opponent giver with no opponents cannot pay any neighbor.
    assert_array_equal(result.recipients[round_number - 1:, 0], -1)
    assert_array_equal(result.payments[round_number - 1:, 0], 0.0)


def test_events_sort_by_round_and_equal_round_events_keep_supplied_order(intervention_network):
    config, inputs = intervention_network
    inputs = replace(inputs, recipients=np.tile([2, 0, 0], (config.rounds, 1)))
    result = simulate(config, inputs, "fixed", interventions=(
        Intervention(agent=0, round=4, giving_preference=1),
        Intervention(agent=0, round=2, giving_preference=-1),
        Intervention(agent=0, round=4, giving_preference=0),
    ))
    assert_array_equal(result.recipients[:, 0], [2, -1, -1, 2, 2, 2, 2, 2])


def test_round_one_intervention_matches_the_same_initial_traits(intervention_network):
    config, inputs = intervention_network
    result = simulate(config, inputs, "fixed", interventions=(
        Intervention(agent=0, round=1, giving_preference=-1, conditional_ally=True),
    ))
    initial = replace(inputs, giving_preferences=np.array([-1, 0, 0]),
                      conditional_allies=np.array([True, False, False]))
    reference = simulate(config, initial, "fixed")
    for name in ("beliefs", "wealth", "relationships", "recipients", "payments"):
        assert_array_equal(getattr(result, name), getattr(reference, name))


def test_new_loyalty_starts_its_clock_before_the_effective_round(intervention_network):
    config, inputs = intervention_network
    result = simulate(replace(config, fixed_payment=0.0), inputs, "fixed", interventions=(
        Intervention(agent=1, round=4, conditional_ally=True),
    ))
    assert_array_equal(result.relationships[:, 0, 1], [1] * 6 + [-1] * 3)
    assert_array_equal(result.relationships[:, 2, 1], [1] * 6 + [-1] * 3)
    assert_array_equal(result.relationships[:, 1, 0], 1)


def test_reapplying_enabled_loyalty_preserves_hidden_unpaid_counters(intervention_network):
    config, inputs = intervention_network
    config = replace(config, fixed_payment=0.0)
    inputs = replace(inputs, conditional_allies=np.array([False, True, False]))
    baseline = simulate(config, inputs, "fixed")
    result = simulate(config, inputs, "fixed", interventions=(
        Intervention(agent=1, round=2, conditional_ally=True),
    ))
    assert_array_equal(result.relationships[:, 0, 1], [1] * 3 + [-1] * 6)
    for name in ("beliefs", "wealth", "relationships", "recipients", "payments"):
        assert_array_equal(getattr(result, name), getattr(baseline, name))


def test_disable_freezes_signs_and_reenable_starts_a_new_clock(intervention_network):
    config, inputs = intervention_network
    inputs = replace(inputs, conditional_allies=np.array([False, True, False]))
    result = simulate(replace(config, fixed_payment=0.0), inputs, "fixed", interventions=(
        Intervention(agent=1, round=3, conditional_ally=False),
        Intervention(agent=1, round=5, conditional_ally=True),
    ))
    assert_array_equal(result.relationships[:, 0, 1], [1] * 7 + [-1] * 2)


def test_reenable_keeps_existing_opponent_signs_until_paid(intervention_network):
    config, inputs = intervention_network
    inputs = replace(inputs, conditional_allies=np.array([False, True, False]))
    result = simulate(replace(config, fixed_payment=0.0), inputs, "fixed", interventions=(
        Intervention(agent=1, round=4, conditional_ally=False),
        Intervention(agent=1, round=6, conditional_ally=True),
    ))
    assert_array_equal(result.relationships[:, 0, 1], [1] * 3 + [-1] * 6)


@pytest.mark.parametrize("mode", ["fixed", "fractional", "hybrid"])
def test_interventions_replay_exactly_preserve_prefix_and_leave_inputs_unchanged(mode):
    config = Config(n_agents=10, rounds=12, attachment=2, public_evidence=False,
                    opponent_fraction=0.4, conditional_ally_fraction=0.5)
    inputs = make_inputs(config, "uniform")
    saved = {name: value.copy() for name, value in vars(inputs).items() if isinstance(value, np.ndarray)}
    events = (Intervention(agent=0, round=6, giving_preference=-1),
              Intervention(agent=2, round=9, conditional_ally=True))
    baseline = simulate(config, inputs, mode)
    first = simulate(config, inputs, mode, interventions=events)
    second = simulate(config, inputs, mode, interventions=events)
    for name in ("beliefs", "wealth", "relationships", "recipients", "payments"):
        assert_array_equal(getattr(first, name), getattr(second, name))
        prefix = 5 if name in ("recipients", "payments") else 6
        assert_array_equal(getattr(first, name)[:prefix], getattr(baseline, name)[:prefix])
    for name, value in saved.items():
        assert_array_equal(getattr(inputs, name), value)


def test_reserving_preference_draws_does_not_change_baseline_results(intervention_network):
    config, inputs = intervention_network
    baseline = simulate(config, replace(inputs, contact_uniforms=None), "fixed")
    result = simulate(config, inputs, "fixed")
    for name in ("beliefs", "wealth", "relationships", "recipients", "payments"):
        assert_array_equal(getattr(result, name), getattr(baseline, name))


def test_legacy_inputs_can_enable_loyalty_without_initial_trait_arrays(intervention_network):
    config, inputs = intervention_network
    inputs = replace(inputs, giving_preferences=None, conditional_allies=None, contact_uniforms=None)
    result = simulate(replace(config, fixed_payment=0.0), inputs, "fixed", interventions=(
        Intervention(agent=1, round=1, conditional_ally=True),
    ))
    assert_array_equal(result.relationships[:, 0, 1], [1] * 3 + [-1] * 6)


def test_enabling_preferences_requires_presampled_contact_draws(intervention_network):
    config, inputs = intervention_network
    with pytest.raises(ValueError, match="requires prepared contact_uniforms"):
        simulate(config, replace(inputs, contact_uniforms=None), "fixed", interventions=(
            Intervention(agent=0, round=2, giving_preference=1),
        ))


@pytest.mark.parametrize("fields", [
    {}, {"giving_preference": 2}, {"giving_preference": 1.0}, {"giving_preference": True},
    {"conditional_ally": 1}, {"conditional_ally": "false"},
    {"giving_preference": 0, "agent": -1}, {"giving_preference": 0, "agent": 0.0},
    {"giving_preference": 0, "agent": False}, {"giving_preference": 0, "round": 0},
    {"giving_preference": 0, "round": 1.5}, {"giving_preference": 0, "round": True},
])
def test_intervention_rejects_invalid_fields(fields):
    with pytest.raises(ValueError, match="intervention"):
        Intervention(**({"agent": 0, "round": 1} | fields))


@pytest.mark.parametrize("event", [
    Intervention(agent=3, round=1, giving_preference=0),
    Intervention(agent=0, round=9, conditional_ally=False),
    {"agent": 0, "round": 1, "giving_preference": 0},
])
def test_simulation_rejects_out_of_run_or_unvalidated_events(intervention_network, event):
    config, inputs = intervention_network
    with pytest.raises(ValueError, match="intervention"):
        simulate(config, inputs, "fixed", interventions=(event,))


@pytest.mark.parametrize("mode", ["fixed", "fractional", "hybrid"])
def test_cash_shocks_enter_before_income_and_payments_and_preserve_history(intervention_network, mode):
    config, inputs = intervention_network
    config = replace(config, income_rate=0.1)
    baseline = simulate(config, inputs, mode)
    result = simulate(config, inputs, mode, interventions=(
        Intervention(agent=0, round=3, cash_delta=5.0),
        Intervention(agent=1, round=3, cash_delta=-4.0),
        Intervention(agent=0, round=3, cash_delta=2.0),
    ))
    assert_array_equal(result.beliefs[:3], baseline.beliefs[:3])
    assert_array_equal(result.wealth[:3], baseline.wealth[:3])
    assert_array_equal(result.payments[:2], baseline.payments[:2])
    assert_array_equal(result.external_cash[2], [7.0, -4.0, 0.0])
    assert_array_equal(np.delete(result.external_cash, 2, axis=0), 0.0)
    for step, recipients in enumerate(result.recipients):
        incoming = np.zeros(config.n_agents)
        np.add.at(incoming, recipients, result.payments[step])
        opening = result.wealth[step] + result.external_cash[step]
        assert_allclose(result.wealth[step + 1], 1.1 * opening - result.payments[step] + incoming)
    assert_allclose(result.wealth[3].sum(), 1.1 * (baseline.wealth[2].sum() + 3.0))


def test_withdrawal_cannot_spend_future_income_or_earlier_history(intervention_network):
    config, inputs = intervention_network
    with pytest.raises(ValueError, match="before round 1 for agent 0.*withdraws more"):
        simulate(replace(config, income_rate=1.0), inputs, "fixed", interventions=(
            Intervention(agent=0, round=1, cash_delta=-10.5),
        ))
    assert_array_equal(inputs.initial_wealth, 10.0)


def test_cash_can_be_withdrawn_to_zero_without_clipping(intervention_network):
    config, inputs = intervention_network
    result = simulate(config, inputs, "fixed", interventions=tuple(
        Intervention(agent=agent, round=1, cash_delta=-10.0) for agent in range(3)
    ))
    assert_array_equal(result.wealth[0], 10.0)
    assert_array_equal(result.wealth[1:], 0.0)
    assert_array_equal(result.payments, 0.0)
    assert_array_equal(result.external_cash[0], -10.0)
    assert_array_equal(result.beliefs[1], inputs.initial_beliefs)


def test_cash_shock_rejects_finite_arithmetic_overflow(intervention_network):
    config, inputs = intervention_network
    with pytest.raises(ValueError, match="before round 1 for agent 0.*finite cash"):
        simulate(config, inputs, "fixed", interventions=(
            Intervention(agent=0, round=1, cash_delta=1e308),
            Intervention(agent=0, round=1, cash_delta=1e308),
        ))


def test_scheduled_withdrawal_can_prevent_income_overflow(intervention_network):
    config, inputs = intervention_network
    config = replace(config, income_rate=1.0, fixed_payment=0.0)
    inputs = replace(inputs, initial_wealth=np.full(3, 1e307))
    with pytest.raises(ValueError, match="income growth exceeds finite"):
        simulate(config, inputs, "fixed")
    result = simulate(config, inputs, "fixed", interventions=tuple(
        Intervention(agent=agent, round=1, cash_delta=-1e307) for agent in range(3)
    ))
    assert_array_equal(result.wealth[1:], 0.0)


def test_total_cash_overflow_is_rejected_even_when_each_balance_is_finite(intervention_network):
    config, inputs = intervention_network
    config = replace(config, income_rate=1.0, fixed_payment=0.0)
    with pytest.raises(ValueError, match="simulation exceeded finite"):
        simulate(config, inputs, "fixed", interventions=tuple(
            Intervention(agent=agent, round=1, cash_delta=4e307) for agent in range(3)
        ))


@pytest.mark.parametrize("strength", [0.0, 0.35, 1.0])
def test_belief_shock_blends_grid_probability_masses_and_preserves_prefix(intervention_network, strength):
    config, inputs = intervention_network
    config = replace(config, fixed_payment=0.0)
    baseline = simulate(config, inputs, "fixed")
    result = simulate(config, inputs, "fixed", interventions=(
        Intervention(agent=0, round=3, belief_center=0.65, belief_width=0.17, belief_strength=strength),
    ))
    target = np.exp(-0.5 * ((inputs.theta - 0.65) / 0.17) ** 2)
    target /= target.sum()
    expected = (1.0 - strength) * inputs.initial_beliefs[0] + strength * target
    assert_array_equal(result.beliefs[:3], baseline.beliefs[:3])
    assert_allclose(result.beliefs[3:, 0], np.tile(expected, (6, 1)))
    assert_allclose(result.beliefs.sum(axis=2), 1.0)
    assert_array_equal(result.wealth, baseline.wealth)
    if strength == 0:
        assert_array_equal(result.beliefs, baseline.beliefs)


def test_belief_shock_precedes_public_evidence(intervention_network):
    config, inputs = intervention_network
    config = replace(config, public_evidence=True, fixed_payment=0.0)
    result = simulate(config, inputs, "fixed", interventions=(
        Intervention(agent=0, round=1, belief_center=0.4, belief_width=0.3, belief_strength=1.0),
    ))
    expected = np.exp(-0.5 * ((inputs.theta - 0.4) / 0.3) ** 2) * inputs.theta
    expected /= expected.sum()
    assert_allclose(result.beliefs[1, 0], expected)
    assert_array_equal(result.beliefs[0], inputs.initial_beliefs)


def test_tiny_belief_width_uses_stable_weights_or_explicitly_rejects_overflow(intervention_network):
    config, inputs = intervention_network
    result = simulate(replace(config, fixed_payment=0.0), inputs, "fixed", interventions=(
        Intervention(agent=0, round=1, belief_center=0.4, belief_width=1e-4, belief_strength=1.0),
    ))
    assert_array_equal(result.beliefs[1, 0], [0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="before round 1 for agent 0.*finite Gaussian"):
        simulate(config, inputs, "fixed", interventions=(
            Intervention(agent=0, round=1, belief_center=0.4, belief_width=1e-300, belief_strength=1.0),
        ))


def test_connection_removal_changes_contacts_and_degree_weight(intervention_network):
    config, inputs = intervention_network
    result = simulate(config, inputs, "fixed", interventions=(
        Intervention(agent=0, round=1, neighbor=1, connection=0),
    ))
    assert_array_equal(result.relationships[0], inputs.relationships)
    assert_array_equal(result.relationships[1:, 0, 1], 0)
    assert_array_equal(result.relationships[1:, 1, 0], 0)
    assert_array_equal(result.recipients[:, 0], 2)
    assert_array_equal(result.recipients[:, 1], 2)
    assert_array_equal(result.recipients[:, 2], inputs.recipients[:, 2])
    beliefs = inputs.initial_beliefs
    assert_allclose(result.beliefs[1, 0], (3.0 * beliefs[0] + beliefs[2]) / 4.0)
    assert_array_equal(result.beliefs[1, 1], beliefs[1])
    assert_allclose(result.beliefs[1, 2], (6.0 * beliefs[2] + beliefs[0] + beliefs[1]) / 8.0)


def test_connection_addition_uses_reserved_draws_only_for_changed_neighborhoods(intervention_network):
    config, inputs = intervention_network
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    inputs = replace(inputs, adjacency=adjacency, relationships=adjacency.astype(np.int8),
                     recipients=np.tile([1, 0, 1], (config.rounds, 1)),
                     contact_uniforms=np.full((config.rounds, 3), 0.9))
    result = simulate(config, inputs, "fixed", interventions=(
        Intervention(agent=0, round=1, neighbor=2, connection=1),
    ))
    assert_array_equal(result.recipients[:, 0], 2)
    # Agent 1 retains its original integer draw despite the reserved draw of 0.9.
    assert_array_equal(result.recipients[:, 1], 0)
    assert_array_equal(result.recipients[:, 2], 1)
    assert_array_equal(result.relationships[1:, 0, 2], 1)
    assert_array_equal(result.relationships[1:, 2, 0], 1)
    assert_allclose(result.beliefs[1, 0], (6.0 * inputs.initial_beliefs[0] + inputs.initial_beliefs[1]) / 7.0)


@pytest.mark.parametrize("preference", [-1, 0, 1])
def test_isolated_agent_has_no_contacts_but_keeps_evidence_and_income(intervention_network, preference):
    config, inputs = intervention_network
    config = replace(config, public_evidence=True, income_rate=0.1)
    result = simulate(config, inputs, "fixed", interventions=(
        Intervention(agent=0, round=1, neighbor=1, connection=0, giving_preference=preference),
        Intervention(agent=0, round=1, neighbor=2, connection=0),
        Intervention(agent=0, round=2, neighbor=2, connection=0),
    ))
    assert_array_equal(result.recipients[:, 0], -1)
    assert np.all(result.recipients[:, 1:] != 0)
    assert_array_equal(result.payments[:, 0], 0.0)
    assert_allclose(result.wealth[:, 0], 10.0 * 1.1 ** np.arange(config.rounds + 1))
    expected = inputs.initial_beliefs[0] * inputs.theta
    assert_allclose(result.beliefs[1, 0], expected / expected.sum())


def test_connection_restore_reuses_original_draws_and_resets_loyalty_clock(intervention_network):
    config, inputs = intervention_network
    inputs = replace(inputs, conditional_allies=np.array([True, True, False]))
    result = simulate(replace(config, fixed_payment=0.0), inputs, "fixed", interventions=(
        Intervention(agent=0, round=2, neighbor=1, connection=0),
        Intervention(agent=0, round=4, neighbor=1, connection=1),
    ))
    assert_array_equal(result.relationships[:, 0, 1], [1, 1, 0, 0, 1, 1, -1, -1, -1])
    assert_array_equal(result.relationships[:, 1, 0], result.relationships[:, 0, 1])
    assert_array_equal(result.recipients[3:], inputs.recipients[3:])


def test_connection_sign_changes_both_directions_and_affects_preferences(intervention_network):
    config, inputs = intervention_network
    result = simulate(config, inputs, "fixed", interventions=(
        Intervention(agent=0, round=1, neighbor=2, connection=-1, giving_preference=-1),
    ))
    assert_array_equal(result.relationships[1:, 0, 2], -1)
    assert_array_equal(result.relationships[1:, 2, 0], -1)
    assert_array_equal(result.recipients[:, 0], 2)


@pytest.mark.parametrize("mode", ["fixed", "fractional", "hybrid"])
def test_mixed_interventions_replay_exactly_and_do_not_mutate_inputs(intervention_network, mode):
    config, inputs = intervention_network
    saved = {name: value.copy() for name, value in vars(inputs).items() if isinstance(value, np.ndarray)}
    events = (
        Intervention(agent=0, round=3, cash_delta=4.0, belief_center=0.7,
                     belief_width=0.2, belief_strength=0.4, neighbor=1, connection=0),
        Intervention(agent=1, round=4, conditional_ally=True, giving_preference=-1),
        Intervention(agent=0, round=5, neighbor=1, connection=-1),
    )
    baseline = simulate(config, inputs, mode)
    first = simulate(config, inputs, mode, interventions=events)
    second = simulate(config, inputs, mode, interventions=events)
    for name in ("beliefs", "wealth", "relationships", "recipients", "payments", "external_cash"):
        assert_array_equal(getattr(first, name), getattr(second, name))
        prefix = 3 if name in ("beliefs", "wealth", "relationships") else 2
        assert_array_equal(getattr(first, name)[:prefix], getattr(baseline, name)[:prefix])
    for name, value in saved.items():
        assert_array_equal(getattr(inputs, name), value)


@pytest.mark.parametrize("fields", [
    {"cash_delta": float("nan")}, {"cash_delta": float("inf")}, {"cash_delta": True},
    {"cash_delta": "1"}, {"belief_center": 0.5},
    {"belief_center": -0.1, "belief_width": 0.2, "belief_strength": 0.5},
    {"belief_center": 1.1, "belief_width": 0.2, "belief_strength": 0.5},
    {"belief_center": 0.5, "belief_width": 0.0, "belief_strength": 0.5},
    {"belief_center": 0.5, "belief_width": -0.2, "belief_strength": 0.5},
    {"belief_center": 0.5, "belief_width": float("inf"), "belief_strength": 0.5},
    {"belief_center": 0.5, "belief_width": 0.2, "belief_strength": -0.1},
    {"belief_center": 0.5, "belief_width": 0.2, "belief_strength": 1.1},
    {"belief_center": 0.5, "belief_width": 0.2, "belief_strength": True},
    {"neighbor": 1}, {"connection": 1}, {"neighbor": 0, "connection": 1},
    {"neighbor": -1, "connection": 1}, {"neighbor": True, "connection": 1},
    {"neighbor": 1.0, "connection": 1}, {"neighbor": 1, "connection": True},
    {"neighbor": 1, "connection": 1.0}, {"neighbor": 1, "connection": 2},
])
def test_extended_intervention_rejects_invalid_fields(fields):
    with pytest.raises(ValueError, match="intervention"):
        Intervention(agent=0, round=1, **fields)


def test_connection_neighbor_must_be_in_run_and_contact_draws_are_required(intervention_network):
    config, inputs = intervention_network
    with pytest.raises(ValueError, match="intervention neighbor"):
        simulate(config, inputs, "fixed", interventions=(
            Intervention(agent=0, round=1, neighbor=3, connection=1),
        ))
    with pytest.raises(ValueError, match="requires prepared contact_uniforms"):
        simulate(config, replace(inputs, contact_uniforms=None), "fixed", interventions=(
            Intervention(agent=0, round=1, neighbor=1, connection=0),
        ))
