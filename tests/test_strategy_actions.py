"""Cash-burning actions, chronological execution, and reproducible policies."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from bayes_cash.model import Config, PreparedInputs
from bayes_cash.strategy import (
    StrategyIntervention, StrategySettings, initialize_action_uniforms, simulate_strategy,
)


@pytest.fixture
def action_run():
    config = Config(n_agents=4, rounds=3, attachment=1, grid_size=3,
                    public_evidence=False, loyalty_rounds=2, fixed_payment=1)
    adjacency = ~np.eye(4, dtype=bool)
    inputs = PreparedInputs(
        theta=np.array([0., .5, 1.]), adjacency=adjacency,
        relationships=-adjacency.astype(np.int8),
        initial_beliefs=np.array([[.7, .2, .1], [.2, .5, .3], [.1, .2, .7], [.3, .3, .4]]),
        initial_wealth=np.full(4, 10.), observations=np.ones(3, dtype=int),
        recipients=np.tile([1, 0, 0, 0], (3, 1)), wealth_distribution="uniform",
        contact_uniforms=np.zeros((3, 4)),
    )
    return config, inputs, StrategySettings(payment_mode="no_payment")


def assert_burn_accounting(result):
    burn = result.incitation_spend + result.reinforcement_spend
    expected = (result.wealth[:-1] + result.external_cash - result.payments
                + result.transfers.sum(axis=1) - burn)
    assert_allclose(result.wealth[1:], expected, rtol=1e-13, atol=1e-13)
    assert_allclose(np.diff(result.wealth.sum(axis=1)),
                    result.external_cash.sum(axis=1) - burn.sum(axis=1), atol=1e-12)
    assert np.all(result.wealth >= 0)


@pytest.mark.parametrize("values", [
    {"incite_b": 1}, {"incite_b": True, "incite_c": 2},
    {"incite_b": 0, "incite_c": 2}, {"incite_b": 1, "incite_c": 1},
    {"incite_b": -1, "incite_c": 2}, {"reinforcement_spend": 0},
    {"reinforcement_spend": -1}, {"reinforcement_spend": float("inf")},
    {"incite_b": 1, "incite_c": 2, "reinforcement_spend": 1},
    {"reinforcement_spend": 1, "cash_delta": 1},
    {"reinforcement_spend": 1, "person_type": "junzi"},
    {"reinforcement_spend": 1, "automatic_incitation_probability": .5},
    {"automatic_incitation_probability": .7, "automatic_reinforcement_probability": .4},
    {"automatic_reinforcement_probability": float("nan")}, {"automatic_reinforcement_spend": 0},
])
def test_paid_entries_and_policy_edits_validate_without_ambiguous_mixed_phases(values):
    with pytest.raises(ValueError):
        StrategyIntervention(0, 1, **values)


@pytest.mark.parametrize("values", [
    {"automatic_incitation_probability": -0.1}, {"automatic_reinforcement_probability": True},
    {"automatic_incitation_probability": .7, "automatic_reinforcement_probability": .4},
    {"automatic_reinforcement_spend": 0},
])
def test_automatic_global_settings_are_validated(values):
    with pytest.raises(ValueError):
        StrategySettings(**values)


def test_action_uniforms_are_reproducible_separate_and_immutable(action_run):
    config, inputs, settings = action_run
    draws = initialize_action_uniforms(config)
    assert draws.shape == (3, 4, 3)
    assert np.all((draws >= 0) & (draws < 1))
    assert_array_equal(draws, initialize_action_uniforms(config))
    assert not np.array_equal(draws[:, :, 0], inputs.contact_uniforms)
    saved = draws.copy()
    draws.flags.writeable = False
    simulate_strategy(config, inputs, settings, action_uniforms=draws)
    assert_array_equal(draws, saved)
    for invalid in (np.zeros((3, 4)), np.full((3, 4, 3), 1.), np.full((3, 4, 3), np.nan)):
        with pytest.raises(ValueError, match="action_uniforms"):
            simulate_strategy(config, inputs, settings, action_uniforms=invalid)


def test_scheduled_incitation_uses_base_ally_and_clears_both_loyalty_directions(action_run):
    config, inputs, settings = action_run
    events = (
        StrategyIntervention(1, 1, neighbor=2, connection=1),
        StrategyIntervention(0, 1, incite_b=1, incite_c=2),
    )
    result = simulate_strategy(config, inputs, replace(settings, incitation_cost=2), interventions=events,
                               person_types=("ordinary", "payment_dependent", "payment_dependent", "ordinary"))
    assert_array_equal(result.persistent_relationships[1:, 1, 2], -1)
    assert_array_equal(result.relationships[1:, 1, 2], -1)
    assert_array_equal(result.relationships[1:, 2, 1], -1)
    assert_allclose(result.incitation_spend[:, 0], [2, 0, 0])
    assert result.actions == [{"round": 1, "agent": 0, "action": "incitation", "source": "scheduled",
                               "status": "executed", "cost": 2., "incite_b": 1, "incite_c": 2, "reason": None}]
    assert_burn_accounting(result)


def test_base_alliance_is_eligible_even_if_temporary_responses_already_expired(action_run):
    config, inputs, settings = action_run
    result = simulate_strategy(
        replace(config, loyalty_rounds=1), inputs, settings,
        person_types=("ordinary", "payment_dependent", "payment_dependent", "ordinary"),
        interventions=(StrategyIntervention(1, 1, neighbor=2, connection=1),
                       StrategyIntervention(0, 1, incite_b=1, incite_c=2)),
    )
    assert result.actions[0]["status"] == "executed"
    assert result.persistent_relationships[1, 1, 2] == -1


@pytest.mark.parametrize("case,match", [("not_allied", "pair_not_allied"), ("junzi", "junzi"),
                                         ("unaffordable", "insufficient_cash"), ("outside", "targets")])
def test_invalid_scheduled_incitation_rejects_run_without_mutating_inputs(action_run, case, match):
    config, inputs, settings = action_run
    saved_wealth, saved_beliefs = inputs.initial_wealth.copy(), inputs.initial_beliefs.copy()
    types = ("ordinary", "junzi" if case == "junzi" else "ordinary", "ordinary", "ordinary")
    events = () if case == "not_allied" else (StrategyIntervention(1, 1, neighbor=2, connection=1),)
    event = StrategyIntervention(0, 1, incite_b=1, incite_c=4 if case == "outside" else 2)
    if case == "unaffordable":
        settings = replace(settings, incitation_cost=11)
    with pytest.raises(ValueError, match=match):
        simulate_strategy(config, inputs, settings, person_types=types, interventions=(*events, event))
    assert_array_equal(inputs.initial_wealth, saved_wealth)
    assert_array_equal(inputs.initial_beliefs, saved_beliefs)


def test_paid_actions_follow_normal_settlement_and_can_use_received_cash(action_run):
    config, inputs, settings = action_run
    inputs = replace(inputs, initial_wealth=np.array([0., 10., 10., 10.]))
    baseline = simulate_strategy(config, inputs, replace(settings, payment_mode="fixed"))
    result = simulate_strategy(config, inputs, replace(settings, payment_mode="fixed"),
                               interventions=(StrategyIntervention(0, 1, reinforcement_spend=3),))
    assert result.transfers[0, :, 0].sum() == 3
    assert result.reinforcement_spend[0, 0] == 3
    assert result.wealth[1, 0] == 0
    alpha = -np.expm1(-3 / settings.reinforcement_scale)
    assert_allclose(result.beliefs[1, 0], (1 - alpha) * baseline.beliefs[1, 0] + alpha * inputs.initial_beliefs[0])
    assert_burn_accounting(result)


def test_pre_round_cash_edit_runs_before_an_earlier_listed_post_round_action(action_run):
    config, inputs, settings = action_run
    events = (StrategyIntervention(0, 1, reinforcement_spend=11), StrategyIntervention(0, 1, cash_delta=1))
    result = simulate_strategy(config, inputs, settings, interventions=events)
    assert result.wealth[1, 0] == 0
    assert result.external_cash[0, 0] == 1
    assert result.actions[0]["cost"] == 11
    assert_burn_accounting(result)


def test_scheduled_action_order_consumes_live_funds_and_rejects_overdraft(action_run):
    config, inputs, settings = action_run
    events = (StrategyIntervention(0, 1, reinforcement_spend=6), StrategyIntervention(0, 1, reinforcement_spend=5))
    with pytest.raises(ValueError, match="insufficient_cash"):
        simulate_strategy(config, inputs, settings, interventions=events)
    result = simulate_strategy(config, inputs, settings, interventions=(events[0], replace(events[1], reinforcement_spend=4)))
    assert [record["cost"] for record in result.actions] == [6, 4]
    assert result.wealth[1, 0] == 0
    assert_burn_accounting(result)


def test_scheduled_incitation_can_break_this_rounds_new_alliance_and_next_round_can_restore_it(action_run):
    config, inputs, settings = action_run
    settings = replace(settings, payment_mode="fixed", payment_strategy="all")
    config = replace(config, opponent_threshold=.01)
    result = simulate_strategy(config, inputs, settings,
                               interventions=(StrategyIntervention(0, 1, incite_b=1, incite_c=2),))
    assert result.persistent_relationships[1, 1, 2] == -1
    assert result.persistent_relationships[2, 1, 2] == 1
    assert result.relationships[1, 1, 2] == -1
    assert_burn_accounting(result)


def test_payment_can_restore_temporary_loyalty_after_incitation_without_restoring_base(action_run):
    config, inputs, settings = action_run
    config = replace(config, fixed_payment=3, opponent_threshold=1)
    settings = replace(settings, payment_mode="fixed", payment_strategy="all")
    result = simulate_strategy(
        config, inputs, settings, person_types=("ordinary", "payment_dependent", "payment_dependent", "ordinary"),
        interventions=(StrategyIntervention(1, 1, neighbor=2, connection=1),
                       StrategyIntervention(0, 1, incite_b=1, incite_c=2)),
    )
    assert_array_equal(result.persistent_relationships[1:, 1, 2], -1)
    assert_array_equal(result.relationships[1:, 1, 2], [-1, 1, 1])
    assert_array_equal(result.relationships[1:, 2, 1], [-1, 1, 1])
    assert_burn_accounting(result)


def test_reinforcement_targets_original_pmf_after_manual_blend_and_restores_erased_support(action_run):
    config, inputs, settings = action_run
    edited = StrategyIntervention(0, 1, belief_center=1, belief_width=.01, belief_strength=1)
    result = simulate_strategy(config, inputs, settings, person_types=("junzi",) * 4,
                               interventions=(edited, StrategyIntervention(0, 1, reinforcement_spend=2)))
    alpha = -np.expm1(-2 / settings.reinforcement_scale)
    expected = (1 - alpha) * np.array([0., 0., 1.]) + alpha * inputs.initial_beliefs[0]
    assert_allclose(result.beliefs[1, 0], expected, rtol=1e-14, atol=0)
    assert result.beliefs[1, 0, 0] > 0
    assert result.wealth[1, 0] == 8
    assert_burn_accounting(result)


def test_tiny_reinforcement_spend_and_repeated_actions_keep_exponential_composition(action_run):
    config, inputs, settings = action_run
    edit = StrategyIntervention(0, 1, belief_center=1, belief_width=.01, belief_strength=1)
    tiny = simulate_strategy(config, inputs, settings, person_types=("junzi",) * 4,
                             interventions=(edit, StrategyIntervention(0, 1, reinforcement_spend=1e-16)))
    assert tiny.beliefs[1, 0, 0] > 0
    assert_allclose(tiny.beliefs[1, 0, 0], -np.expm1(-1e-17) * .7, rtol=1e-14, atol=0)
    repeated = simulate_strategy(config, inputs, settings, person_types=("junzi",) * 4,
                                 interventions=(edit, StrategyIntervention(0, 1, reinforcement_spend=2),
                                                StrategyIntervention(0, 1, reinforcement_spend=3)))
    single = simulate_strategy(config, inputs, settings, person_types=("junzi",) * 4,
                               interventions=(edit, StrategyIntervention(0, 1, reinforcement_spend=5)))
    assert_allclose(repeated.beliefs, single.beliefs, atol=1e-15)
    assert_array_equal(repeated.wealth, single.wealth)
    assert_array_equal(repeated.reinforcement_spend, single.reinforcement_spend)


def test_actions_preserve_prefix_and_input_arrays(action_run):
    config, inputs, settings = action_run
    draws = initialize_action_uniforms(config)
    baseline = simulate_strategy(config, inputs, settings, action_uniforms=draws)
    modified = simulate_strategy(config, inputs, settings, action_uniforms=draws,
                                 interventions=(StrategyIntervention(0, 3, reinforcement_spend=1),))
    assert_array_equal(modified.beliefs[:3], baseline.beliefs[:3])
    assert_array_equal(modified.wealth[:3], baseline.wealth[:3])
    assert_array_equal(modified.transfers[:2], baseline.transfers[:2])
    replay = simulate_strategy(config, inputs, settings, action_uniforms=draws,
                               interventions=(StrategyIntervention(0, 3, reinforcement_spend=1),))
    assert_array_equal(replay.beliefs, modified.beliefs)
    assert replay.actions == modified.actions


def test_automatic_categorical_thresholds_choose_only_one_action_per_actor(action_run):
    config, inputs, settings = action_run
    settings = replace(settings, automatic_incitation_probability=.25, automatic_reinforcement_probability=.5)
    draws = np.full((3, 4, 3), .99)
    draws[0, :, 0] = [.1, .25, .7499, .75]
    draws[0, :, 2] = [.4, .3, .2, .1]
    result = simulate_strategy(config, inputs, settings, action_uniforms=draws,
                               interventions=(StrategyIntervention(1, 1, neighbor=2, connection=1),))
    assert [(event["agent"], event["action"]) for event in result.actions] == [
        (2, "reinforcement"), (1, "reinforcement"), (0, "incitation"),
    ]
    assert all(event["source"] == "automatic" and event["status"] == "executed" for event in result.actions)
    assert_array_equal(result.incitation_spend[0], [1, 0, 0, 0])
    assert_array_equal(result.reinforcement_spend[0], [0, 1, 1, 0])
    assert_burn_accounting(result)


@pytest.mark.parametrize("first_actor", [0, 3])
def test_automatic_random_priority_controls_live_pair_availability(action_run, first_actor):
    config, inputs, settings = action_run
    settings = replace(settings, automatic_incitation_probability=1)
    draws = np.zeros((3, 4, 3))
    draws[:, :, 2] = [.2, .8, .9, .3]
    draws[0, first_actor, 2] = .1
    result = simulate_strategy(config, inputs, settings, action_uniforms=draws,
                               interventions=(StrategyIntervention(1, 1, neighbor=2, connection=1),))
    events = [event for event in result.actions if event["round"] == 1]
    assert events[0]["agent"] == first_actor
    assert events[0]["status"] == "executed"
    assert all(event["status"] == "skipped" and event["cost"] == 0 and event["reason"] == "no_eligible_pair"
               for event in events[1:])
    assert_array_equal(result.incitation_spend[0] > 0, np.arange(4) == first_actor)
    assert_burn_accounting(result)


@pytest.mark.parametrize("pair_draw,expected", [(0., (1, 2)), (.999, (1, 3))])
def test_automatic_pair_draw_uses_sorted_eligible_pairs_and_excludes_actor(action_run, pair_draw, expected):
    config, inputs, settings = action_run
    draws = np.zeros((3, 4, 3))
    draws[:, 0, 1] = pair_draw
    events = (
        StrategyIntervention(0, 1, automatic_incitation_probability=1),
        StrategyIntervention(0, 1, neighbor=1, connection=1),
        StrategyIntervention(1, 1, neighbor=2, connection=1),
        StrategyIntervention(1, 1, neighbor=3, connection=1),
    )
    result = simulate_strategy(config, inputs, settings, action_uniforms=draws, interventions=events)
    assert (result.actions[0]["incite_b"], result.actions[0]["incite_c"]) == expected
    assert result.persistent_relationships[1, 0, 1] == 1
    assert_burn_accounting(result)


def test_automatic_incitation_excludes_junzi_targets_but_junzi_actor_can_spend(action_run):
    config, inputs, settings = action_run
    events = (StrategyIntervention(0, 1, automatic_incitation_probability=1),
              StrategyIntervention(1, 1, neighbor=2, connection=1),
              StrategyIntervention(1, 1, neighbor=3, connection=1))
    result = simulate_strategy(config, inputs, settings, interventions=events,
                               person_types=("junzi", "ordinary", "ordinary", "junzi"),
                               action_uniforms=np.full((3, 4, 3), .99))
    assert (result.actions[0]["incite_b"], result.actions[0]["incite_c"]) == (1, 2)
    assert result.actions[0]["status"] == "executed"
    assert result.persistent_relationships[1, 1, 3] == 1
    assert all(event["reason"] == "no_eligible_pair" for event in result.actions[1:])
    assert_burn_accounting(result)


def test_automatic_insufficient_cash_records_selected_pair_without_charging(action_run):
    config, inputs, settings = action_run
    inputs = replace(inputs, initial_wealth=np.array([0., 10., 10., 10.]))
    result = simulate_strategy(config, inputs, settings, action_uniforms=np.zeros((3, 4, 3)),
                               interventions=(StrategyIntervention(0, 1, automatic_incitation_probability=1),
                                              StrategyIntervention(1, 1, neighbor=2, connection=1)))
    assert all(event["reason"] == "insufficient_cash" and event["cost"] == 0
               and (event["incite_b"], event["incite_c"]) == (1, 2) for event in result.actions)
    assert_array_equal(result.incitation_spend, 0)
    assert_array_equal(result.persistent_relationships[1:, 1, 2], 1)
    assert_burn_accounting(result)


def test_scheduled_actions_consume_cash_before_automatic_actions(action_run):
    config, inputs, settings = action_run
    result = simulate_strategy(config, inputs, settings, action_uniforms=np.zeros((3, 4, 3)),
                               interventions=(StrategyIntervention(0, 1, automatic_reinforcement_probability=1),
                                              StrategyIntervention(0, 1, reinforcement_spend=10)))
    assert result.actions[0]["source"] == "scheduled"
    assert result.actions[0]["cost"] == 10
    assert all(event["source"] == "automatic" and event["reason"] == "insufficient_cash"
               for event in result.actions[1:])
    assert result.wealth[1, 0] == 0
    assert_burn_accounting(result)


def test_scheduled_incitation_changes_automatic_eligibility_in_the_same_round(action_run):
    config, inputs, settings = action_run
    events = (StrategyIntervention(0, 1, automatic_incitation_probability=1),
              StrategyIntervention(1, 1, neighbor=2, connection=1),
              StrategyIntervention(3, 1, incite_b=1, incite_c=2))
    result = simulate_strategy(config, inputs, settings, interventions=events, action_uniforms=np.zeros((3, 4, 3)))
    assert result.actions[0]["agent"] == 3 and result.actions[0]["source"] == "scheduled"
    assert result.actions[1]["agent"] == 0 and result.actions[1]["reason"] == "no_eligible_pair"


def test_individual_automatic_policy_override_persists_and_retains_fractional_values(action_run):
    config, inputs, settings = action_run
    settings = replace(settings, automatic_incitation_probability=0, automatic_reinforcement_probability=0,
                       automatic_reinforcement_spend=1)
    draws = np.full((3, 4, 3), .4)
    event = StrategyIntervention(0, 2, automatic_reinforcement_probability=.5, automatic_reinforcement_spend=.5)
    result = simulate_strategy(config, inputs, settings, interventions=(event,), action_uniforms=draws)
    assert [(record["round"], record["agent"], record["cost"]) for record in result.actions] == [(2, 0, .5), (3, 0, .5)]
    assert_array_equal(result.reinforcement_spend[:, 0], [0, .5, .5])
    assert_array_equal(result.reinforcement_spend[:, 1:], 0)
    assert_burn_accounting(result)


def test_partial_policy_edit_validates_combined_live_probabilities(action_run):
    config, inputs, settings = action_run
    settings = replace(settings, automatic_incitation_probability=.8)
    with pytest.raises(ValueError, match="sum to at most 1"):
        simulate_strategy(config, inputs, settings,
                          interventions=(StrategyIntervention(0, 1, automatic_reinforcement_probability=.3),))
    # A combined edit is checked as a whole, without rejecting an intermediate assignment.
    settings = replace(settings, automatic_incitation_probability=.1, automatic_reinforcement_probability=.8)
    result = simulate_strategy(config, inputs, settings, action_uniforms=np.full((3, 4, 3), .99),
                               interventions=(StrategyIntervention(0, 1, automatic_incitation_probability=.8,
                                                                  automatic_reinforcement_probability=.1),))
    assert result.actions == []


def test_shared_automatic_draws_replay_exactly_and_do_not_change_original_inputs(action_run):
    config, inputs, settings = action_run
    settings = replace(settings, automatic_incitation_probability=.25, automatic_reinforcement_probability=.5,
                       automatic_reinforcement_spend=.25)
    draws = initialize_action_uniforms(config)
    initial_beliefs, initial_wealth, draw_copy = inputs.initial_beliefs.copy(), inputs.initial_wealth.copy(), draws.copy()
    first = simulate_strategy(config, inputs, settings, action_uniforms=draws)
    again = simulate_strategy(config, inputs, settings, action_uniforms=draw_copy)
    assert first.actions == again.actions
    assert_array_equal(first.wealth, again.wealth)
    assert_array_equal(first.beliefs, again.beliefs)
    assert_array_equal(inputs.initial_beliefs, initial_beliefs)
    assert_array_equal(inputs.initial_wealth, initial_wealth)
    assert_array_equal(draws, draw_copy)
    assert_burn_accounting(first)
