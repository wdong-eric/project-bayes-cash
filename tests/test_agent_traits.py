from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from bayes_cash.model import Config, PreparedInputs, contact_weights, make_inputs, simulate, social_update


@pytest.fixture
def trait_network():
    config = Config(n_agents=4, rounds=1, attachment=1, grid_size=3, public_evidence=False)
    adjacency = ~np.eye(4, dtype=bool)
    inputs = PreparedInputs(
        theta=np.array([0.0, 0.5, 1.0]),
        adjacency=adjacency,
        relationships=adjacency.astype(np.int8),
        initial_beliefs=np.array([[0.7, 0.2, 0.1], [0.2, 0.2, 0.6],
                                 [0.6, 0.3, 0.1], [0.5, 0.4, 0.1]]),
        initial_wealth=np.full(4, 10.0),
        observations=np.ones(1, dtype=int),
        recipients=np.array([[1, 0, 0, 0]]),
        wealth_distribution="uniform",
        giving_preferences=np.zeros(4, dtype=np.int8),
        conditional_allies=np.zeros(4, dtype=bool),
        contact_uniforms=np.zeros((1, 4)),
    )
    return config, inputs


@pytest.mark.parametrize(
    ("preference", "probability", "expected"),
    [(1, 0.8, [1, 1, 2, 2, 3, 3]), (-1, 0.8, [1, 2, 3, 3, 3, 3]),
     (1, 1.0, [1, 1, 1, 2, 2, 2]), (1, 0.0, [3, 3, 3, 3, 3, 3])],
)
def test_preference_probability_weights_groups_then_uniform_neighbors(
    trait_network, preference, probability, expected,
):
    config, inputs = trait_network
    signs = inputs.relationships.copy()
    signs[0, 3] = signs[3, 0] = -1
    draws = np.zeros((6, 4))
    draws[:, 0] = [0.05, 0.15, 0.45, 0.65, 0.85, 0.95]
    inputs = replace(
        inputs, relationships=signs, observations=np.ones(6, dtype=int),
        recipients=np.tile(inputs.recipients, (6, 1)), contact_uniforms=draws,
        giving_preferences=np.array([preference, 0, 0, 0]),
    )
    result = simulate(replace(config, rounds=6, preference_probability=probability), inputs, "fixed")
    assert_array_equal(result.recipients[:, 0], expected)
    assert_array_equal(result.recipients[:, 1:], inputs.recipients[:, 1:])


@pytest.mark.parametrize(
    ("preference", "probability", "recipient"),
    [(-1, 1.0, -1), (1, 0.0, -1), (-1, 0.8, 1), (1, 0.2, 1)],
)
def test_empty_groups_fall_back_only_when_the_available_group_has_weight(
    trait_network, preference, probability, recipient,
):
    config, inputs = trait_network
    inputs = replace(inputs, giving_preferences=np.array([preference, 0, 0, 0]))
    result = simulate(replace(config, preference_probability=probability), inputs, "fixed")
    assert result.recipients[0, 0] == recipient
    assert result.payments[0, 0] == (0.0 if recipient == -1 else config.fixed_payment)
    assert_allclose(result.wealth.sum(axis=1), inputs.initial_wealth.sum())


def test_unsigned_prepared_contacts_still_record_negative_skipped_contact(trait_network):
    config, inputs = trait_network
    inputs = replace(
        inputs, recipients=inputs.recipients.astype(np.uint8),
        giving_preferences=np.array([-1, 0, 0, 0]),
    )
    result = simulate(replace(config, preference_probability=1.0), inputs, "fixed")
    assert result.recipients.dtype.kind == "i"
    assert result.recipients[0, 0] == -1
    assert result.payments[0, 0] == 0.0
    assert_allclose(result.wealth.sum(axis=1), inputs.initial_wealth.sum())


@pytest.mark.parametrize("rule", ["attenuation", "tanh"])
def test_no_contact_has_no_social_influence_even_when_payment_argument_is_positive(rule):
    beliefs = np.array([[0.2, 0.8], [0.7, 0.3]])
    recipients = np.array([-1, -1])
    kwargs = dict(influence_rule=rule, opening_wealth=np.ones(2))
    weights = contact_weights(recipients, np.ones(2), np.array([[0, -1], [-1, 0]]), 1.0, **kwargs)
    assert_array_equal(weights, [0.0, 0.0])
    assert_array_equal(
        social_update(beliefs, recipients, np.ones(2), np.ones(2),
                      np.array([[0, -1], [-1, 0]]), 1.0, **kwargs),
        beliefs,
    )


@pytest.mark.parametrize("rule", ["attenuation", "tanh"])
def test_positive_payment_converts_only_its_direction_before_current_influence(trait_network, rule):
    config, inputs = trait_network
    signs = inputs.relationships.copy()
    signs[0, 1] = signs[1, 0] = -1
    signs[2, 1] = signs[1, 2] = -1
    inputs = replace(
        inputs, relationships=signs, initial_wealth=np.array([10.0, 10.0, 0.0, 0.0]),
        conditional_allies=np.array([False, True, False, False]),
    )
    config = replace(config, influence_rule=rule, opponent_threshold=2.0)
    result = simulate(config, inputs, "fixed")
    unconverted = simulate(config, replace(inputs, conditional_allies=np.zeros(4, dtype=bool)), "fixed")
    assert result.relationships[1, 0, 1] == 1
    assert result.relationships[1, 1, 0] == -1
    assert result.relationships[1, 2, 1] == -1
    assert result.beliefs[1, 1, 0] > inputs.initial_beliefs[1, 0]
    assert unconverted.beliefs[1, 1, 0] < inputs.initial_beliefs[1, 0]
    assert_array_equal(inputs.relationships, signs)
    assert_array_equal(result.relationships[0], signs)


@pytest.mark.parametrize("requested", [0.0, 100.0])
@pytest.mark.parametrize("rule", ["attenuation", "tanh"])
def test_fifth_zero_or_unaffordable_payment_defects_before_that_rounds_influence(
    trait_network, requested, rule,
):
    config, inputs = trait_network
    inputs = replace(
        inputs, observations=np.ones(5, dtype=int), recipients=np.tile([1, 0, 1, 1], (5, 1)),
        contact_uniforms=None, conditional_allies=np.array([False, True, False, False]),
    )
    config = replace(config, rounds=5, fixed_payment=requested, influence_rule=rule)
    result = simulate(config, inputs, "fixed")
    assert_array_equal(result.payments, np.zeros((5, 4)))
    assert_array_equal(result.relationships[:5, 0, 1], np.ones(5))
    assert result.relationships[5, 0, 1] == -1
    assert result.relationships[5, 1, 0] == 1
    assert_array_equal(result.beliefs[:5], np.broadcast_to(inputs.initial_beliefs, (5, 4, 3)))
    assert result.beliefs[5, 1, 0] < inputs.initial_beliefs[1, 0]


def test_positive_payment_on_expiry_round_resets_clock_and_payment_revives_defector(trait_network):
    config, inputs = trait_network
    recipients = np.tile(inputs.recipients, (12, 1))
    recipients[:, 0] = [1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1]
    inputs = replace(
        inputs, observations=np.ones(12, dtype=int), recipients=recipients, contact_uniforms=None,
        conditional_allies=np.array([False, True, False, False]),
    )
    result = simulate(replace(config, rounds=12), inputs, "fixed")
    assert_array_equal(result.relationships[:, 0, 1], [1] * 11 + [-1, 1])
    # No payment from agent 2 to recipient 1: that separate loyalty expires normally.
    assert_array_equal(result.relationships[:, 2, 1], [1] * 5 + [-1] * 8)
    assert_array_equal(result.relationships[:, 1, 0], np.ones(13))


def test_skipped_contacts_count_as_unpaid_and_new_opponents_can_be_targeted(trait_network):
    config, inputs = trait_network
    inputs = replace(
        inputs, observations=np.ones(6, dtype=int), recipients=np.tile(inputs.recipients, (6, 1)),
        contact_uniforms=np.zeros((6, 4)), giving_preferences=np.array([-1, 0, 0, 0]),
        conditional_allies=np.array([False, True, False, False]),
    )
    result = simulate(replace(config, rounds=6, preference_probability=1.0, fixed_payment=0.0), inputs, "fixed")
    assert_array_equal(result.recipients[:, 0], [-1, -1, -1, -1, -1, 1])
    assert_array_equal(result.relationships[:, 0, 1], [1, 1, 1, 1, 1, -1, -1])


def test_traits_use_separate_random_streams_and_match_between_wealth_arms():
    config = Config(n_agents=30, rounds=4, attachment=2, ally_giver_fraction=0.3,
                    opponent_giver_fraction=0.4, conditional_ally_fraction=0.5)
    uniform = make_inputs(config, "uniform")
    logarithmic = make_inputs(config, "loguniform")
    baseline = make_inputs(replace(config, ally_giver_fraction=0.0, opponent_giver_fraction=0.0,
                                  conditional_ally_fraction=0.0), "uniform")
    for name in ("adjacency", "relationships", "initial_beliefs", "observations", "recipients"):
        assert_array_equal(getattr(uniform, name), getattr(baseline, name))
        assert_array_equal(getattr(uniform, name), getattr(logarithmic, name))
    assert_array_equal(uniform.initial_wealth, baseline.initial_wealth)
    for name in ("giving_preferences", "conditional_allies", "contact_uniforms"):
        assert_array_equal(getattr(uniform, name), getattr(logarithmic, name))
    assert baseline.contact_uniforms is None
    assert set(uniform.giving_preferences) == {-1, 0, 1}
    assert np.any(uniform.conditional_allies & (uniform.giving_preferences != 0))
    no_conditional = make_inputs(replace(config, conditional_ally_fraction=0.0), "uniform")
    assert_array_equal(uniform.giving_preferences, no_conditional.giving_preferences)
    assert_array_equal(uniform.contact_uniforms, no_conditional.contact_uniforms)


@pytest.mark.parametrize("mode", ["fixed", "fractional", "hybrid"])
@pytest.mark.parametrize("rule", ["attenuation", "tanh"])
def test_combined_traits_preserve_cash_belief_normalization_and_prepared_arrays(mode, rule):
    config = Config(n_agents=12, rounds=20, attachment=2, public_evidence=False,
                    opponent_fraction=0.5, ally_giver_fraction=0.4, opponent_giver_fraction=0.6,
                    conditional_ally_fraction=0.6, preference_probability=1.0, influence_rule=rule)
    inputs = make_inputs(config, "loguniform")
    saved = {name: value.copy() for name, value in vars(inputs).items() if isinstance(value, np.ndarray)}
    result = simulate(config, inputs, mode)
    assert_allclose(result.wealth.sum(axis=1), inputs.initial_wealth.sum(), rtol=1e-14)
    assert np.all(result.wealth >= 0)
    assert_allclose(result.beliefs.sum(axis=2), 1.0)
    assert np.all(result.beliefs >= 0)
    for name, value in saved.items():
        assert_array_equal(getattr(inputs, name), value)


def test_legacy_prepared_inputs_remain_valid_and_static_history_uses_a_broadcast_view(trait_network):
    config, inputs = trait_network
    legacy = replace(inputs, giving_preferences=None, conditional_allies=None, contact_uniforms=None)
    result = simulate(config, legacy, "fixed")
    assert_array_equal(result.recipients, legacy.recipients)
    assert_array_equal(result.relationships, np.broadcast_to(legacy.relationships, (2, 4, 4)))
    assert result.relationships.strides[0] == 0


@pytest.mark.parametrize(
    ("overrides", "field"),
    [({"ally_giver_fraction": 0.5}, "giving_preferences"),
     ({"opponent_giver_fraction": 0.5}, "contact_uniforms"),
     ({"conditional_ally_fraction": 0.5}, "conditional_allies")],
)
def test_enabled_features_require_their_prepared_arrays(trait_network, overrides, field):
    config, inputs = trait_network
    with pytest.raises(ValueError, match="requires prepared"):
        simulate(replace(config, **overrides), replace(inputs, **{field: None}), "fixed")


@pytest.mark.parametrize("overrides", [
    {"ally_giver_fraction": -0.1}, {"opponent_giver_fraction": 1.1},
    {"ally_giver_fraction": 0.6, "opponent_giver_fraction": 0.6},
    {"conditional_ally_fraction": np.nan}, {"preference_probability": True},
    {"preference_probability": -0.1}, {"loyalty_rounds": 0}, {"loyalty_rounds": 2.5},
])
def test_config_rejects_invalid_trait_values(overrides):
    with pytest.raises(ValueError):
        Config(**overrides)


@pytest.mark.parametrize(("field", "value"), [
    ("giving_preferences", np.array([0, 1, 2, 0])),
    ("giving_preferences", np.zeros(4)),
    ("conditional_allies", np.ones(4, dtype=int)),
    ("conditional_allies", np.ones(3, dtype=bool)),
    ("contact_uniforms", np.ones((1, 4))),
    ("contact_uniforms", np.zeros((2, 4))),
    ("contact_uniforms", np.full((1, 4), np.nan)),
])
def test_prepared_inputs_validate_trait_array_contracts(trait_network, field, value):
    _, inputs = trait_network
    with pytest.raises(ValueError, match=field):
        replace(inputs, **{field: value})
