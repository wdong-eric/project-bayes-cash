"""Complete-network experiment with mutual, persistent payment-built alliances."""

from dataclasses import dataclass, replace
from numbers import Real
from statistics import NormalDist

import numpy as np

from .model import (
    Config, Intervention, PAYMENT_MODES, PreparedInputs, Result,
    bayesian_update, contact_weights, payment_amounts, social_update,
)


@dataclass(frozen=True)
class GaussianComponent:
    """A component of the population distribution before truncation to [0, 1]."""

    mean: float = 0.5
    variance: float = 0.04
    weight: float = 1.0

    def __post_init__(self):
        for name in ("mean", "variance", "weight"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"Gaussian component {name} must be a finite real number")
        if not 0 <= self.mean <= 1:
            raise ValueError("Gaussian component mean must lie in [0, 1]")
        if self.variance <= 0 or self.weight < 0:
            raise ValueError("Gaussian component variance must be positive and weight nonnegative")


@dataclass(frozen=True)
class StrategySettings:
    """Population priors and saved, inactive proposals for future spending rules."""

    payment_mode: str = "fractional"
    belief_distribution: str = "uniform"
    within_agent_variance: float = 0.01
    components: tuple[GaussianComponent, ...] = (GaussianComponent(),)
    incitation_cost: float = 1.0
    reinforcement_scale: float = 10.0

    def __post_init__(self):
        if self.payment_mode not in (*PAYMENT_MODES, "no_payment"):
            raise ValueError("strategy payment_mode must be fixed, fractional, hybrid, or no_payment")
        if self.belief_distribution not in ("uniform", "gaussian_mixture"):
            raise ValueError("belief_distribution must be uniform or gaussian_mixture")
        for name in ("within_agent_variance", "incitation_cost", "reinforcement_scale"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite real number")
        if not isinstance(self.components, tuple) or not 1 <= len(self.components) <= 3:
            raise ValueError("components must be a tuple of one to three Gaussian components")
        if not all(isinstance(component, GaussianComponent) for component in self.components):
            raise ValueError("components must contain validated GaussianComponent objects")
        if not any(component.weight > 0 for component in self.components):
            raise ValueError("at least one Gaussian component must have positive weight")


def _gaussian_masses(theta: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    """Normalize Gaussian shapes on the existing hypothesis grid without floors."""
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        log_mass = -0.5 * ((theta[None, :] - centers[:, None]) / width) ** 2
    if not np.all(np.isfinite(log_mass)):
        raise ValueError("Gaussian width exceeds finite Gaussian arithmetic")
    mass = np.exp(log_mass - log_mass.max(axis=1, keepdims=True))
    return mass / mass.sum(axis=1, keepdims=True)


def initialize_beliefs(config: Config, settings: StrategySettings) -> tuple[np.ndarray, np.ndarray]:
    """Use distinct population quantiles, then shuffle their assignment to agents.

    Uniform centers include both endpoints. Gaussian centers use midpoint
    quantiles of the entire weighted mixture conditioned on [0, 1]. Component
    means/variances are pre-truncation parameters; discrete agent beliefs have
    their own common shape variance and need not realize these exact moments.
    """
    if settings.belief_distribution == "uniform":
        centers = np.linspace(0.0, 1.0, config.n_agents)
    else:
        components = [component for component in settings.components if component.weight > 0]
        normals = [NormalDist(component.mean, np.sqrt(component.variance)) for component in components]
        lower = np.array([normal.cdf(0.0) for normal in normals])
        retained = np.array([normal.cdf(1.0) for normal in normals]) - lower
        if np.any(retained <= 0):
            raise ValueError("Gaussian component has numerically unresolvable mass inside [0, 1]")
        weights = np.array([component.weight for component in components], dtype=float)
        weights /= weights.max()
        weights /= weights.sum()
        total_mass = np.dot(weights, retained)
        quantiles = (np.arange(config.n_agents) + 0.5) / config.n_agents
        if len(normals) == 1:
            probabilities = lower[0] + quantiles * retained[0]
            if np.any((probabilities <= 0) | (probabilities >= 1)):
                raise ValueError("Gaussian quantiles are numerically unresolvable")
            centers = np.array([normals[0].inv_cdf(float(value)) for value in probabilities])
        else:
            centers = np.empty(config.n_agents)
            for index, quantile in enumerate(quantiles):
                left, right = 0.0, 1.0
                for _ in range(64):
                    midpoint = (left + right) / 2.0
                    cumulative = np.dot(
                        weights, np.array([normal.cdf(midpoint) for normal in normals]) - lower,
                    ) / total_mass
                    if cumulative < quantile:
                        left = midpoint
                    else:
                        right = midpoint
                centers[index] = (left + right) / 2.0
        if not np.all(np.isfinite(centers)) or np.any((centers < 0) | (centers > 1)):
            raise ValueError("Gaussian centers could not be resolved inside [0, 1]")
        if np.any(np.diff(centers) <= 0):
            raise ValueError("Gaussian centers are numerically indistinguishable; increase variance")
    # A separate stream keeps ID assignment independent of all run-time draws.
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 193785]))
    centers = centers[rng.permutation(config.n_agents)]
    theta = np.linspace(0.0, 1.0, config.grid_size)
    beliefs = _gaussian_masses(theta, centers, np.sqrt(settings.within_agent_variance))
    if np.unique(beliefs, axis=0).shape[0] != config.n_agents:
        raise ValueError("initial beliefs are numerically indistinguishable; adjust within_agent_variance or grid_size")
    return centers, beliefs


def simulate_strategy(
    config: Config, inputs: PreparedInputs, settings: StrategySettings,
    *, interventions: tuple[Intervention, ...] = (),
) -> Result:
    """Run complete-graph payments with synchronous influence and sticky alliances.

    Positive payment to a conditional recipient creates an ally, as in the
    observatory. Otherwise, positive opponent tanh influence creates an ally.
    Either conversion commits both directions after simultaneous pooling; all
    contacts use the same opening signs, except their own conditional-payment
    conversion. Alliances persist until a manual connection intervention.
    Payment-dependent expiry and both proposed spending mechanisms are inactive.
    """
    if (
        inputs.initial_wealth.size != config.n_agents
        or inputs.theta.size != config.grid_size
        or inputs.observations.size != config.rounds
    ):
        raise ValueError("config dimensions must match prepared inputs")
    adjacency = ~np.eye(config.n_agents, dtype=bool)
    if not np.array_equal(inputs.adjacency, adjacency):
        raise ValueError("strategy requires a complete network")
    if not np.array_equal(inputs.relationships, -adjacency.astype(np.int8)):
        raise ValueError("strategy requires all initial connections to be mutual opponents")
    if config.conditional_ally_fraction > 0 and inputs.conditional_allies is None:
        raise ValueError("payment-dependent loyalty requires prepared conditional_allies")
    if config.ally_giver_fraction + config.opponent_giver_fraction > 0 and inputs.giving_preferences is None:
        raise ValueError("preferential giving requires prepared giving_preferences")
    for event in interventions:
        if not isinstance(event, Intervention):
            raise ValueError("interventions must contain validated Intervention objects")
        if event.agent >= config.n_agents or event.round > config.rounds:
            raise ValueError("intervention agent and round must lie within the configured run")
        if event.neighbor is not None and event.neighbor >= config.n_agents:
            raise ValueError("intervention neighbor must lie within the configured run")
        if event.connection == 0:
            raise ValueError("strategy connections cannot be removed from the complete network")
    events = sorted(interventions, key=lambda event: event.round)
    preferences = (
        inputs.giving_preferences.copy() if inputs.giving_preferences is not None
        else np.zeros(config.n_agents, dtype=np.int8)
    )
    conditional = (
        inputs.conditional_allies.copy() if inputs.conditional_allies is not None
        else np.zeros(config.n_agents, dtype=bool)
    )
    if inputs.contact_uniforms is None and (
        np.any(preferences) or any(event.giving_preference in (-1, 1) for event in events)
    ):
        raise ValueError("preferential giving requires prepared contact_uniforms")
    beliefs = np.empty((config.rounds + 1, config.n_agents, config.grid_size))
    wealth = np.empty((config.rounds + 1, config.n_agents))
    payments = np.empty((config.rounds, config.n_agents))
    external_cash = np.zeros_like(payments)
    recipients_history = np.empty(inputs.recipients.shape, dtype=int)
    relationships = inputs.relationships.copy()
    history = np.empty((config.rounds + 1, config.n_agents, config.n_agents), dtype=np.int8)
    beliefs[0], wealth[0], history[0] = inputs.initial_beliefs, inputs.initial_wealth, relationships
    self_weight = np.full(
        config.n_agents,
        (config.n_agents - 1) * config.reference_payment * (1.0 - config.social_rate) / config.social_rate,
    )
    payment_config = replace(config, fixed_payment=0.0) if settings.payment_mode == "no_payment" else config
    payment_mode = "fixed" if settings.payment_mode == "no_payment" else settings.payment_mode
    event_index = 0
    for step in range(config.rounds):
        opening_beliefs = beliefs[step].copy()
        opening_wealth = wealth[step].copy()
        while event_index < len(events) and events[event_index].round == step + 1:
            event = events[event_index]
            if event.giving_preference is not None:
                preferences[event.agent] = event.giving_preference
            if event.conditional_ally is not None:
                conditional[event.agent] = event.conditional_ally
            if event.cash_delta is not None:
                with np.errstate(over="ignore", invalid="ignore"):
                    opening_wealth[event.agent] += event.cash_delta
                    external_cash[step, event.agent] += event.cash_delta
                    total_cash = opening_wealth.sum()
                if opening_wealth[event.agent] < 0:
                    raise ValueError(f"intervention before round {event.round} withdraws more cash than the agent holds")
                if not np.isfinite(total_cash) or not np.isfinite(external_cash[step, event.agent]):
                    raise ValueError(f"intervention before round {event.round} exceeds finite cash arithmetic")
            if event.belief_center is not None:
                target = _gaussian_masses(inputs.theta, np.array([event.belief_center]), event.belief_width)[0]
                opening_beliefs[event.agent] = (
                    (1.0 - event.belief_strength) * opening_beliefs[event.agent]
                    + event.belief_strength * target
                )
            if event.connection is not None:
                relationships[event.agent, event.neighbor] = relationships[event.neighbor, event.agent] = event.connection
            event_index += 1
        post_evidence = (
            bayesian_update(opening_beliefs, inputs.theta, inputs.observations[step])
            if config.public_evidence else opening_beliefs
        )
        recipients = recipients_history[step]
        recipients[:] = inputs.recipients[step]
        for payer in np.flatnonzero(preferences):
            neighbors = np.flatnonzero(adjacency[payer])
            preferred = relationships[payer, neighbors] == preferences[payer]
            weights = np.zeros(neighbors.size)
            if np.any(preferred):
                weights[preferred] = config.preference_probability / preferred.sum()
            if np.any(~preferred):
                weights[~preferred] = (1.0 - config.preference_probability) / (~preferred).sum()
            cumulative = np.cumsum(weights)
            if cumulative[-1] > 0:
                cumulative /= cumulative[-1]
                recipients[payer] = neighbors[np.searchsorted(cumulative, inputs.contact_uniforms[step, payer], side="right")]
            else:
                recipients[payer] = -1
        payments[step] = payment_amounts(payment_config, opening_wealth, payment_mode)
        active = recipients >= 0
        payments[step, ~active] = 0.0
        paid = np.flatnonzero(active & (payments[step] > 0))
        conditional_paid = paid[conditional[recipients[paid]]]
        influence_relationships = relationships.copy()
        influence_relationships[conditional_paid, recipients[conditional_paid]] = 1
        weights = contact_weights(
            recipients, payments[step], influence_relationships, config.reference_payment,
            influence_rule="tanh", opening_wealth=opening_wealth,
            opponent_threshold=config.opponent_threshold,
        )
        beliefs[step + 1] = social_update(
            post_evidence, recipients, payments[step], self_weight,
            influence_relationships, config.reference_payment,
            influence_rule="tanh", opening_wealth=opening_wealth,
            opponent_threshold=config.opponent_threshold,
        )
        # Commit together so a reverse same-round contact keeps its own weight.
        converted = np.union1d(np.flatnonzero(weights > 0), conditional_paid)
        targets = recipients[converted]
        relationships[converted, targets] = relationships[targets, converted] = 1
        history[step + 1] = relationships
        incoming = np.zeros(config.n_agents)
        np.add.at(incoming, recipients[active], payments[step, active])
        with np.errstate(over="ignore", invalid="ignore"):
            wealth[step + 1] = (1.0 + config.income_rate) * opening_wealth - payments[step] + incoming
            total_cash = wealth[step + 1].sum()
        if not np.isfinite(total_cash) or not np.all(np.isfinite(beliefs[step + 1])):
            raise ValueError("simulation exceeded finite numerical arithmetic; reduce rounds or income_rate")
    return Result(beliefs, wealth, payments, settings.payment_mode, recipients_history, history, external_cash)
