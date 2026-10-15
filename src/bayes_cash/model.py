"""Discrete Bayesian beliefs with cash-funded, directed social influence."""

from dataclasses import dataclass
from numbers import Real

import networkx as nx
import numpy as np


PAYMENT_MODES = ("fixed", "fractional", "hybrid")
INFLUENCE_RULES = ("attenuation", "tanh")


@dataclass(frozen=True)
class Config:
    n_agents: int = 100
    rounds: int = 500
    attachment: int = 3
    grid_size: int = 21
    true_bias: float = 0.6
    social_rate: float = 0.25
    reference_payment: float = 1.0
    opponent_fraction: float = 0.0
    influence_rule: str = "attenuation"
    opponent_threshold: float = 0.05
    income_rate: float = 0.0
    wealth_min: float = 1.0
    wealth_max: float = 100.0
    fixed_payment: float = 1.0
    wealth_fraction: float = 0.05
    seed: int = 42
    public_evidence: bool = True
    ally_giver_fraction: float = 0.0
    opponent_giver_fraction: float = 0.0
    preference_probability: float = 0.8
    conditional_ally_fraction: float = 0.0
    loyalty_rounds: int = 5

    def __post_init__(self):
        for name in ("n_agents", "rounds", "attachment", "grid_size", "seed", "loyalty_rounds"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.n_agents < 2 or self.rounds < 1 or self.grid_size < 2:
            raise ValueError("n_agents and grid_size must be at least 2; rounds must be positive")
        if not 1 <= self.attachment < self.n_agents:
            raise ValueError("attachment must satisfy 1 <= attachment < n_agents")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.loyalty_rounds < 1:
            raise ValueError("loyalty_rounds must be positive")
        if not isinstance(self.public_evidence, (bool, np.bool_)):
            raise ValueError("public_evidence must be boolean")
        for name in (
            "true_bias", "social_rate", "reference_payment", "wealth_min",
            "wealth_max", "fixed_payment", "wealth_fraction", "opponent_fraction",
            "opponent_threshold", "income_rate",
            "ally_giver_fraction", "opponent_giver_fraction", "preference_probability",
            "conditional_ally_fraction",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        if not 0 <= self.true_bias <= 1:
            raise ValueError("true_bias must lie in [0, 1]")
        if not 0 < self.social_rate < 1:
            raise ValueError("social_rate must lie strictly between 0 and 1")
        if self.reference_payment <= 0:
            raise ValueError("reference_payment must be positive")
        if not 0 <= self.opponent_fraction <= 1:
            raise ValueError("opponent_fraction must lie in [0, 1]")
        for name in (
            "ally_giver_fraction", "opponent_giver_fraction", "preference_probability",
            "conditional_ally_fraction",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.ally_giver_fraction + self.opponent_giver_fraction > 1:
            raise ValueError("ally_giver_fraction and opponent_giver_fraction must sum to at most 1")
        if self.influence_rule not in INFLUENCE_RULES:
            raise ValueError(f"influence_rule must be one of {INFLUENCE_RULES}")
        if self.opponent_threshold < 0 or self.income_rate < 0:
            raise ValueError("opponent_threshold and income_rate must be nonnegative")
        if not 0 < self.wealth_min < self.wealth_max:
            raise ValueError("wealth bounds must satisfy 0 < wealth_min < wealth_max")
        if self.fixed_payment < 0 or not 0 <= self.wealth_fraction <= 1:
            raise ValueError("fixed_payment must be nonnegative; wealth_fraction must lie in [0, 1]")


@dataclass(frozen=True)
class Intervention:
    """Change an agent before a one-based update round.

    Trait settings persist until changed again. Enabling payment-dependent loyalty
    starts fresh unpaid-round counters while retaining the current relationship
    signs; disabling it freezes those signs. Reapplying an enabled setting leaves
    its counters intact. Cash and belief interventions change the state once:
    cash is an external injection or withdrawal, and beliefs blend toward a
    Gaussian probability mass on the existing grid. Connections
    set both directed signs, or remove both, and reset the pair's loyalty clocks.
    """

    agent: int
    round: int
    giving_preference: int | None = None
    conditional_ally: bool | None = None
    cash_delta: float | None = None
    belief_center: float | None = None
    belief_width: float | None = None
    belief_strength: float | None = None
    neighbor: int | None = None
    connection: int | None = None

    def __post_init__(self):
        for name in ("agent", "round"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"intervention {name} must be an integer")
        if self.agent < 0 or self.round < 1:
            raise ValueError("intervention agent must be nonnegative and round must be positive")
        if self.giving_preference is not None and (
            isinstance(self.giving_preference, (bool, np.bool_))
            or not isinstance(self.giving_preference, (int, np.integer))
            or self.giving_preference not in (-1, 0, 1)
        ):
            raise ValueError("intervention giving_preference must be an integer in {-1, 0, 1}")
        if self.conditional_ally is not None and not isinstance(self.conditional_ally, (bool, np.bool_)):
            raise ValueError("intervention conditional_ally must be boolean")
        for name in ("cash_delta", "belief_center", "belief_width", "belief_strength"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value)
            ):
                raise ValueError(f"intervention {name} must be a finite real number")
        belief_fields = (self.belief_center, self.belief_width, self.belief_strength)
        if any(value is not None for value in belief_fields):
            if any(value is None for value in belief_fields):
                raise ValueError("intervention belief_center, belief_width, and belief_strength are required together")
            if not 0 <= self.belief_center <= 1 or not 0 <= self.belief_strength <= 1:
                raise ValueError("intervention belief_center and belief_strength must lie in [0, 1]")
            if self.belief_width <= 0:
                raise ValueError("intervention belief_width must be positive")
        if self.neighbor is not None or self.connection is not None:
            for name in ("neighbor", "connection"):
                value = getattr(self, name)
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                    raise ValueError(f"intervention {name} must be an integer; neighbor and connection are required together")
            if self.neighbor < 0 or self.neighbor == self.agent:
                raise ValueError("intervention neighbor must be nonnegative and different from agent")
            if self.connection not in (-1, 0, 1):
                raise ValueError("intervention connection must lie in {-1, 0, 1}")
        if all(getattr(self, name) is None for name in (
            "giving_preference", "conditional_ally", "cash_delta", "belief_center", "connection",
        )):
            raise ValueError("intervention must change at least one setting")


@dataclass(frozen=True)
class PreparedInputs:
    """Validated simulation inputs; array contents must be treated as immutable."""

    theta: np.ndarray
    adjacency: np.ndarray
    relationships: np.ndarray
    initial_beliefs: np.ndarray
    initial_wealth: np.ndarray
    observations: np.ndarray
    recipients: np.ndarray
    wealth_distribution: str
    giving_preferences: np.ndarray | None = None
    conditional_allies: np.ndarray | None = None
    contact_uniforms: np.ndarray | None = None

    def __post_init__(self):
        for name in (
            "theta", "adjacency", "relationships", "initial_beliefs", "initial_wealth",
            "observations", "recipients",
        ):
            if not isinstance(getattr(self, name), np.ndarray):
                raise ValueError(f"{name} must be a NumPy array")
        if self.wealth_distribution not in ("uniform", "loguniform"):
            raise ValueError("wealth_distribution must be 'uniform' or 'loguniform'")
        for name in ("theta", "initial_beliefs", "initial_wealth"):
            value = getattr(self, name)
            if value.dtype.kind not in "fiu" or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite real numbers")
        if self.theta.ndim != 1 or self.theta.size < 2:
            raise ValueError("theta must be a one-dimensional grid with at least two points")
        if not np.all((0 <= self.theta) & (self.theta <= 1)) or not np.all(self.theta[1:] > self.theta[:-1]):
            raise ValueError("theta must be strictly increasing within [0, 1]")
        if self.adjacency.ndim != 2 or self.adjacency.shape[0] != self.adjacency.shape[1]:
            raise ValueError("adjacency must be square")
        n_agents = self.adjacency.shape[0]
        if (
            n_agents < 2 or self.adjacency.dtype.kind != "b"
            or not np.array_equal(self.adjacency, self.adjacency.T)
            or np.any(np.diag(self.adjacency)) or not np.all(self.adjacency.any(axis=1))
        ):
            raise ValueError("adjacency must be boolean, symmetric, loop-free, and have no isolated agents")
        if (
            self.relationships.shape != self.adjacency.shape
            or self.relationships.dtype.kind != "i"
            or not np.all(np.isin(self.relationships, [-1, 0, 1]))
            or not np.array_equal(self.relationships, self.relationships.T)
            or not np.array_equal(self.relationships != 0, self.adjacency)
        ):
            raise ValueError("relationships must be symmetric signed integers in {-1, 0, 1}, nonzero exactly on edges")
        if self.initial_beliefs.shape != (n_agents, self.theta.size):
            raise ValueError("initial_beliefs must have shape (n_agents, grid_size)")
        if np.any(self.initial_beliefs < 0) or not np.allclose(
            self.initial_beliefs.sum(axis=1), 1.0, rtol=1e-12, atol=1e-12,
        ):
            raise ValueError("each initial belief must be a nonnegative probability mass summing to one")
        if self.initial_wealth.shape != (n_agents,) or np.any(self.initial_wealth < 0):
            raise ValueError("initial_wealth must be a nonnegative vector of length n_agents")
        total_wealth = self.initial_wealth.sum()
        if not np.isfinite(total_wealth) or total_wealth <= 0:
            raise ValueError("initial_wealth must have a positive finite total")
        if (
            self.observations.ndim != 1 or self.observations.size < 1
            or self.observations.dtype.kind not in "biu"
            or not np.all((self.observations == 0) | (self.observations == 1))
        ):
            raise ValueError("observations must be a nonempty binary vector")
        if self.recipients.shape != (self.observations.size, n_agents) or self.recipients.dtype.kind not in "iu":
            raise ValueError("recipients must be an integer array of shape (rounds, n_agents)")
        if np.any(self.recipients < 0) or np.any(self.recipients >= n_agents):
            raise ValueError("recipient indices must lie within the network")
        if not np.all(self.adjacency[np.arange(n_agents), self.recipients]):
            raise ValueError("each recipient must be a neighbor of its payer")
        if self.giving_preferences is not None and (
            not isinstance(self.giving_preferences, np.ndarray)
            or self.giving_preferences.shape != (n_agents,)
            or self.giving_preferences.dtype.kind != "i"
            or not np.all(np.isin(self.giving_preferences, [-1, 0, 1]))
        ):
            raise ValueError("giving_preferences must be a signed integer vector in {-1, 0, 1}")
        if self.conditional_allies is not None and (
            not isinstance(self.conditional_allies, np.ndarray)
            or self.conditional_allies.shape != (n_agents,)
            or self.conditional_allies.dtype.kind != "b"
        ):
            raise ValueError("conditional_allies must be a boolean vector of length n_agents")
        if self.contact_uniforms is not None and (
            not isinstance(self.contact_uniforms, np.ndarray)
            or self.contact_uniforms.shape != self.recipients.shape
            or self.contact_uniforms.dtype.kind not in "fiu"
            or not np.all(np.isfinite(self.contact_uniforms))
            or not np.all((0 <= self.contact_uniforms) & (self.contact_uniforms < 1))
        ):
            raise ValueError("contact_uniforms must have shape (rounds, n_agents) and values in [0, 1)")


@dataclass(frozen=True)
class Result:
    beliefs: np.ndarray
    wealth: np.ndarray
    payments: np.ndarray
    payment_mode: str
    recipients: np.ndarray | None = None
    relationships: np.ndarray | None = None
    external_cash: np.ndarray | None = None


def make_inputs(config: Config, wealth_distribution: str) -> PreparedInputs:
    """Presample all randomness, sharing its streams between experiment arms."""
    if wealth_distribution not in ("uniform", "loguniform"):
        raise ValueError("wealth_distribution must be 'uniform' or 'loguniform'")
    (
        graph_rng, prior_rng, wealth_rng, evidence_rng, contact_rng, relationship_rng,
        giver_rng, conditional_rng, preference_rng,
    ) = (
        np.random.default_rng(stream)
        for stream in np.random.SeedSequence(config.seed).spawn(9)
    )
    graph = nx.barabasi_albert_graph(
        config.n_agents, config.attachment,
        seed=int(graph_rng.integers(0, np.iinfo(np.int32).max)),
    )
    adjacency = nx.to_numpy_array(graph, nodelist=range(config.n_agents), dtype=bool)
    relationships = adjacency.astype(np.int8)
    edge_i, edge_j = np.nonzero(np.triu(adjacency, k=1))
    opponents = relationship_rng.random(edge_i.size) < config.opponent_fraction
    relationships[edge_i[opponents], edge_j[opponents]] = -1
    relationships[edge_j[opponents], edge_i[opponents]] = -1
    theta = np.linspace(0.0, 1.0, config.grid_size)
    means = prior_rng.uniform(0.0, 1.0, size=(config.n_agents, 1))
    widths = prior_rng.uniform(0.2, 0.8, size=(config.n_agents, 1))
    initial_beliefs = np.exp(-0.5 * ((theta - means) / widths) ** 2)
    initial_beliefs /= initial_beliefs.sum(axis=1, keepdims=True)
    wealth_quantiles = wealth_rng.random(config.n_agents)
    if wealth_distribution == "uniform":
        initial_wealth = config.wealth_min + wealth_quantiles * (config.wealth_max - config.wealth_min)
    else:
        initial_wealth = np.exp(
            np.log(config.wealth_min)
            + wealth_quantiles * (np.log(config.wealth_max) - np.log(config.wealth_min))
        )
    observations = evidence_rng.random(config.rounds) < config.true_bias
    recipients = np.empty((config.rounds, config.n_agents), dtype=int)
    for payer in range(config.n_agents):
        recipients[:, payer] = contact_rng.choice(
            np.flatnonzero(adjacency[payer]), size=config.rounds,
        )
    giver_quantiles = giver_rng.random(config.n_agents)
    giving_preferences = np.zeros(config.n_agents, dtype=np.int8)
    giving_preferences[giver_quantiles < config.ally_giver_fraction] = 1
    giving_preferences[
        (giver_quantiles >= config.ally_giver_fraction)
        & (giver_quantiles < config.ally_giver_fraction + config.opponent_giver_fraction)
    ] = -1
    conditional_allies = conditional_rng.random(config.n_agents) < config.conditional_ally_fraction
    # Reserve this independent stream even when preferences are only enabled later.
    contact_uniforms = preference_rng.random((config.rounds, config.n_agents))
    return PreparedInputs(
        theta, adjacency, relationships, initial_beliefs, initial_wealth,
        observations, recipients, wealth_distribution,
        giving_preferences, conditional_allies, contact_uniforms,
    )


def bayesian_update(beliefs: np.ndarray, theta: np.ndarray, observation: bool | int) -> np.ndarray:
    """Condition every agent's grid masses on one shared Bernoulli outcome."""
    likelihood = theta if observation else 1.0 - theta
    posterior = beliefs * likelihood
    evidence = posterior.sum(axis=1, keepdims=True)
    if np.any(evidence == 0):
        raise ValueError("observed outcome has zero probability under an agent's belief")
    return posterior / evidence


def contact_weights(
    recipients: np.ndarray,
    payments: np.ndarray,
    relationships: np.ndarray,
    reference_payment: float,
    *,
    influence_rule: str = "attenuation",
    opening_wealth: np.ndarray | None = None,
    opponent_threshold: float = 0.05,
) -> np.ndarray:
    """Map each directed payment to its signed influence on the recipient.

    Tanh uses the recipient's opening wealth as both amplitude and money scale.
    The current relationship supplies the opponent threshold; payment-dependent
    loyalty is updated before this function. A recipient of -1 means no contact.
    At zero recipient wealth the analytic prefactor limit is zero.
    """
    active = recipients >= 0
    opponents = np.zeros(recipients.size, dtype=bool)
    opponents[active] = relationships[np.flatnonzero(active), recipients[active]] < 0
    if influence_rule == "attenuation":
        weights = payments.astype(float, copy=True)
        weights[opponents] = -reference_payment / (1.0 + payments[opponents] / reference_payment)
        weights[~active] = 0.0
        return weights
    if influence_rule != "tanh":
        raise ValueError(f"influence_rule must be one of {INFLUENCE_RULES}")
    if opening_wealth is None:
        raise ValueError("tanh influence requires opening_wealth")
    recipient_wealth = np.zeros(recipients.size)
    recipient_wealth[active] = opening_wealth[recipients[active]]
    solvent = recipient_wealth > 0
    weights = np.zeros(recipients.size)
    # Subtract cash amounts first to keep an exactly threshold-sized payment neutral.
    neutral_payment = opponent_threshold * recipient_wealth[solvent] * opponents[solvent]
    weights[solvent] = recipient_wealth[solvent] * np.tanh(
        (payments[solvent] - neutral_payment) / recipient_wealth[solvent]
    )
    return weights


def social_update(
    beliefs: np.ndarray,
    recipients: np.ndarray,
    payments: np.ndarray,
    self_weight: np.ndarray,
    relationships: np.ndarray,
    reference_payment: float,
    *,
    influence_rule: str = "attenuation",
    opening_wealth: np.ndarray | None = None,
    opponent_threshold: float = 0.05,
) -> np.ndarray:
    """Apply signed pooling, projecting only recipients of negative influence.

    This source-model positive-part projection can erase support; no probability
    floor restores an excluded hypothesis. Effective tanh signs are recalculated
    for every contact, so a baseline opponent can instead attract this round.
    """
    weights = contact_weights(
        recipients, payments, relationships, reference_payment,
        influence_rule=influence_rule, opening_wealth=opening_wealth,
        opponent_threshold=opponent_threshold,
    )
    negative = weights < 0
    active = recipients >= 0
    incoming = np.zeros(beliefs.shape[0])
    np.add.at(incoming, recipients[active], np.abs(weights[active]))
    repulsion = np.zeros(beliefs.shape[0])
    np.add.at(repulsion, recipients[negative], -weights[negative])
    paid_beliefs = np.zeros_like(beliefs, dtype=float)
    np.add.at(paid_beliefs, recipients[active], weights[active, None] * beliefs[active])
    updated = beliefs.astype(float, copy=True)
    influenced = incoming > 0
    updated[influenced] = (
        (self_weight[influenced] + 2.0 * repulsion[influenced])[:, None]
        * beliefs[influenced] + paid_beliefs[influenced]
    ) / (self_weight[influenced] + incoming[influenced])[:, None]
    repelled = repulsion > 0
    updated[repelled] = np.maximum(updated[repelled], 0.0)
    updated[repelled] /= updated[repelled].sum(axis=1, keepdims=True)
    return updated


def payment_amounts(config: Config, opening_wealth: np.ndarray, payment_mode: str) -> np.ndarray:
    """Skip requests beyond opening balances or, when enabled, current income."""
    if payment_mode not in PAYMENT_MODES:
        raise ValueError(f"payment_mode must be one of {PAYMENT_MODES}")
    if payment_mode == "fixed":
        requested = config.fixed_payment
    elif payment_mode == "fractional":
        requested = config.wealth_fraction * opening_wealth
    else:
        requested = np.maximum(config.fixed_payment, config.wealth_fraction * opening_wealth)
    budget = opening_wealth
    if config.income_rate > 0:
        budget = np.minimum(budget, config.income_rate * opening_wealth)
    return np.where(budget >= requested, requested, 0.0)


def simulate(
    config: Config, inputs: PreparedInputs, payment_mode: str,
    *, interventions: tuple[Intervention, ...] = (),
) -> Result:
    """Select contacts, update payer-specific loyalty, pool beliefs, and settle cash.

    ``relationships[j, i]`` is recipient i's receptivity to payer j. Contacts use
    opening-round signs; any positive actual payment makes a conditional recipient
    an ally immediately. Its fifth unpaid round (by default) repels before pooling.
    Interventions apply before evidence, contacts, income, and payments in their
    one-based update round; equal-round events retain their supplied order.
    Replaying from prepared inputs
    preserves the full state, including loyalty counters, before each intervention.
    """
    if (
        inputs.initial_wealth.size != config.n_agents
        or inputs.theta.size != config.grid_size
        or inputs.observations.size != config.rounds
    ):
        raise ValueError("config dimensions must match prepared inputs")
    if config.ally_giver_fraction + config.opponent_giver_fraction > 0 and (
        inputs.giving_preferences is None or inputs.contact_uniforms is None
    ):
        raise ValueError("preferential giving requires prepared giving_preferences and contact_uniforms")
    if config.conditional_ally_fraction > 0 and inputs.conditional_allies is None:
        raise ValueError("payment-dependent loyalty requires prepared conditional_allies")
    for event in interventions:
        if not isinstance(event, Intervention):
            raise ValueError("interventions must contain validated Intervention objects")
        if event.agent >= config.n_agents or event.round > config.rounds:
            raise ValueError("intervention agent and round must lie within the configured run")
        if event.neighbor is not None and event.neighbor >= config.n_agents:
            raise ValueError("intervention neighbor must lie within the configured run")
    events = sorted(interventions, key=lambda event: event.round)
    giving_preferences = (
        inputs.giving_preferences.copy() if inputs.giving_preferences is not None
        else np.zeros(config.n_agents, dtype=np.int8)
    )
    conditional_allies = (
        inputs.conditional_allies.copy() if inputs.conditional_allies is not None
        else np.zeros(config.n_agents, dtype=bool)
    )
    preferential_payers = np.flatnonzero(giving_preferences)
    if inputs.contact_uniforms is None and (
        preferential_payers.size or any(event.giving_preference in (-1, 1) for event in events)
    ):
        raise ValueError("preferential giving requires prepared contact_uniforms")
    topology_changes = any(event.connection is not None for event in events)
    if topology_changes and inputs.contact_uniforms is None:
        raise ValueError("changing connections requires prepared contact_uniforms")
    adjacency = inputs.adjacency.copy()
    conditional_edges = adjacency & conditional_allies[None, :]
    dynamic_relationships = (
        np.any(conditional_edges) or any(event.conditional_ally for event in events) or topology_changes
    )
    if config.income_rate > 0 and not any(event.cash_delta is not None for event in events) and (
        np.log(inputs.initial_wealth.sum())
        + config.rounds * np.log1p(config.income_rate)
        >= np.log(np.finfo(float).max)
    ):
        raise ValueError("income growth exceeds finite nominal money; reduce rounds or income_rate")
    beliefs = np.empty((config.rounds + 1, config.n_agents, config.grid_size))
    wealth = np.empty((config.rounds + 1, config.n_agents))
    payments = np.empty((config.rounds, config.n_agents))
    external_cash = np.zeros((config.rounds, config.n_agents))
    actual_recipients = np.empty(inputs.recipients.shape, dtype=int)
    relationships = inputs.relationships.copy()
    if dynamic_relationships:
        relationship_history = np.empty(
            (config.rounds + 1, config.n_agents, config.n_agents), dtype=relationships.dtype,
        )
        relationship_history[0] = relationships
        unpaid_rounds = np.zeros_like(relationships, dtype=int)
    else:
        relationship_history = np.broadcast_to(
            relationships, (config.rounds + 1, config.n_agents, config.n_agents),
        )
    beliefs[0] = inputs.initial_beliefs
    wealth[0] = inputs.initial_wealth
    # Equal reference payments from every neighbor recover the source ally update.
    self_weight = (
        adjacency.sum(axis=1) * config.reference_payment
        * (1.0 - config.social_rate) / config.social_rate
    )
    event_index = 0
    for step in range(config.rounds):
        opening_beliefs = beliefs[step].copy()
        opening_wealth = wealth[step].copy()
        while event_index < len(events) and events[event_index].round == step + 1:
            event = events[event_index]
            if event.giving_preference is not None:
                giving_preferences[event.agent] = event.giving_preference
                preferential_payers = np.flatnonzero(giving_preferences)
            if event.conditional_ally is not None:
                if event.conditional_ally and not conditional_allies[event.agent]:
                    unpaid_rounds[:, event.agent] = 0
                conditional_allies[event.agent] = event.conditional_ally
                conditional_edges[:, event.agent] = adjacency[:, event.agent] & event.conditional_ally
            if event.cash_delta is not None:
                with np.errstate(over="ignore", invalid="ignore"):
                    opening_wealth[event.agent] += event.cash_delta
                    external_cash[step, event.agent] += event.cash_delta
                    total_cash = opening_wealth.sum()
                if opening_wealth[event.agent] < 0:
                    raise ValueError(
                        f"intervention before round {event.round} for agent {event.agent} "
                        "withdraws more cash than the agent holds"
                    )
                if not np.isfinite(total_cash) or not np.isfinite(external_cash[step, event.agent]):
                    raise ValueError(
                        f"intervention before round {event.round} for agent {event.agent} "
                        "exceeds finite cash arithmetic"
                    )
            if event.belief_center is not None:
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    log_target = -0.5 * ((inputs.theta - event.belief_center) / event.belief_width) ** 2
                if not np.all(np.isfinite(log_target)):
                    raise ValueError(
                        f"intervention before round {event.round} for agent {event.agent} "
                        "belief width exceeds finite Gaussian arithmetic"
                    )
                target = np.exp(log_target - log_target.max())
                target /= target.sum()
                opening_beliefs[event.agent] = (
                    (1.0 - event.belief_strength) * opening_beliefs[event.agent]
                    + event.belief_strength * target
                )
            if event.connection is not None:
                i, j = event.agent, event.neighbor
                adjacency[i, j] = adjacency[j, i] = event.connection != 0
                relationships[i, j] = relationships[j, i] = event.connection
                unpaid_rounds[i, j] = unpaid_rounds[j, i] = 0
                conditional_edges = adjacency & conditional_allies[None, :]
                self_weight = (
                    adjacency.sum(axis=1) * config.reference_payment
                    * (1.0 - config.social_rate) / config.social_rate
                )
            event_index += 1
        post_evidence = (
            bayesian_update(opening_beliefs, inputs.theta, inputs.observations[step])
            if config.public_evidence else opening_beliefs
        )
        payments[step] = payment_amounts(config, opening_wealth, payment_mode)
        recipients = actual_recipients[step]
        recipients[:] = inputs.recipients[step]
        if topology_changes:
            changed_uniform_payers = np.flatnonzero(
                np.any(adjacency != inputs.adjacency, axis=1) & (giving_preferences == 0)
            )
            for payer in changed_uniform_payers:
                neighbors = np.flatnonzero(adjacency[payer])
                recipients[payer] = (
                    neighbors[int(inputs.contact_uniforms[step, payer] * neighbors.size)]
                    if neighbors.size else -1
                )
        for payer in preferential_payers:
            neighbors = np.flatnonzero(adjacency[payer])
            if neighbors.size == 0:
                recipients[payer] = -1
                continue
            preferred = relationships[payer, neighbors] == giving_preferences[payer]
            weights = np.zeros(neighbors.size)
            if np.any(preferred):
                weights[preferred] = config.preference_probability / preferred.sum()
            if np.any(~preferred):
                weights[~preferred] = (1.0 - config.preference_probability) / (~preferred).sum()
            cumulative = np.cumsum(weights)
            if cumulative[-1] > 0:
                cumulative /= cumulative[-1]
                recipients[payer] = neighbors[np.searchsorted(
                    cumulative, inputs.contact_uniforms[step, payer], side="right",
                )]
            else:
                recipients[payer] = -1
        active = recipients >= 0
        payments[step, ~active] = 0.0
        if dynamic_relationships:
            unpaid_rounds[conditional_edges] += 1
            paid = np.flatnonzero(active & (payments[step] > 0))
            conditional_paid = paid[conditional_allies[recipients[paid]]]
            relationships[conditional_paid, recipients[conditional_paid]] = 1
            unpaid_rounds[conditional_paid, recipients[conditional_paid]] = 0
            relationships[conditional_edges & (unpaid_rounds >= config.loyalty_rounds)] = -1
            relationship_history[step + 1] = relationships
        beliefs[step + 1] = social_update(
            post_evidence, recipients, payments[step], self_weight,
            relationships, config.reference_payment,
            influence_rule=config.influence_rule, opening_wealth=opening_wealth,
            opponent_threshold=config.opponent_threshold,
        )
        incoming = np.zeros(config.n_agents)
        np.add.at(incoming, recipients[active], payments[step, active])
        with np.errstate(over="ignore", invalid="ignore"):
            if config.income_rate > 0:
                wealth[step + 1] = (1.0 + config.income_rate) * opening_wealth - payments[step] + incoming
            else:
                wealth[step + 1] = opening_wealth - payments[step] + incoming
            total_cash = wealth[step + 1].sum()
        if not np.isfinite(total_cash) or not np.all(np.isfinite(beliefs[step + 1])):
            raise ValueError("simulation exceeded finite numerical arithmetic; reduce rounds or income_rate")
    return Result(beliefs, wealth, payments, payment_mode, actual_recipients, relationship_history, external_cash)
