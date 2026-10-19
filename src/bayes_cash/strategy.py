"""Complete-network strategies with persistent and recipient-specific allegiance."""

from dataclasses import dataclass, replace
from numbers import Real
from statistics import NormalDist

import numpy as np

from .model import (
    Config, PAYMENT_MODES, PreparedInputs, bayesian_update, payment_amounts,
)


PAYMENT_STRATEGIES = ("random_one", "all", "farthest", "farthest_random")
PERSON_TYPES = ("ordinary", "payment_dependent", "junzi")


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
    payment_strategy: str = "random_one"
    junzi_fraction: float = 0.0
    belief_distribution: str = "uniform"
    within_agent_variance: float = 0.01
    components: tuple[GaussianComponent, ...] = (GaussianComponent(),)
    incitation_cost: float = 1.0
    reinforcement_scale: float = 10.0

    def __post_init__(self):
        if self.payment_mode not in (*PAYMENT_MODES, "no_payment"):
            raise ValueError("strategy payment_mode must be fixed, fractional, hybrid, or no_payment")
        if self.payment_strategy not in PAYMENT_STRATEGIES:
            raise ValueError(f"payment_strategy must be one of {PAYMENT_STRATEGIES}")
        if (isinstance(self.junzi_fraction, (bool, np.bool_))
            or not isinstance(self.junzi_fraction, Real) or not np.isfinite(self.junzi_fraction)
            or not 0 <= self.junzi_fraction <= 1):
            raise ValueError("junzi_fraction must lie in [0, 1]")
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


def wasserstein_distance(
    beliefs: np.ndarray, reference_beliefs: np.ndarray, theta: np.ndarray,
) -> np.ndarray:
    """Return Wasserstein-2 between PMFs on a shared increasing theta grid.

    Leading dimensions broadcast; the final axis holds normalized probability
    masses. Integrate squared quantile separation over probability, then take
    its square root. Matching mass in sorted order is optimal in one dimension.
    """
    beliefs, reference_beliefs = np.broadcast_arrays(beliefs, reference_beliefs)
    left_cdf = np.cumsum(beliefs, axis=-1)
    right_cdf = np.cumsum(reference_beliefs, axis=-1)
    # Anchor both probability domains at exactly one despite summation roundoff.
    left_cdf /= left_cdf[..., -1:]
    right_cdf /= right_cdf[..., -1:]
    boundaries = np.concatenate((left_cdf[..., :-1], right_cdf[..., :-1]), axis=-1)
    order = np.argsort(boundaries, axis=-1, kind="stable")
    boundaries = np.take_along_axis(boundaries, order, axis=-1)
    mass = np.diff(boundaries, prepend=0.0, append=1.0, axis=-1)
    zero_index = np.zeros((*beliefs.shape[:-1], 1), dtype=int)
    left_index = np.concatenate((zero_index, np.cumsum(order < theta.size - 1, axis=-1)), axis=-1)
    right_index = np.arange(mass.shape[-1]) - left_index
    return np.sqrt((mass * (theta[left_index] - theta[right_index]) ** 2).sum(axis=-1))


@dataclass(frozen=True)
class StrategyIntervention:
    """Apply a lab-specific agent change before a one-based update round."""

    agent: int
    round: int
    person_type: str | None = None
    payment_strategy: str | None = None
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
        if self.person_type is not None and self.person_type not in PERSON_TYPES:
            raise ValueError(f"intervention person_type must be one of {PERSON_TYPES}")
        if self.payment_strategy is not None and self.payment_strategy not in PAYMENT_STRATEGIES:
            raise ValueError(f"intervention payment_strategy must be one of {PAYMENT_STRATEGIES}")
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
            if self.connection not in (-1, 1):
                raise ValueError("strategy connections cannot be removed; connection must be -1 or 1")
        if all(getattr(self, name) is None for name in (
            "person_type", "payment_strategy", "cash_delta", "belief_center", "connection",
        )):
            raise ValueError("intervention must change at least one setting")


@dataclass(frozen=True)
class StrategyResult:
    beliefs: np.ndarray
    wealth: np.ndarray
    payments: np.ndarray
    contacts: np.ndarray
    transfers: np.ndarray
    relationships: np.ndarray
    persistent_relationships: np.ndarray
    external_cash: np.ndarray


def initialize_person_types(config: Config, settings: StrategySettings) -> tuple[str, ...]:
    """Sample one exclusive recipient type from a separate reproducible stream."""
    if config.conditional_ally_fraction + settings.junzi_fraction > 1:
        raise ValueError("payment-dependent and junzi fractions must sum to at most 1")
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 456173]))
    draws = rng.random(config.n_agents)
    types = np.full(config.n_agents, "ordinary", dtype=object)
    types[draws < config.conditional_ally_fraction] = "payment_dependent"
    types[(draws >= config.conditional_ally_fraction)
          & (draws < config.conditional_ally_fraction + settings.junzi_fraction)] = "junzi"
    return tuple(types)


def simulate_strategy(
    config: Config, inputs: PreparedInputs, settings: StrategySettings,
    *, interventions: tuple[StrategyIntervention, ...] = (),
    person_types: tuple[str, ...] | None = None,
) -> StrategyResult:
    """Split each budget over synchronous contacts and layer recipient responses.

    Ordinary recipients use the persistent mutual base. Payment-dependent
    recipients instead use their own temporary, directed paid-loyalty signs.
    Junzi retain their response signs and ignore incoming social influence, but
    still learn from evidence, exchange cash, and influence other recipients.
    Targeting uses post-evidence beliefs and opening cash. Persistent conversions
    commit after pooling, independently of the temporary loyalty override.
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
    if inputs.contact_uniforms is None:
        raise ValueError("strategy requires prepared contact_uniforms")
    if config.conditional_ally_fraction + settings.junzi_fraction > 1:
        raise ValueError("payment-dependent and junzi fractions must sum to at most 1")
    if person_types is None:
        person_types = initialize_person_types(config, settings)
    if (not isinstance(person_types, tuple) or len(person_types) != config.n_agents
        or any(value not in PERSON_TYPES for value in person_types)):
        raise ValueError("person_types must be a tuple containing one supported type per agent")
    for event in interventions:
        if not isinstance(event, StrategyIntervention):
            raise ValueError("interventions must contain validated StrategyIntervention objects")
        if event.agent >= config.n_agents or event.round > config.rounds:
            raise ValueError("intervention agent and round must lie within the configured run")
        if event.neighbor is not None and event.neighbor >= config.n_agents:
            raise ValueError("intervention neighbor must lie within the configured run")
    events = sorted(interventions, key=lambda event: event.round)
    types = np.array(person_types, dtype=object)
    strategies = np.full(config.n_agents, settings.payment_strategy, dtype=object)
    beliefs = np.empty((config.rounds + 1, config.n_agents, config.grid_size))
    wealth = np.empty((config.rounds + 1, config.n_agents))
    payments = np.empty((config.rounds, config.n_agents))
    external_cash = np.zeros_like(payments)
    contacts = np.zeros((config.rounds, config.n_agents, config.n_agents), dtype=bool)
    transfers = np.zeros(contacts.shape)
    history = np.empty((config.rounds + 1, config.n_agents, config.n_agents), dtype=np.int8)
    persistent_history = np.empty_like(history)
    persistent = inputs.relationships.copy()
    loyal = persistent.copy()
    frozen = persistent.copy()
    effective = persistent.copy()
    unpaid_rounds = np.zeros_like(persistent, dtype=int)
    beliefs[0], wealth[0] = inputs.initial_beliefs, inputs.initial_wealth
    history[0], persistent_history[0] = effective, persistent
    self_weight = np.full(
        config.n_agents,
        (config.n_agents - 1) * config.reference_payment * (1.0 - config.social_rate) / config.social_rate,
    )
    payment_config = replace(config, fixed_payment=0.0) if settings.payment_mode == "no_payment" else config
    payment_mode = "fixed" if settings.payment_mode == "no_payment" else settings.payment_mode
    neighbors = [np.flatnonzero(adjacency[payer]) for payer in range(config.n_agents)]
    event_index = 0
    for step in range(config.rounds):
        opening_beliefs = beliefs[step].copy()
        opening_wealth = wealth[step].copy()
        while event_index < len(events) and events[event_index].round == step + 1:
            event = events[event_index]
            if event.person_type is not None and event.person_type != types[event.agent]:
                if event.person_type == "payment_dependent":
                    loyal[:, event.agent] = effective[:, event.agent]
                    unpaid_rounds[:, event.agent] = 0
                elif event.person_type == "junzi":
                    frozen[:, event.agent] = effective[:, event.agent]
                types[event.agent] = event.person_type
            if event.payment_strategy is not None:
                strategies[event.agent] = event.payment_strategy
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
                i, j = event.agent, event.neighbor
                persistent[i, j] = persistent[j, i] = event.connection
                loyal[i, j] = loyal[j, i] = event.connection
                frozen[i, j] = frozen[j, i] = event.connection
                unpaid_rounds[i, j] = unpaid_rounds[j, i] = 0
            # Equal-round interventions see all earlier assignments in log order.
            effective[:] = persistent
            effective[:, types == "payment_dependent"] = loyal[:, types == "payment_dependent"]
            effective[:, types == "junzi"] = frozen[:, types == "junzi"]
            event_index += 1
        post_evidence = (
            bayesian_update(opening_beliefs, inputs.theta, inputs.observations[step])
            if config.public_evidence else opening_beliefs
        )
        farthest = None
        if np.any(np.isin(strategies, ("farthest", "farthest_random"))):
            distances = wasserstein_distance(post_evidence[:, None, :], post_evidence[None, :, :], inputs.theta)
            np.fill_diagonal(distances, -np.inf)
            # np.argmax chooses the first (lowest-ID) recipient on an exact tie.
            farthest = distances.argmax(axis=1)
        for payer, strategy in enumerate(strategies):
            if strategy == "all":
                contacts[step, payer] = adjacency[payer]
            elif strategy == "random_one":
                choice = int(inputs.contact_uniforms[step, payer] * neighbors[payer].size)
                contacts[step, payer, neighbors[payer][choice]] = True
            else:
                contacts[step, payer, farthest[payer]] = True
                if strategy == "farthest_random":
                    candidates = neighbors[payer][neighbors[payer] != farthest[payer]]
                    # With two agents, the farthest recipient is the only option.
                    if candidates.size:
                        choice = int(inputs.contact_uniforms[step, payer] * candidates.size)
                        contacts[step, payer, candidates[choice]] = True
        payments[step] = payment_amounts(payment_config, opening_wealth, payment_mode)
        transfers[step] = contacts[step] * (payments[step] / contacts[step].sum(axis=1))[:, None]
        conditional = types == "payment_dependent"
        junzi = types == "junzi"
        conditional_edges = adjacency & conditional[None, :]
        unpaid_rounds[conditional_edges] += 1
        paid = (transfers[step] > 0) & conditional_edges
        unpaid_rounds[paid] = 0
        loyal[paid] = 1
        loyal[conditional_edges & (unpaid_rounds >= config.loyalty_rounds)] = -1
        effective[:] = persistent
        effective[:, conditional] = loyal[:, conditional]
        effective[:, junzi] = frozen[:, junzi]
        weights = np.zeros_like(transfers[step])
        solvent = opening_wealth > 0
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            threshold_payment = config.opponent_threshold * opening_wealth
            weights[:, solvent] = opening_wealth[solvent] * np.tanh(
                (transfers[step][:, solvent] - threshold_payment[solvent] * (effective[:, solvent] < 0))
                / opening_wealth[solvent]
            )
        weights[~contacts[step]] = 0.0
        weights[:, junzi] = 0.0
        incoming_weight = np.abs(weights).sum(axis=0)
        repulsion = np.maximum(-weights, 0.0).sum(axis=0)
        updated = post_evidence.copy()
        influenced = incoming_weight > 0
        updated[influenced] = (
            (self_weight[influenced] + 2.0 * repulsion[influenced])[:, None] * post_evidence[influenced]
            + (weights.T @ post_evidence)[influenced]
        ) / (self_weight[influenced] + incoming_weight[influenced])[:, None]
        repelled = repulsion > 0
        updated[repelled] = np.maximum(updated[repelled], 0.0)
        updated[repelled] /= updated[repelled].sum(axis=1, keepdims=True)
        beliefs[step + 1] = updated
        converted = (contacts[step] & (persistent < 0) & solvent[None, :]
                     & (transfers[step] > threshold_payment[None, :])
                     & ~junzi[:, None] & ~junzi[None, :])
        persistent[converted | converted.T] = 1
        effective[:] = persistent
        effective[:, conditional] = loyal[:, conditional]
        effective[:, junzi] = frozen[:, junzi]
        history[step + 1], persistent_history[step + 1] = effective, persistent
        with np.errstate(over="ignore", invalid="ignore"):
            wealth[step + 1] = (1.0 + config.income_rate) * opening_wealth - payments[step] + transfers[step].sum(axis=0)
            total_cash = wealth[step + 1].sum()
        if not np.isfinite(total_cash) or not np.all(np.isfinite(beliefs[step + 1])):
            raise ValueError("simulation exceeded finite numerical arithmetic; reduce rounds or income_rate")
    return StrategyResult(beliefs, wealth, payments, contacts, transfers, history, persistent_history, external_cash)
