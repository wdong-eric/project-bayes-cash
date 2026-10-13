"""Discrete Bayesian beliefs with cash-funded, directed social influence."""

from dataclasses import dataclass
from numbers import Real

import networkx as nx
import numpy as np


PAYMENT_MODES = ("fixed", "fractional", "hybrid")


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
    wealth_min: float = 1.0
    wealth_max: float = 100.0
    fixed_payment: float = 1.0
    wealth_fraction: float = 0.05
    seed: int = 42
    public_evidence: bool = True

    def __post_init__(self):
        for name in ("n_agents", "rounds", "attachment", "grid_size", "seed"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.n_agents < 2 or self.rounds < 1 or self.grid_size < 2:
            raise ValueError("n_agents and grid_size must be at least 2; rounds must be positive")
        if not 1 <= self.attachment < self.n_agents:
            raise ValueError("attachment must satisfy 1 <= attachment < n_agents")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not isinstance(self.public_evidence, (bool, np.bool_)):
            raise ValueError("public_evidence must be boolean")
        for name in (
            "true_bias", "social_rate", "reference_payment", "wealth_min",
            "wealth_max", "fixed_payment", "wealth_fraction", "opponent_fraction",
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
        if not 0 < self.wealth_min < self.wealth_max:
            raise ValueError("wealth bounds must satisfy 0 < wealth_min < wealth_max")
        if self.fixed_payment < 0 or not 0 <= self.wealth_fraction <= 1:
            raise ValueError("fixed_payment must be nonnegative; wealth_fraction must lie in [0, 1]")


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


@dataclass(frozen=True)
class Result:
    beliefs: np.ndarray
    wealth: np.ndarray
    payments: np.ndarray
    payment_mode: str


def make_inputs(config: Config, wealth_distribution: str) -> PreparedInputs:
    """Presample all randomness, sharing its streams between experiment arms."""
    if wealth_distribution not in ("uniform", "loguniform"):
        raise ValueError("wealth_distribution must be 'uniform' or 'loguniform'")
    graph_rng, prior_rng, wealth_rng, evidence_rng, contact_rng, relationship_rng = (
        np.random.default_rng(stream)
        for stream in np.random.SeedSequence(config.seed).spawn(6)
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
    return PreparedInputs(
        theta, adjacency, relationships, initial_beliefs, initial_wealth,
        observations, recipients, wealth_distribution,
    )


def bayesian_update(beliefs: np.ndarray, theta: np.ndarray, observation: bool | int) -> np.ndarray:
    """Condition every agent's grid masses on one shared Bernoulli outcome."""
    likelihood = theta if observation else 1.0 - theta
    posterior = beliefs * likelihood
    evidence = posterior.sum(axis=1, keepdims=True)
    if np.any(evidence == 0):
        raise ValueError("observed outcome has zero probability under an agent's belief")
    return posterior / evidence


def social_update(
    beliefs: np.ndarray,
    recipients: np.ndarray,
    payments: np.ndarray,
    self_weight: np.ndarray,
    relationships: np.ndarray,
    reference_payment: float,
) -> np.ndarray:
    """Attract toward paid allies; payments weaken scheduled opponents' repulsion.

    Opponents carry weight ``-reference_payment / (1 + payment / reference_payment)``.
    The signed difference update is truncated at zero and renormalized whenever
    an opponent contributes. This source-model projection can erase support;
    no probability floor is added to restore an excluded hypothesis.
    """
    opponents = relationships[np.arange(beliefs.shape[0]), recipients] < 0
    weights = payments.astype(float, copy=True)
    weights[opponents] = -reference_payment / (1.0 + payments[opponents] / reference_payment)
    incoming = np.zeros(beliefs.shape[0])
    np.add.at(incoming, recipients, np.abs(weights))
    repulsion = np.zeros(beliefs.shape[0])
    np.add.at(repulsion, recipients[opponents], -weights[opponents])
    paid_beliefs = np.zeros_like(beliefs, dtype=float)
    np.add.at(paid_beliefs, recipients, weights[:, None] * beliefs)
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
    """Compute affordable payments from opening balances; skip unaffordable requests."""
    if payment_mode not in PAYMENT_MODES:
        raise ValueError(f"payment_mode must be one of {PAYMENT_MODES}")
    if payment_mode == "fixed":
        requested = config.fixed_payment
    elif payment_mode == "fractional":
        requested = config.wealth_fraction * opening_wealth
    else:
        requested = np.maximum(config.fixed_payment, config.wealth_fraction * opening_wealth)
    return np.where(opening_wealth >= requested, requested, 0.0)


def simulate(config: Config, inputs: PreparedInputs, payment_mode: str) -> Result:
    """Run synchronous evidence, payment-weighted pooling, and cash settlement."""
    if (
        inputs.initial_wealth.size != config.n_agents
        or inputs.theta.size != config.grid_size
        or inputs.observations.size != config.rounds
    ):
        raise ValueError("config dimensions must match prepared inputs")
    beliefs = np.empty((config.rounds + 1, config.n_agents, config.grid_size))
    wealth = np.empty((config.rounds + 1, config.n_agents))
    payments = np.empty((config.rounds, config.n_agents))
    beliefs[0] = inputs.initial_beliefs
    wealth[0] = inputs.initial_wealth
    # Equal reference payments from every neighbor recover the source ally update.
    self_weight = (
        inputs.adjacency.sum(axis=1) * config.reference_payment
        * (1.0 - config.social_rate) / config.social_rate
    )
    for step in range(config.rounds):
        post_evidence = (
            bayesian_update(beliefs[step], inputs.theta, inputs.observations[step])
            if config.public_evidence else beliefs[step]
        )
        opening_wealth = wealth[step]
        payments[step] = payment_amounts(config, opening_wealth, payment_mode)
        recipients = inputs.recipients[step]
        beliefs[step + 1] = social_update(
            post_evidence, recipients, payments[step], self_weight,
            inputs.relationships, config.reference_payment,
        )
        incoming = np.zeros(config.n_agents)
        np.add.at(incoming, recipients, payments[step])
        wealth[step + 1] = opening_wealth - payments[step] + incoming
    return Result(beliefs, wealth, payments, payment_mode)
