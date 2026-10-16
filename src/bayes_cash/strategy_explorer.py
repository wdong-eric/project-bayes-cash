"""Browser boundary for the complete-network belief preservation experiment."""

from dataclasses import asdict, fields, replace
import hashlib
from pathlib import Path

import numpy as np

from . import explorer
from .model import Config, Intervention
from .strategy import GaussianComponent, StrategySettings, initialize_beliefs, simulate_strategy


KIND = "belief-preservation"
FORMAT_VERSION = 1
DEFAULT_CONFIG = replace(explorer.DEFAULT_CONFIG, opponent_fraction=1.0, influence_rule="tanh")
SOURCE_SHA256 = hashlib.sha256(b"".join(
    Path(__file__).with_name(name).read_bytes()
    for name in ("model.py", "strategy.py", "explorer.py", "strategy_explorer.py")
)).hexdigest()


def model_identity():
    return {**explorer.model_identity(), "sha256": SOURCE_SHA256, "kind": KIND}


def parse_request(payload):
    """Validate model choices, saved identity, and export size before allocating."""
    allowed = {"kind", "config", "settings", "distribution", "interventions",
               "format_version", "model", "prepared_inputs", "initial_centers"}
    if not isinstance(payload, dict) or set(payload) - allowed or payload.get("kind", KIND) != KIND:
        raise ValueError("request must be a belief-preservation experiment with supported fields")
    saved_fields = {"format_version", "model", "prepared_inputs", "initial_centers"}
    if set(payload) & saved_fields:
        if set(payload) != allowed:
            raise ValueError("saved preservation experiment is missing settings, inputs, centers, or its log")
        if type(payload["format_version"]) is not int or payload["format_version"] != FORMAT_VERSION:
            raise ValueError("unsupported preservation experiment format_version")
        if payload["model"] != model_identity():
            raise ValueError("saved experiment model or dependency version differs; use the original implementation")

    overrides = payload.get("config", {})
    if not isinstance(overrides, dict) or set(overrides) - {field.name for field in fields(Config)}:
        raise ValueError("config contains unknown fields or is not an object")
    config = replace(DEFAULT_CONFIG, **overrides)
    if config.opponent_fraction != 1 or config.influence_rule != "tanh" or config.income_rate != 0:
        raise ValueError("preservation experiments require all initial opponents, tanh influence, and zero income")
    if not 4 <= config.n_agents <= 100 or not 1 <= config.rounds <= 1000 or not 5 <= config.grid_size <= 51:
        raise ValueError("use 4–100 agents, 1–1000 rounds, and 5–51 grid points")
    distribution = payload.get("distribution", "loguniform")
    if distribution not in ("uniform", "loguniform"):
        raise ValueError("wealth distribution must be uniform or loguniform")
    raw_settings = payload.get("settings", {})
    if not isinstance(raw_settings, dict) or set(raw_settings) - {field.name for field in fields(StrategySettings)}:
        raise ValueError("settings contains unknown fields or is not an object")
    settings_values = dict(raw_settings)
    if "components" in settings_values:
        components = settings_values["components"]
        component_fields = {field.name for field in fields(GaussianComponent)}
        if (not isinstance(components, list) or not 1 <= len(components) <= 3
            or any(not isinstance(c, dict) or set(c) != component_fields for c in components)):
            raise ValueError("components must contain one to three objects with mean, variance, and weight")
        settings_values["components"] = tuple(GaussianComponent(**c) for c in components)
    settings = StrategySettings(**settings_values)
    raw_events = payload.get("interventions", [])
    if not isinstance(raw_events, list) or len(raw_events) > 1000:
        raise ValueError("interventions must be a list of at most 1000 changes")
    interventions = []
    event_fields = {field.name for field in fields(Intervention)}
    for event in raw_events:
        if not isinstance(event, dict) or set(event) - event_fields or not {"agent", "round"} <= set(event):
            raise ValueError("each intervention requires agent, round, and supported manual settings")
        change = Intervention(**event)
        if (change.agent >= config.n_agents or change.round > config.rounds
            or change.neighbor is not None and change.neighbor >= config.n_agents):
            raise ValueError("intervention agent, neighbor, or round is outside this experiment")
        if change.connection == 0:
            raise ValueError("connections cannot be removed from the complete network")
        interventions.append(change)
    frame_count = len(explorer.display_frames(config, interventions))
    # Count both histories and every exported per-agent ledger, not only PMFs.
    values = frame_count * (2 if interventions else 1) * (
        config.n_agents * (config.grid_size + 5) + config.n_agents ** 2 + 2
    )
    if values > explorer.MAX_TRAJECTORY_VALUES:
        raise ValueError("preservation histories exceed the total trajectory limit; reduce agents, rounds, or grid size")
    return config, settings, distribution, interventions


def simulation_payload(payload):
    """Replay identical inputs with and without the ordered manual intervention log."""
    config, settings, distribution, interventions = parse_request(payload)
    frames = explorer.display_frames(config, interventions)
    if "prepared_inputs" in payload:
        inputs = explorer.restore_inputs(payload["prepared_inputs"], config, distribution, ["complete"])["complete"]
        centers = np.asarray(payload["initial_centers"])
        if (centers.shape != (config.n_agents,) or centers.dtype.kind not in "fiu"
            or not np.all(np.isfinite(centers)) or not np.all((centers >= 0) & (centers <= 1))
            or np.unique(centers).size != config.n_agents):
            raise ValueError("saved initial_centers must be distinct finite values within [0, 1]")
        if not np.array_equal(inputs.adjacency, ~np.eye(config.n_agents, dtype=bool)):
            raise ValueError("saved preservation inputs must describe a complete network")
        if not np.array_equal(inputs.relationships, -inputs.adjacency.astype(int)):
            raise ValueError("saved preservation inputs must start with every pair opposed")
        expected_centers, expected_beliefs = initialize_beliefs(config, settings)
        if (not np.array_equal(centers, expected_centers)
            or not np.array_equal(inputs.initial_beliefs, expected_beliefs)):
            raise ValueError("saved initial centers and beliefs do not match the recorded hierarchy settings")
    else:
        inputs = explorer.make_network_inputs(config, distribution, ["complete"])["complete"]
        centers, initial_beliefs = initialize_beliefs(config, settings)
        inputs = replace(inputs, initial_beliefs=initial_beliefs)
    settings_json = {**asdict(settings), "components": [asdict(c) for c in settings.components]}
    experiment = {
        "kind": KIND, "format_version": FORMAT_VERSION, "model": model_identity(),
        "config": asdict(config), "settings": settings_json, "distribution": distribution,
        "initial_centers": centers.tolist(),
        "interventions": [{name: value for name, value in asdict(event).items() if value is not None}
                          for event in interventions],
        "prepared_inputs": {"complete": {
            name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in vars(inputs).items()
        }},
    }
    angles = 2 * np.pi * np.arange(config.n_agents) / config.n_agents
    response = {
        "experiment": experiment, "config": asdict(config), "settings": settings_json,
        "distribution": distribution, "frames": frames.tolist(), "theta": inputs.theta.tolist(),
        "initial_centers": centers.tolist(), "observations": inputs.observations.astype(int).tolist(),
        "positions": np.column_stack((0.5 + 0.43 * np.cos(angles), 0.5 + 0.43 * np.sin(angles))).tolist(),
        "conditional_allies": inputs.conditional_allies.tolist(),
        "giving_preferences": inputs.giving_preferences.tolist(), "baseline": None,
        "metadata": {"sampled": len(frames) < config.rounds + 1,
                     "frame_count": len(frames), "simulated_rounds": config.rounds,
                     "spending_mechanisms": "inactive", "unpaid_defection": "inactive",
                     "conventions": {
                         "initial_beliefs": "Uniform centers are evenly spaced from 0 to 1. Gaussian mixtures use midpoint quantiles of the weighted mixture conditioned on [0,1], then a seeded shuffle assigns centers to agents. Component weights refer to the unbounded mixture; truncation changes effective shares. Each agent's Gaussian PMF has the configured common shape variance and is normalized on the theta grid. Input centers/variances need not equal the bounded discrete moments.",
                         "persistent_alliances": "A positive tanh contact creates a mutual alliance. A payment-dependent recipient instead accepts any positive payment as allied, including when its cash is zero. Qualifying conversions are committed together after pooling. Stored ally signs never expire, and no mixed directed relationships are stored. Only manual relationship assignments can break an alliance in this version; unpaid-round duration is inactive.",
                         "tanh_influence": "For payer j and recipient i, w = W_i tanh((payment - kappa W_i times opponent_indicator) / W_i), using opening cash. Equality at the threshold is neutral. At zero recipient cash the influence is zero. Conditional positive-payment acceptance is applied to that contact before computing its weight. Both directions use a frozen snapshot, so a new mutual conversion cannot affect the reverse contact in the same round.",
                         "social_pooling": "The existing signed social pooling and positive-part projection are retained. Self-weight = (N - 1) times reference_payment times (1 - social_rate) / social_rate. Negative influence can erase support; no floor restores it. Increasing population also increases self-weight.",
                         "contacts_and_payments": "Each agent chooses at most one other agent per update. Uniform contacts use saved recipients. Preference types use saved uniforms and opening relationships with the chosen group probability; if no group has positive probability, no contact is made. Fixed, fractional, hybrid max(fixed, fraction times cash), or zero payment is requested. An unaffordable request is skipped, not reduced. Recipients receive the full actual payment.",
                         "evidence": "Public evidence uses one shared Bernoulli observation each update, before social pooling. It is on by default; switch it off for a social-only diagnostic. Conditioning on an outcome with zero probability raises an error.",
                         "timing": "State 0 is initialization. Interventions execute before their one-based update, in log order for ties, then evidence, contact choice, payments, synchronous pooling, mutual sign commitment, and cash settlement. A manual opponent assignment may be reversed by a qualifying payment that same round. Sampled frames show actual states, including both sides of each intervention.",
                         "manual_interventions": "Behavior and mutual-sign assignments persist until superseded. Cash adjustments and Gaussian blends occur once, carrying their consequences forward. The manual Gaussian target uses a standard-deviation width, whereas initialization uses variance. Connection removal is rejected to preserve completeness. Manual Gaussian blending is free and is distinct from the inactive paid self-reinforcement proposal.",
                         "cash_accounting": "Cash is in arbitrary nominal units. There is no income or active money-burning mechanism. Total cash changes only through external manual adjustments; withdrawals cannot overdraw. Payments only transfer money. Saved cost proposals have no effect on current cash or beliefs.",
                         "proposed_spending": "Inactive: incitation would burn the configurable fixed cost from a distinct A to turn allied B-C into opponents. Reinforcement would burn s and blend toward the original PMF with alpha = 1 - exp(-s / reinforcement_scale). Neither proposal executes or has a log action in this version.",
                         "preservation_metric": "Total variation = 0.5 times the sum of absolute differences between the current and original PMF, between 0 and 1. It measures full-distribution preservation, not accuracy about the true bias. The displayed population metric is the mean of agents' individual distances.",
                         "paired_replay": "Baseline and current run share exact initial arrays, evidence, and contact randomness; the baseline omits the intervention log. Actual contacts can diverge after behavior/sign edits. Full experiment JSON fixes the setup and inputs. A separate intervention log replaces the current log and is interpreted against the current setup. Changing horizon or seed creates a different experiment; no prefix stability across newly generated runs is promised.",
                     }},
    }
    logs = {"scenario": interventions}
    if interventions:
        logs["baseline"] = ()
    for name, events in logs.items():
        try:
            result = simulate_strategy(config, inputs, settings, interventions=events)
        except ValueError as exc:
            raise ValueError(f"Complete / {settings.payment_mode} / {name}: {exc}") from exc
        distance = 0.5 * np.abs(result.beliefs - inputs.initial_beliefs[None, :, :]).sum(axis=2)
        zero = np.zeros(config.n_agents).tolist()
        response[name] = {
            "beliefs": result.beliefs[frames].tolist(), "wealth": result.wealth[frames].tolist(),
            "relationships": result.relationships[frames].tolist(),
            "payments": [zero] + result.payments[frames[1:] - 1].tolist(),
            "recipients": [[-1] * config.n_agents] + result.recipients[frames[1:] - 1].tolist(),
            "external_cash": [zero] + result.external_cash[frames[1:] - 1].tolist(),
            "distance": distance[frames].tolist(),
            "metrics": {"wealth_total": result.wealth.sum(axis=1)[frames].tolist(),
                        "mean_distance": distance.mean(axis=1)[frames].tolist()},
        }
    return response
