"""Browser boundary for the complete-network belief preservation experiment."""

from dataclasses import asdict, fields, replace
import hashlib
from pathlib import Path

import numpy as np

from . import explorer
from .model import Config
from .strategy import (
    GaussianComponent, StrategyIntervention, StrategySettings, initialize_beliefs,
    initialize_person_types, simulate_strategy, wasserstein_distance,
)


KIND = "belief-preservation"
FORMAT_VERSION = 2
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
               "format_version", "model", "prepared_inputs", "initial_centers", "person_types"}
    if not isinstance(payload, dict) or set(payload) - allowed or payload.get("kind", KIND) != KIND:
        raise ValueError("request must be a belief-preservation experiment with supported fields")
    saved_fields = {"format_version", "model", "prepared_inputs", "initial_centers", "person_types"}
    if set(payload) & saved_fields:
        if type(payload.get("format_version")) is not int or payload["format_version"] != FORMAT_VERSION:
            raise ValueError("unsupported preservation experiment format_version; use the original implementation for version 1")
        if set(payload) != allowed:
            raise ValueError("saved preservation experiment is missing settings, inputs, centers, person types, or its log")
        if payload["model"] != model_identity():
            raise ValueError("saved experiment model or dependency version differs; use the original implementation")

    overrides = payload.get("config", {})
    if not isinstance(overrides, dict) or set(overrides) - {field.name for field in fields(Config)}:
        raise ValueError("config contains unknown fields or is not an object")
    config = replace(DEFAULT_CONFIG, **overrides)
    if config.opponent_fraction != 1 or config.influence_rule != "tanh" or config.income_rate != 0:
        raise ValueError("preservation experiments require all initial opponents, tanh influence, and zero income")
    if config.ally_giver_fraction != 0 or config.opponent_giver_fraction != 0:
        raise ValueError("lab payment strategies replace ally/opponent giving preferences")
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
    if config.conditional_ally_fraction + settings.junzi_fraction > 1:
        raise ValueError("payment-dependent and junzi fractions must sum to at most 1")
    raw_events = payload.get("interventions", [])
    if not isinstance(raw_events, list) or len(raw_events) > 1000:
        raise ValueError("interventions must be a list of at most 1000 changes")
    interventions = []
    event_fields = {field.name for field in fields(StrategyIntervention)}
    for event in raw_events:
        if not isinstance(event, dict) or set(event) - event_fields or not {"agent", "round"} <= set(event):
            raise ValueError("each intervention requires agent, round, and supported manual settings")
        change = StrategyIntervention(**event)
        if (change.agent >= config.n_agents or change.round > config.rounds
            or change.neighbor is not None and change.neighbor >= config.n_agents):
            raise ValueError("intervention agent, neighbor, or round is outside this experiment")
        if change.connection == 0:
            raise ValueError("connections cannot be removed from the complete network")
        interventions.append(change)
    frame_count = len(explorer.display_frames(config, interventions))
    # Count both histories and every exported per-agent ledger, not only PMFs.
    values = frame_count * (2 if interventions else 1) * (
        config.n_agents * (config.grid_size + 4) + 4 * config.n_agents ** 2 + 2
    )
    if values > explorer.MAX_TRAJECTORY_VALUES:
        raise ValueError("preservation histories exceed the total trajectory limit; reduce agents, rounds, or grid size")
    return config, settings, distribution, interventions


def simulation_payload(payload):
    """Replay identical inputs with and without the ordered manual intervention log."""
    config, settings, distribution, interventions = parse_request(payload)
    frames = explorer.display_frames(config, interventions)
    person_types = initialize_person_types(config, settings)
    if "prepared_inputs" in payload:
        if payload["person_types"] != list(person_types):
            raise ValueError("saved person_types do not match the recorded population settings")
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
        "initial_centers": centers.tolist(), "person_types": list(person_types),
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
        "person_types": list(person_types), "baseline": None,
        "metadata": {"sampled": len(frames) < config.rounds + 1,
                     "frame_count": len(frames), "simulated_rounds": config.rounds,
                     "spending_mechanisms": "inactive", "unpaid_defection": "active",
                     "conventions": {
                         "initial_beliefs": "Uniform centers are evenly spaced from 0 to 1. Gaussian mixtures use midpoint quantiles of the weighted mixture conditioned on [0,1], then a seeded shuffle assigns centers to agents. Component weights refer to the unbounded mixture; truncation changes effective shares. Each agent's Gaussian PMF has the configured common shape variance and is normalized on the theta grid. Input centers/variances need not equal the bounded discrete moments.",
                         "persistent_alliances": "Persistent mutual signs and temporary loyalty are separate. A positive raw tanh contact on a base opponent pair commits both base signs to ally after pooling. Raw means before temporary paid loyalty is applied. Equality is neutral. Base alliances do not expire. A pair involving a junzi cannot flip through payment. Manual assignments can set both base signs; incitation is inactive.",
                         "person_types": "Types are disjoint seeded categorical draws: ordinary, payment-dependent, or junzi. Fractions are sampling probabilities, not exact population counts. An ordinary recipient uses its persistent base sign. A payment-dependent recipient becomes allied toward a payer on any positive actual payment, resets that pair's unpaid clock, and becomes opposed on the configured consecutive unpaid round, before pooling. Unpaid rounds count even without contact. This directed loyalty overrides, but never erases, the base alliance. A junzi receives zero social influence and no payment-driven sign changes; evidence and manual edits still apply. Junzi may transfer and receive cash and influence other people.",
                         "directed_relationships": "Entry [payer][recipient] is the recipient's response to that payer. Effective signs may be asymmetric even though the underlying persistent signs remain mutual. The page displays both layers. State 0 starts opposed in both layers. Junzi response signs are retained for inspection and manual changes but carry zero incoming social influence.",
                         "tanh_influence": "For contacted payer j and recipient i, w = W_i tanh((P_ji - kappa W_i times effective_opponent_indicator) / W_i), using opening cash and the per-recipient transfer. Noncontacts have zero weight. At zero recipient cash the influence is zero; junzi recipients always have zero weight. Conditional paid loyalty and expiry apply before calculating influence. Base conversion is evaluated independently from opening base signs; conversions commit after synchronous pooling, so a new base alliance cannot alter a reverse contact in the same round.",
                         "social_pooling": "Self-weight a = (N - 1) times reference_payment times (1 - social_rate) / social_rate is inertia toward the current post-evidence belief, not money spent and not an anchor to the initial belief. Pooling is q_i + sum_j w_ji(q_j - q_i)/(a + sum_j abs(w_ji)). For positive weights the denominator normalizes a weighted average. Negative influence uses the same bounded scale, followed by positive-part normalization. It can erase support; no floor restores it. The social-rate calibration assumes N - 1 influences each of reference-payment size, not the actual sparse realized contacts.",
                         "contacts_and_payments": "Each payer selects random one, all other agents, farthest, or farthest plus a uniformly random distinct other agent. Farthest uses Wasserstein-2 between current post-evidence PMFs, with lowest agent ID breaking exact ties. All random choices use saved uniforms. Fixed, fractional, hybrid max(fixed, fraction times opening cash), or zero specifies one total round budget per payer. An unaffordable total request is skipped. Split the actual budget equally over selected recipients. Zero-payment contacts still occur and may repel; this matters when comparing strategies that make different numbers of contacts.",
                         "evidence": "Public evidence uses one shared Bernoulli observation each update, before social pooling. It is on by default; switch it off for a social-only diagnostic. Conditioning on an outcome with zero probability raises an error.",
                         "timing": "State 0 is initialization. Ordered interventions execute before their one-based update, then shared evidence, recipient selection, total-budget splitting, paid loyalty and unpaid expiry, simultaneous belief pooling, persistent mutual flips, and cash settlement. Manual signs can change again through payments or loyalty expiry in the same round. Saved frames include both sides of each intervention; contact and transfer ledgers describe the single update producing the displayed state.",
                         "manual_interventions": "Person-type and payment-strategy assignments persist until superseded. Entering payment dependence starts fresh clocks from current responses; entering junzi freezes current responses; entering ordinary resumes base signs. Reassigning an unchanged type does not reset its clocks. A mutual sign edit sets both base and effective signs and resets the pair's clocks. Cash adjustments and Gaussian blends occur once. Gaussian target width is a standard deviation, whereas initialization uses variance. Connection removal is rejected. Free manual Gaussian blending is distinct from inactive paid self-reinforcement.",
                         "cash_accounting": "Cash is in arbitrary nominal units. There is no income or active money-burning mechanism. Total cash changes only through external manual adjustments; withdrawals cannot overdraw. Payments only transfer money. Saved cost proposals have no effect on current cash or beliefs.",
                         "proposed_spending": "Inactive: incitation would burn the configurable fixed cost from a distinct A to turn allied B-C into opponents. Reinforcement would burn s and blend toward the original PMF with alpha = 1 - exp(-s / reinforcement_scale). Neither proposal executes or has a log action in this version.",
                         "preservation_metric": "Wasserstein-2 distance = sqrt(integral from 0 to 1 of (Q_current(u) - Q_initial(u)) squared du), where Q is the inverse CDF. Discrete PMFs are matched by probability mass in increasing theta order. It is the minimum root mean squared transport distance along theta, so longer moves receive greater weight. On [0,1] it lies between 0 and 1 and has theta units. Zero means identical PMFs. The population metric is the mean of individual distances. This measures preservation of each original PMF, not accuracy about the true bias. Recipient targeting instead compares the payer's and recipient's current PMFs.",
                         "paired_replay": "Baseline and current run share exact initial arrays, person types, evidence, and target-selection uniforms; the baseline omits interventions. Actual contacts may diverge after changes to beliefs or strategies. Full version-2 experiment JSON fixes setup and inputs. A separate version-2 intervention log replaces the current log against the current setup. Version-1 files require their original implementation; they are not silently reinterpreted. Changing horizon or seed creates a different experiment; no prefix stability across newly generated runs is promised.",
                     }},
    }
    logs = {"scenario": interventions}
    if interventions:
        logs["baseline"] = ()
    for name, events in logs.items():
        try:
            result = simulate_strategy(config, inputs, settings, person_types=person_types, interventions=events)
        except ValueError as exc:
            raise ValueError(f"Complete / {settings.payment_mode} / {name}: {exc}") from exc
        distance = wasserstein_distance(result.beliefs, inputs.initial_beliefs, inputs.theta)
        zero = np.zeros(config.n_agents).tolist()
        zero_transfers = np.zeros((config.n_agents, config.n_agents)).tolist()
        no_contacts = np.zeros((config.n_agents, config.n_agents), dtype=bool).tolist()
        response[name] = {
            "beliefs": result.beliefs[frames].tolist(), "wealth": result.wealth[frames].tolist(),
            "relationships": result.relationships[frames].tolist(),
            "payments": [zero] + result.payments[frames[1:] - 1].tolist(),
            "contacts": [no_contacts] + result.contacts[frames[1:] - 1].tolist(),
            "transfers": [zero_transfers] + result.transfers[frames[1:] - 1].tolist(),
            "persistent_relationships": result.persistent_relationships[frames].tolist(),
            "external_cash": [zero] + result.external_cash[frames[1:] - 1].tolist(),
            "distance": distance[frames].tolist(),
            "metrics": {"wealth_total": result.wealth.sum(axis=1)[frames].tolist(),
                        "mean_distance": distance.mean(axis=1)[frames].tolist()},
        }
    return response
