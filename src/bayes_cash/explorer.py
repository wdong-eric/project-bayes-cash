"""Explore paired belief trajectories using the canonical Python simulator."""

import argparse
from dataclasses import asdict, fields, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import hashlib
import json
from pathlib import Path
import platform
from threading import Lock
from urllib.parse import urlsplit

import networkx as nx
import numpy as np

from .experiment import COLORS, LABELS, diagnostics
from .model import PAYMENT_MODES, Config, Intervention, PreparedInputs, make_inputs, simulate


TOPOLOGIES = {
    "barabasi_albert": "BA hubs",
    "regular": "Random regular",
    "complete": "Complete",
}
MODES = (*PAYMENT_MODES, "no_payment")
DEFAULT_CONFIG = Config(n_agents=24, rounds=160, attachment=2, opponent_fraction=0.2)
MAX_TRAJECTORY_VALUES = 4_000_000
MAX_REQUEST_BYTES = 16 * 1024 * 1024
EXPERIMENT_FORMAT_VERSION = 1
DATA_MARKER = "/*__EXPLORER_DATA__*/"
# A running server must identify the implementation it loaded, even after a source edit.
MODEL_SOURCE_SHA256 = hashlib.sha256(Path(__file__).with_name("model.py").read_bytes()).hexdigest()


def model_identity():
    """Identify the numerical implementation required by a saved experiment."""
    return {
        "sha256": MODEL_SOURCE_SHA256,
        "numpy": np.__version__, "networkx": nx.__version__,
        "python": platform.python_version(),
    }


def parse_request(payload):
    """Validate browser input and its total trajectory budget before allocating."""
    allowed = {"config", "distribution", "topologies", "interventions",
               "format_version", "model", "prepared_inputs"}
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError("request must contain only experiment settings, interventions, and saved inputs")
    if set(payload) & {"format_version", "model", "prepared_inputs"}:
        if not {"format_version", "model", "prepared_inputs", "config", "distribution",
                "topologies", "interventions"} <= set(payload):
            raise ValueError("saved experiment is missing its version, model, settings, inputs, or intervention log")
        if type(payload["format_version"]) is not int or payload["format_version"] != EXPERIMENT_FORMAT_VERSION:
            raise ValueError("unsupported experiment format_version")
        if payload["model"] != model_identity():
            raise ValueError("saved experiment model or dependency version differs; use the original implementation to reproduce it")
    overrides = payload.get("config", {})
    if not isinstance(overrides, dict) or set(overrides) - {field.name for field in fields(Config)}:
        raise ValueError("config contains unknown fields or is not an object")
    config = replace(DEFAULT_CONFIG, **overrides)
    distribution = payload.get("distribution", "loguniform")
    if distribution not in ("uniform", "loguniform"):
        raise ValueError("distribution must be uniform or loguniform")
    topologies = payload.get("topologies", list(TOPOLOGIES))
    if (
        not isinstance(topologies, list) or not topologies
        or not all(isinstance(item, str) and item in TOPOLOGIES for item in topologies)
        or len(set(topologies)) != len(topologies)
    ):
        raise ValueError("topologies must be a nonempty list of distinct supported networks")
    if not 4 <= config.n_agents <= 100:
        raise ValueError("n_agents must be between 4 and 100")
    if not 1 <= config.rounds <= 1000:
        raise ValueError("rounds must be between 1 and 1000")
    if not 5 <= config.grid_size <= 51:
        raise ValueError("grid_size must be between 5 and 51")
    if "regular" in topologies and 2 * config.attachment >= config.n_agents:
        raise ValueError("random-regular degree 2 * attachment must be less than n_agents")
    raw_events = payload.get("interventions", [])
    if not isinstance(raw_events, list) or len(raw_events) > 1000:
        raise ValueError("interventions must be a list of at most 1000 changes")
    event_fields = {field.name for field in fields(Intervention)}
    interventions = []
    for event in raw_events:
        if not isinstance(event, dict) or set(event) - event_fields or not {"agent", "round"} <= set(event):
            raise ValueError("each intervention requires agent, round, and supported behavior settings")
        change = Intervention(**event)
        if (change.agent >= config.n_agents or change.round > config.rounds
            or change.neighbor is not None and change.neighbor >= config.n_agents):
            raise ValueError("intervention agent or round is outside this experiment")
        interventions.append(change)
    frame_count = len(display_frames(config, interventions))
    comparison_count = 2 if interventions else 1
    belief_values = (
        frame_count * config.n_agents * config.grid_size * len(MODES) * len(topologies) * comparison_count
    )
    initial_dynamic = config.conditional_ally_fraction > 0
    intervention_dynamic = any(event.conditional_ally or event.connection is not None for event in interventions)
    relationship_values = frame_count * config.n_agents ** 2 * len(MODES) * len(topologies) * (
        int(initial_dynamic or intervention_dynamic) + int(bool(interventions) and initial_dynamic)
    )
    cash_values = frame_count * config.n_agents * len(MODES) * len(topologies) * comparison_count
    if belief_values + relationship_values + cash_values > MAX_TRAJECTORY_VALUES:
        raise ValueError(
            f"requested {belief_values:,} belief values, {relationship_values:,} relationship values, "
            f"and {cash_values:,} cash adjustment values "
            f"exceeds the {MAX_TRAJECTORY_VALUES:,} total limit"
            f"{' including the baseline comparison' if interventions else ''}; "
            "reduce agents, rounds, grid size, or networks"
        )
    return config, distribution, topologies, interventions


def display_frames(config, interventions):
    """Keep intervention boundaries exact even when the rest is downsampled."""
    return np.unique(np.r_[
        np.linspace(0, config.rounds, min(config.rounds + 1, 501), dtype=int),
        np.array([t for event in interventions for t in (event.round - 1, event.round)], dtype=int),
    ])


def restore_inputs(saved, config, distribution, topologies):
    """Read exact recorded inputs through the model's domain validation."""
    if not isinstance(saved, dict) or set(saved) != set(topologies):
        raise ValueError("saved prepared_inputs must match the selected topologies")
    names = {field.name for field in fields(PreparedInputs)}
    prepared = {}
    for topology in topologies:
        values = saved[topology]
        if not isinstance(values, dict) or set(values) != names:
            raise ValueError("saved prepared_inputs contains missing or unknown fields")
        current = PreparedInputs(**{
            name: value if name == "wealth_distribution" or value is None else np.asarray(value)
            for name, value in values.items()
        })
        if (current.initial_wealth.size != config.n_agents
            or current.theta.size != config.grid_size
            or current.observations.size != config.rounds
            or current.wealth_distribution != distribution):
            raise ValueError("saved inputs do not match the experiment dimensions or wealth distribution")
        if current.contact_uniforms is None or current.giving_preferences is None or current.conditional_allies is None:
            raise ValueError("saved experiments require complete agent traits and contact randomness")
        if not np.array_equal(current.theta, np.linspace(0, 1, config.grid_size)):
            raise ValueError("saved theta must match the experiment's equally spaced grid")
        if prepared:
            first = next(iter(prepared.values()))
            for name in ("theta", "initial_beliefs", "initial_wealth", "observations",
                         "giving_preferences", "conditional_allies"):
                if not np.array_equal(getattr(current, name), getattr(first, name)):
                    raise ValueError(f"saved networks must share {name}")
        prepared[topology] = current
    return prepared


def make_network_inputs(config, distribution, topologies):
    """Pair priors, wealth and evidence; retain the exact requested-horizon BA run.

    Other networks are new full-belief experiments, not replays of the earlier
    cash-only controls. Their graph, contacts and edge signs have independent
    seeded streams. Changing the horizon can change a node's contact sequence.
    """
    original = make_inputs(config, distribution)
    prepared = {}
    for topology in topologies:
        if topology == "barabasi_albert":
            prepared[topology] = original
            continue
        index = tuple(TOPOLOGIES).index(topology)
        graph_rng, contact_rng, relationship_rng, preference_rng = (
            np.random.default_rng(stream)
            for stream in np.random.SeedSequence([config.seed, 837211, index]).spawn(4)
        )
        if topology == "regular":
            degree = 2 * config.attachment
            if degree >= config.n_agents:
                raise ValueError("random-regular degree 2 * attachment must be less than n_agents")
            for _ in range(1000):
                graph = nx.random_regular_graph(
                    degree, config.n_agents,
                    seed=int(graph_rng.integers(0, np.iinfo(np.int32).max)),
                )
                if nx.is_connected(graph):
                    break
            else:
                raise ValueError("could not sample a connected regular network; choose another seed")
            adjacency = nx.to_numpy_array(graph, nodelist=range(config.n_agents), dtype=bool)
        else:
            adjacency = ~np.eye(config.n_agents, dtype=bool)
        relationships = adjacency.astype(np.int8)
        edge_i, edge_j = np.nonzero(np.triu(adjacency, k=1))
        opponents = relationship_rng.random(edge_i.size) < config.opponent_fraction
        relationships[edge_i[opponents], edge_j[opponents]] = -1
        relationships[edge_j[opponents], edge_i[opponents]] = -1
        recipients = np.empty((config.rounds, config.n_agents), dtype=int)
        for payer in range(config.n_agents):
            recipients[:, payer] = contact_rng.choice(
                np.flatnonzero(adjacency[payer]), size=config.rounds,
            )
        prepared[topology] = replace(
            original, adjacency=adjacency, relationships=relationships, recipients=recipients,
            contact_uniforms=preference_rng.random((config.rounds, config.n_agents)),
        )
    return prepared


def simulation_payload(payload):
    """Compute all rounds, then sample display frames without interpolating PMFs."""
    config, distribution, topologies, interventions = parse_request(payload)
    frames = display_frames(config, interventions)
    prepared = (
        restore_inputs(payload["prepared_inputs"], config, distribution, topologies)
        if "prepared_inputs" in payload else make_network_inputs(config, distribution, topologies)
    )
    initial_dynamic_networks = sum(bool(np.any(inputs.conditional_allies)) for inputs in prepared.values())
    dynamic_networks = sum(
        bool(np.any(inputs.conditional_allies)
             or any(event.conditional_ally or event.connection is not None for event in interventions))
        for inputs in prepared.values()
    )
    if interventions:
        dynamic_networks += initial_dynamic_networks
    total_values = len(frames) * len(MODES) * (
        config.n_agents * (config.grid_size + 1) * len(topologies) * (2 if interventions else 1)
        + config.n_agents ** 2 * dynamic_networks
    )
    if total_values > MAX_TRAJECTORY_VALUES:
        raise ValueError(
            "prepared input relationship histories"
            f"{' and baseline comparison' if interventions else ''} "
            "exceed the total trajectory limit; reduce experiment size"
        )
    first = next(iter(prepared.values()))
    experiment = {
        "format_version": EXPERIMENT_FORMAT_VERSION, "model": model_identity(),
        "config": asdict(config), "distribution": distribution, "topologies": topologies,
        "interventions": [
            {name: value for name, value in asdict(event).items() if value is not None}
            for event in interventions
        ],
        "prepared_inputs": {
            topology: {name: value.tolist() if isinstance(value, np.ndarray) else value
                       for name, value in vars(inputs).items()}
            for topology, inputs in prepared.items()
        },
    }
    response = {
        "experiment": experiment,
        "config": asdict(config), "distribution": distribution,
        "theta": first.theta.tolist(), "observations": first.observations.astype(int).tolist(),
        "giving_preferences": first.giving_preferences.tolist(),
        "conditional_allies": first.conditional_allies.tolist(),
        "frames": frames.tolist(), "topologies": [],
        "modes": [{"id": mode, "label": LABELS[mode], "color": COLORS[mode]} for mode in MODES],
        "metadata": {
            "sampled": len(frames) < config.rounds + 1,
            "frame_count": len(frames), "simulated_rounds": config.rounds,
            "conventions": {
                "beliefs": "Probability masses on an equally spaced theta grid, including endpoints; not continuous densities.",
                "rounds": "Frame t is the state after t updates. At t>0 its payment/contact arrays contain ledger index t-1, the update producing that state; frame 0 has zero payments and no contacts. Sampled frames retain actual states, not averages or interpolation.",
                "evidence": "One shared Bernoulli outcome per round when enabled, before social influence and cash settlement.",
                "pairing": "Identical priors, initial wealth, agent traits, and public observations across networks; initial signs and contact randomness shared across payment modes within each network. Realized contacts can diverge as loyalty changes.",
                "baseline": "For each network and payment rule, the baseline uses the identical prepared initial state, observations, and contact random arrays with an empty intervention log. Both trajectories use the same display frames, including intervention boundaries. Realized contacts may diverge after an intervention. With no interventions, the current scenarios are also the baseline and duplicate arrays are omitted. Baselines are derived from saved inputs, not separately stored in experiment JSON.",
                "topologies": "Initial families: BA attachment m; connected random regular degree 2m; complete. Connection interventions subsequently change adjacency and degree-dependent self-weight. BA and regular initial mean degrees are unequal and their densities can differ substantially. Isolated agents observe evidence and receive configured income but have no contacts.",
                "randomness": "BA exactly uses make_inputs at the requested horizon; other graphs use independent graph/contact/sign streams. Changing rounds can change contacts; no prefix-stable claim.",
                "scope": "New exploratory full-belief runs. Non-BA trajectories do not replay the earlier cash-only network controls; a single seed is not an ensemble conclusion.",
                "cash": "Nominal arbitrary units. Cash interventions are external injections or withdrawals before the numbered update, not transfers. Income equals income_rate times opening wealth after those changes; income and transfers settle together. Positive income also caps each requested payment at that round's income. Overdrafts are rejected, not clipped. Total cash change equals external adjustments plus income. Wealth Gini is undefined (null) when total cash is zero.",
                "payments": "Uniform givers sample a neighbor. Preferential givers choose an ally/opponent group using preference_probability and sample uniformly within it; normalize over available groups with positive probability. No eligible group means no contact (recipient -1) and no payment. Opening-round recipient receptivity determines the group. Pay only affordable requests; hybrid requests max(fixed, fraction times opening wealth).",
                "loyalty": "Independent recipient trait. relationships[payer][recipient] is the recipient's receptivity to that payer. Initial sampled signs and zero unpaid counters; positive actual payments immediately set ally and reset that pair's counter. After loyalty_rounds consecutive unpaid simulation rounds, set opponent before influence. Reverse and other-payer relations are independent. Scenario relationship frames are states after the displayed round; static runs use initial edges.",
                "influence": "Self-weight is degree times reference_payment times (1-social_rate)/social_rate. Negative pooling uses the model's positive-part projection and renormalization, which can erase support.",
                "opponents": "Network lines show current connections. Ally, opponent, and mixed directed stances have distinct encodings; the relationship table reports each direction. Attenuation: current opponents have negative weight -reference_payment/(1+payment/reference_payment), including unpaid contacts. Tanh: recipient opening wealth times tanh(payment/wealth - opponent_threshold), with zero ally threshold and zero weight at zero recipient wealth; effective signs can change.",
                "no_payment": "fixed_payment=0 with all other settings retained. Opponents can still repel and configured income is still injected; this is not a social-off control.",
                "interventions": "Changes apply before the numbered update's evidence, contacts, income, and payments, across all networks and payment rules. Earlier states are unchanged. Equal-round changes use log order: trait assignments persist, cash deltas add once, and belief blends compose once. Enabling payment dependence starts fresh unpaid clocks; disabling keeps current relationships. Re-enabling starts fresh clocks again. Repeating an enabled setting preserves clocks. Belief edits blend the current PMF with a Gaussian target normalized by a sum on theta; its center and width are Gaussian input parameters, not the achieved discrete mean and SD. Setting a connection assigns both directed signs and resets that pair's unpaid clocks, including for an existing edge; removing it clears both directions. Uniform givers retain their original scheduled recipient when their neighbor set matches its initial state; otherwise stored contact uniforms choose among sorted current neighbors. Saved experiment JSON records exact initial inputs, random draws, ordered changes, and model/dependency identity.",
            },
        },
    }
    scenario_logs = {"scenarios": interventions}
    if interventions:
        scenario_logs["baseline_scenarios"] = ()
    layout_seed = int(np.random.default_rng(np.random.SeedSequence([config.seed, 73665])).integers(0, 2**31))
    for topology, inputs in prepared.items():
        graph = nx.from_numpy_array(inputs.adjacency)
        positions = (
            nx.circular_layout(graph, scale=0.45, center=(0.5, 0.5))
            if topology == "complete" else
            nx.spring_layout(graph, seed=layout_seed, iterations=60, scale=0.45, center=(0.5, 0.5))
        )
        edge_i, edge_j = np.nonzero(np.triu(inputs.adjacency, k=1))
        network = {
            "id": topology, "label": TOPOLOGIES[topology],
            "positions": [positions[node].tolist() for node in range(config.n_agents)],
            "edges": np.column_stack((edge_i, edge_j, inputs.relationships[edge_i, edge_j])).tolist(),
            "degrees": inputs.adjacency.sum(axis=1).tolist(),
            "realized_opponent_fraction": float(np.mean(inputs.relationships[edge_i, edge_j] < 0)),
        }
        for scenario_key, changes in scenario_logs.items():
            network[scenario_key] = {}
            for mode in MODES:
                try:
                    result = (
                        simulate(replace(config, fixed_payment=0.0), inputs, "fixed", interventions=changes)
                        if mode == "no_payment" else simulate(config, inputs, mode, interventions=changes)
                    )
                except ValueError as exc:
                    comparison = " / baseline without interventions" if scenario_key == "baseline_scenarios" else ""
                    raise ValueError(f"{TOPOLOGIES[topology]} / {LABELS[mode]}{comparison}: {exc}") from exc
                metrics = diagnostics(result, inputs.theta, config.true_bias)
                external_cash = result.external_cash.sum(axis=1)
                metrics["total_external_cash"] = np.r_[0.0, external_cash]
                metrics["cumulative_external_cash"] = np.cumsum(metrics["total_external_cash"])
                metrics["total_income"] = np.r_[0.0, config.income_rate * (metrics["wealth_total"][:-1] + external_cash)]
                network[scenario_key][mode] = {
                    "beliefs": result.beliefs[frames].tolist(),
                    "wealth": result.wealth[frames].tolist(),
                    "payments": [np.zeros(config.n_agents).tolist()] + result.payments[frames[1:] - 1].tolist(),
                    "recipients": [[]] + result.recipients[frames[1:] - 1].tolist(),
                    "external_cash": [np.zeros(config.n_agents).tolist()] + result.external_cash[frames[1:] - 1].tolist(),
                    "metrics": {
                        key: [None if np.isnan(value) else value for value in values[frames].tolist()]
                        if key == "wealth_gini" else values[frames].tolist()
                        for key, values in metrics.items()
                    },
                }
                if np.any(inputs.conditional_allies) or any(event.conditional_ally or event.connection is not None for event in changes):
                    network[scenario_key][mode]["relationships"] = result.relationships[frames].tolist()
        response["topologies"].append(network)
    return response


def render_html(data=None, *, template_name="explorer.html"):
    """Embed replay data without allowing JSON text to close its script element."""
    template = files("bayes_cash").joinpath(f"web/{template_name}").read_text(encoding="utf-8")
    if data is None:
        return template
    serialized = json.dumps(data, allow_nan=False, separators=(",", ":"))
    serialized = serialized.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    if DATA_MARKER not in template:
        raise ValueError("explorer template is missing its data marker")
    return template.replace(DATA_MARKER, f"window.EXPLORER_DATA = {serialized};", 1)


def export_html(destination, payload=None):
    """Write a portable replay of one run, refusing to overwrite existing files."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}; choose a new --export path.")
    payload = {} if payload is None else payload
    if isinstance(payload, dict) and payload.get("kind") == "belief-preservation":
        from .strategy_explorer import simulation_payload as strategy_payload

        html = render_html(strategy_payload(payload), template_name="strategy.html")
    else:
        html = render_html(simulation_payload(payload))
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(html)
    return destination


class ExplorerHandler(BaseHTTPRequestHandler):
    """Only the explorer and simulation endpoint are exposed on loopback."""

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def reply(self, status, body, content_type="application/json; charset=utf-8"):
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(encoded)

    def error(self, status, message):
        self.reply(status, json.dumps({"error": message}))

    def valid_host(self):
        host = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if host not in allowed:
            self.error(403, "Only the local explorer host is accepted")
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin != f"http://{host}":
            self.error(403, "Cross-origin requests are not accepted")
            return False
        return True

    def do_GET(self):
        if not self.valid_host():
            return
        path = urlsplit(self.path).path
        if path not in ("/", "/index.html", "/strategy.html"):
            self.error(404, "Not found")
            return
        template_name = "strategy.html" if path == "/strategy.html" else "explorer.html"
        self.reply(200, render_html(template_name=template_name), "text/html; charset=utf-8")

    def do_POST(self):
        if not self.valid_host():
            return
        if self.path not in ("/api/simulate", "/api/strategy/simulate"):
            self.error(404, "Not found")
            return
        if self.headers.get_content_type() != "application/json":
            self.error(415, "Content-Type must be application/json")
            return
        if self.headers.get("Transfer-Encoding"):
            self.error(400, "Transfer-Encoding is not supported")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.error(400, "Invalid Content-Length")
            return
        if not 1 <= length <= MAX_REQUEST_BYTES:
            self.error(413, f"Request body must contain 1 to {MAX_REQUEST_BYTES} bytes")
            return
        if not self.server.simulation_lock.acquire(blocking=False):
            self.error(429, "A simulation is already running; try again when it finishes")
            return
        try:
            payload = json.loads(self.rfile.read(length))
            if self.path == "/api/strategy/simulate":
                from .strategy_explorer import simulation_payload as strategy_payload

                result = strategy_payload(payload)
            else:
                result = simulation_payload(payload)
            self.reply(200, json.dumps(result, allow_nan=False, separators=(",", ":")))
        except (ValueError, TypeError, OverflowError) as exc:
            self.error(400, str(exc))
        finally:
            self.server.simulation_lock.release()


def make_server(port=8765):
    server = ThreadingHTTPServer(("127.0.0.1", port), ExplorerHandler)
    server.simulation_lock = Lock()
    return server


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "With no arguments, start the local app and open its printed URL in a browser. "
            "Saved HTML replays work offline; new simulations require the local app. "
            "Example: python -m bayes_cash.explorer --export replay.html --config request.json"
        ),
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Local HTTP port (default: 8765); use 0 to choose an available port",
    )
    parser.add_argument(
        "--export", type=Path, metavar="HTML",
        help="Save a standalone replay and exit; existing files are never overwritten",
    )
    parser.add_argument(
        "--config", type=Path, metavar="JSON",
        help="Request settings or a saved experiment JSON, including interventions, for --export",
    )
    args = parser.parse_args()
    if args.config and not args.export:
        parser.error("--config requires --export")
    if args.export:
        payload = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        print(f"Saved standalone replay to {export_html(args.export, payload).resolve()}")
        return
    with make_server(args.port) as server:
        print(f"Open http://127.0.0.1:{server.server_port}/ (Ctrl-C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
