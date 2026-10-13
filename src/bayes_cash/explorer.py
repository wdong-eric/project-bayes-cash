"""Explore paired belief trajectories using the canonical Python simulator."""

import argparse
from dataclasses import asdict, fields, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

import networkx as nx
import numpy as np

from .experiment import COLORS, LABELS, diagnostics
from .model import PAYMENT_MODES, Config, PreparedInputs, make_inputs, simulate


TOPOLOGIES = {
    "barabasi_albert": "BA hubs",
    "regular": "Random regular",
    "complete": "Complete",
}
MODES = (*PAYMENT_MODES, "no_payment")
DEFAULT_CONFIG = Config(n_agents=24, rounds=160, attachment=2, opponent_fraction=0.2)
MAX_BELIEF_VALUES = 4_000_000
MAX_REQUEST_BYTES = 32_768
DATA_MARKER = "/*__EXPLORER_DATA__*/"


def parse_request(payload):
    """Validate browser input and its total trajectory budget before allocating."""
    if not isinstance(payload, dict) or set(payload) - {"config", "distribution", "topologies"}:
        raise ValueError("request must contain only config, distribution, and topologies")
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
    frame_count = min(config.rounds + 1, 501)
    belief_values = frame_count * config.n_agents * config.grid_size * len(MODES) * len(topologies)
    if belief_values > MAX_BELIEF_VALUES:
        raise ValueError(
            f"requested {belief_values:,} belief values exceeds the {MAX_BELIEF_VALUES:,} limit; "
            "reduce agents, rounds, grid size, or networks"
        )
    return config, distribution, topologies


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
        graph_rng, contact_rng, relationship_rng = (
            np.random.default_rng(stream)
            for stream in np.random.SeedSequence([config.seed, 837211, index]).spawn(3)
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
        prepared[topology] = PreparedInputs(
            original.theta, adjacency, relationships, original.initial_beliefs,
            original.initial_wealth, original.observations, recipients, distribution,
        )
    return prepared


def simulation_payload(payload):
    """Compute all rounds, then sample display frames without interpolating PMFs."""
    config, distribution, topologies = parse_request(payload)
    frames = np.linspace(0, config.rounds, min(config.rounds + 1, 501), dtype=int)
    prepared = make_network_inputs(config, distribution, topologies)
    first = next(iter(prepared.values()))
    response = {
        "config": asdict(config), "distribution": distribution,
        "theta": first.theta.tolist(), "observations": first.observations.astype(int).tolist(),
        "frames": frames.tolist(), "topologies": [],
        "modes": [{"id": mode, "label": LABELS[mode], "color": COLORS[mode]} for mode in MODES],
        "metadata": {
            "sampled": len(frames) < config.rounds + 1,
            "frame_count": len(frames), "simulated_rounds": config.rounds,
            "conventions": {
                "beliefs": "Probability masses on an equally spaced theta grid, including endpoints; not continuous densities.",
                "rounds": "Frame t is the state after t updates. At t>0 its payment/contact arrays contain ledger index t-1, the update producing that state; frame 0 has zero payments and no contacts. Sampled frames retain actual states, not averages or interpolation.",
                "evidence": "One shared Bernoulli outcome per round when enabled, before social influence and cash settlement.",
                "pairing": "Identical priors, initial wealth, and public observations across networks; contacts and edge signs shared across payment modes within each network.",
                "topologies": "BA attachment m; connected random regular degree 2m; complete. These also change degree-dependent self-weight. BA and regular mean degrees are close but unequal.",
                "randomness": "BA exactly uses make_inputs at the requested horizon; other graphs use independent graph/contact/sign streams. Changing rounds can change contacts; no prefix-stable claim.",
                "scope": "New exploratory full-belief runs. Non-BA trajectories do not replay the earlier cash-only network controls; a single seed is not an ensemble conclusion.",
                "cash": "Nominal arbitrary units. Income equals income_rate times opening wealth; income and transfers settle together. Positive income also caps each requested payment at that round's income.",
                "payments": "One uniformly sampled neighbor per payer per round. Pay the requested amount only if affordable; otherwise skip. Hybrid requests max(fixed, fraction times opening wealth).",
                "influence": "Self-weight is degree times reference_payment times (1-social_rate)/social_rate. Negative pooling uses the model's positive-part projection and renormalization, which can erase support.",
                "opponents": "Static baseline edge signs. Attenuation: opponents have negative weight -reference_payment/(1+payment/reference_payment), including unpaid contacts. Tanh: recipient opening wealth times tanh(payment/wealth - opponent_threshold), with zero ally threshold and zero weight at zero recipient wealth; effective signs can change.",
                "no_payment": "fixed_payment=0 with all other settings retained. Opponents can still repel and configured income is still injected; this is not a social-off control.",
            },
        },
    }
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
            "recipients": [[]] + inputs.recipients[frames[1:] - 1].tolist(),
            "realized_opponent_fraction": float(np.mean(inputs.relationships[edge_i, edge_j] < 0)),
            "scenarios": {},
        }
        for mode in MODES:
            result = (
                simulate(replace(config, fixed_payment=0.0), inputs, "fixed")
                if mode == "no_payment" else simulate(config, inputs, mode)
            )
            metrics = diagnostics(result, inputs.theta, config.true_bias)
            metrics["total_income"] = np.r_[0.0, config.income_rate * metrics["wealth_total"][:-1]]
            network["scenarios"][mode] = {
                "beliefs": result.beliefs[frames].tolist(),
                "wealth": result.wealth[frames].tolist(),
                "payments": [np.zeros(config.n_agents).tolist()] + result.payments[frames[1:] - 1].tolist(),
                "metrics": {key: values[frames].tolist() for key, values in metrics.items()},
            }
        response["topologies"].append(network)
    return response


def render_html(data=None):
    """Embed replay data without allowing JSON text to close its script element."""
    template = files("bayes_cash").joinpath("web/explorer.html").read_text(encoding="utf-8")
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
    html = render_html(simulation_payload({} if payload is None else payload))
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
        if urlsplit(self.path).path not in ("/", "/index.html"):
            self.error(404, "Not found")
            return
        self.reply(200, render_html(), "text/html; charset=utf-8")

    def do_POST(self):
        if not self.valid_host():
            return
        if self.path != "/api/simulate":
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
        help="Settings for --export: config overrides, distribution, and topologies (otherwise viewer defaults)",
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
