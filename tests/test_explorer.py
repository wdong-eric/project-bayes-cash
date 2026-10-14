"""Protect simulator fidelity, paired controls, and the local viewer boundary."""

from dataclasses import replace
from http.client import HTTPConnection
import json
from threading import Thread

import networkx as nx
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from bayes_cash import explorer
from bayes_cash.model import Config, make_inputs, simulate


@pytest.fixture
def request_payload():
    return {
        "config": {"n_agents": 8, "rounds": 12, "attachment": 2, "grid_size": 7},
        "distribution": "loguniform", "topologies": list(explorer.TOPOLOGIES),
    }


def test_network_inputs_pair_all_common_randomness_and_preserve_ba():
    config = Config(
        n_agents=8, rounds=505, attachment=2, grid_size=7, opponent_fraction=0.3,
        ally_giver_fraction=0.5, opponent_giver_fraction=0.5, conditional_ally_fraction=0.5,
    )
    original = make_inputs(config, "uniform")
    prepared = explorer.make_network_inputs(config, "uniform", list(explorer.TOPOLOGIES))
    for name in original.__dataclass_fields__:
        assert_array_equal(getattr(prepared["barabasi_albert"], name), getattr(original, name))
    for topology, current in prepared.items():
        for name in ("initial_wealth", "initial_beliefs", "observations", "theta", "giving_preferences", "conditional_allies"):
            assert_array_equal(getattr(current, name), getattr(original, name))
        assert nx.is_connected(nx.from_numpy_array(current.adjacency))
        assert current.adjacency[np.arange(config.n_agents), current.recipients].all()
        isolated = explorer.make_network_inputs(config, "uniform", [topology])[topology]
        assert_array_equal(current.recipients, isolated.recipients)
        assert_array_equal(current.relationships, isolated.relationships)
        assert_array_equal(current.contact_uniforms, isolated.contact_uniforms)
    assert_array_equal(prepared["regular"].adjacency.sum(axis=1), np.full(8, 4))
    assert_array_equal(prepared["complete"].adjacency.sum(axis=1), np.full(8, 7))


@pytest.mark.parametrize("traits", [{}, {
    "ally_giver_fraction": 0.5, "opponent_giver_fraction": 0.5,
    "conditional_ally_fraction": 1.0,
}])
def test_ba_payload_matches_canonical_simulation_and_sampled_rounds(request_payload, traits):
    request_payload["config"]["rounds"] = 505
    request_payload["config"].update(traits)
    request_payload["topologies"] = ["barabasi_albert"]
    payload = explorer.simulation_payload(request_payload)
    config = Config(**payload["config"])
    inputs = make_inputs(config, "loguniform")
    frames = np.array(payload["frames"])
    assert len(frames) == 501
    assert frames[0] == 0 and frames[-1] == config.rounds
    assert np.all(np.diff(frames) > 0)
    assert payload["metadata"]["sampled"]
    network = payload["topologies"][0]
    assert_array_equal(payload["giving_preferences"], inputs.giving_preferences)
    assert_array_equal(payload["conditional_allies"], inputs.conditional_allies)
    for mode in explorer.MODES:
        run_config = replace(config, fixed_payment=0) if mode == "no_payment" else config
        expected = simulate(run_config, inputs, "fixed" if mode == "no_payment" else mode)
        actual = network["scenarios"][mode]
        assert_array_equal(actual["beliefs"], expected.beliefs[frames])
        assert_array_equal(actual["wealth"], expected.wealth[frames])
        assert_array_equal(actual["payments"][0], np.zeros(config.n_agents))
        assert_array_equal(actual["payments"][1:], expected.payments[frames[1:] - 1])
        assert actual["recipients"][0] == []
        assert_array_equal(actual["recipients"][1:], expected.recipients[frames[1:] - 1])
        if traits:
            assert_array_equal(actual["relationships"], expected.relationships[frames])
        else:
            assert "relationships" not in actual


@pytest.mark.parametrize("income_rate", [0.0, 0.02])
@pytest.mark.parametrize("influence_rule", ["attenuation", "tanh"])
def test_all_topology_modes_obey_mass_and_cash_balance(request_payload, income_rate, influence_rule):
    request_payload["config"].update(
        income_rate=income_rate, opponent_fraction=0.25, influence_rule=influence_rule,
    )
    payload = explorer.simulation_payload(request_payload)
    initial = payload["topologies"][0]["scenarios"]["fixed"]
    for network in payload["topologies"]:
        for mode, scenario in network["scenarios"].items():
            beliefs = np.array(scenario["beliefs"])
            wealth = np.array(scenario["wealth"])
            assert_array_equal(beliefs[0], initial["beliefs"][0])
            assert_array_equal(wealth[0], initial["wealth"][0])
            assert_allclose(beliefs.sum(axis=2), 1, atol=1e-12)
            assert np.min(beliefs) >= 0 and np.min(wealth) >= 0
            total = wealth.sum(axis=1)
            assert_allclose(total, total[0] * (1 + income_rate) ** np.arange(len(total)), rtol=1e-12)
            assert_allclose(scenario["metrics"]["total_income"], np.r_[0, income_rate * total[:-1]])
            if mode == "no_payment":
                assert_array_equal(scenario["payments"], np.zeros_like(wealth))
                assert_allclose(wealth, wealth[0] * (1 + income_rate) ** np.arange(len(total))[:, None])


def test_no_payment_preserves_opponent_influence(request_payload):
    request_payload["config"].update(opponent_fraction=1.0, public_evidence=False)
    payload = explorer.simulation_payload(request_payload)
    config = Config(**payload["config"])
    prepared = explorer.make_network_inputs(config, "loguniform", list(explorer.TOPOLOGIES))
    for network in payload["topologies"]:
        control = network["scenarios"]["no_payment"]
        canonical = simulate(replace(config, fixed_payment=0), prepared[network["id"]], "fixed")
        assert_array_equal(control["beliefs"], canonical.beliefs)
        assert not np.allclose(control["beliefs"][0], control["beliefs"][-1])


@pytest.mark.parametrize("payload, message", [
    ([], "request must"),
    ({"mystery": 1}, "request must"),
    ({"config": {"mystery": 1}}, "unknown fields"),
    ({"config": {"n_agents": True}}, "integer"),
    ({"config": {"n_agents": 101}}, "n_agents"),
    ({"config": {"rounds": 1001}}, "rounds"),
    ({"config": {"grid_size": 52}}, "grid_size"),
    ({"config": {"n_agents": 6, "attachment": 3}}, "2 \\* attachment"),
    ({"config": {"n_agents": 100, "rounds": 1000, "grid_size": 51}}, "belief values"),
    ({"config": {"income_rate": float("nan")}}, "finite real"),
    ({"config": {"ally_giver_fraction": 0.8, "opponent_giver_fraction": 0.8}}, "fraction"),
    ({"config": {"preference_probability": 1.1}}, "preference_probability"),
    ({"config": {"loyalty_rounds": 0}}, "loyalty_rounds"),
    ({"config": {"n_agents": 100, "rounds": 100, "conditional_ally_fraction": 1}}, "relationship values"),
    ({"distribution": "normal"}, "distribution"),
    ({"topologies": []}, "topologies"),
    ({"topologies": ["regular", "regular"]}, "topologies"),
    ({"topologies": ["unrecognized"]}, "topologies"),
])
def test_request_rejects_invalid_or_excessive_work_before_allocation(payload, message, monkeypatch):
    def unexpected_allocation(*args):
        pytest.fail("invalid requests must be rejected before preparing arrays")
    monkeypatch.setattr(explorer, "make_network_inputs", unexpected_allocation)
    with pytest.raises(ValueError, match=message):
        explorer.simulation_payload(payload)


@pytest.fixture
def server():
    server = explorer.make_server(0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def send_request(server, method, path, body=None, headers=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    connection.request(method, path, body=body, headers={} if headers is None else headers)
    response = connection.getresponse()
    result = response.status, response.read(), dict(response.getheaders())
    connection.close()
    return result


def test_endpoint_runs_canonical_payload(server, request_payload):
    status, body, headers = send_request(
        server, "POST", "/api/simulate", json.dumps(request_payload),
        {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{server.server_port}"},
    )
    assert status == 200
    actual = json.loads(body)
    assert actual["config"]["n_agents"] == 8
    assert len(actual["topologies"]) == 3
    assert headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.parametrize("method,path,body,headers,status", [
    ("GET", "/../../pyproject.toml", None, {}, 404),
    ("GET", "/", None, {"Host": "evil.example"}, 403),
    ("POST", "/api/simulate", "{}", {"Content-Type": "text/plain"}, 415),
    ("POST", "/api/simulate", "{}", {"Content-Type": "application/json", "Origin": "https://evil.example"}, 403),
    ("POST", "/api/simulate", "{broken", {"Content-Type": "application/json"}, 400),
    ("POST", "/api/simulate", '{"topologies":[]}', {"Content-Type": "application/json"}, 400),
    ("POST", "/api/simulate", '{"config":{"income_rate":1e308}}', {"Content-Type": "application/json"}, 400),
    ("POST", "/api/simulate", " " * (explorer.MAX_REQUEST_BYTES + 1), {"Content-Type": "application/json"}, 413),
    ("POST", "/other", "{}", {"Content-Type": "application/json"}, 404),
], ids=["unknown-file", "untrusted-host", "wrong-content-type", "cross-origin", "invalid-json", "invalid-config", "model-error", "oversize", "unknown-api"])
def test_endpoint_errors_are_structured(server, method, path, body, headers, status):
    actual, data, _ = send_request(server, method, path, body, headers)
    assert actual == status
    assert isinstance(json.loads(data)["error"], str)


def test_server_bounds_concurrent_simulation_work(server):
    with server.simulation_lock:
        status, body, _ = send_request(server, "POST", "/api/simulate", "{}", {"Content-Type": "application/json"})
    assert status == 429
    assert "already running" in json.loads(body)["error"]


def test_standalone_export_embeds_safely_and_refuses_overwrite(tmp_path, request_payload, monkeypatch):
    template = tmp_path / "web"
    template.mkdir()
    (template / "explorer.html").write_text(f"<script>{explorer.DATA_MARKER}</script>")
    monkeypatch.setattr(explorer, "files", lambda package: tmp_path)
    dangerous = {"label": "</script><script>alert(1)</script>\u2028"}
    rendered = explorer.render_html(dangerous)
    assert rendered.count("</script>") == 1
    assert "\\u003c/script>" in rendered
    assert "\\u2028" in rendered
    target = tmp_path / "replay.html"
    explorer.export_html(target, request_payload)
    assert "window.EXPLORER_DATA = " in target.read_text()
    assert explorer.DATA_MARKER not in target.read_text()
    original = target.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        explorer.export_html(target, request_payload)
    assert target.read_bytes() == original
