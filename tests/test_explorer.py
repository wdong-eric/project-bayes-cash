"""Protect simulator fidelity, paired controls, and the local viewer boundary."""

from dataclasses import fields, replace
from html.parser import HTMLParser
from http.client import HTTPConnection
import json
from threading import Thread

import networkx as nx
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from bayes_cash import explorer
from bayes_cash.model import Config, Intervention, make_inputs, simulate


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
    assert "baseline_scenarios" not in network
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
    ("POST", "/api/simulate", "{}", {"Content-Type": "application/json", "Content-Length": str(explorer.MAX_REQUEST_BYTES + 1)}, 413),
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


def test_conventions_document_every_model_and_intervention_parameter():
    class ParameterReference(HTMLParser):
        def __init__(self):
            super().__init__()
            self.depth = 0
            self.documented = {"data-parameter": set(), "data-intervention-parameter": set()}

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "details" and (self.depth or attrs.get("id") == "model-conventions"):
                self.depth += 1
            if self.depth:
                for attribute, names in self.documented.items():
                    names.update(attrs.get(attribute, "").split())

        def handle_endtag(self, tag):
            if tag == "details" and self.depth:
                self.depth -= 1

    reference = ParameterReference()
    reference.feed(explorer.render_html())
    assert reference.documented["data-parameter"] == {
        field.name for field in fields(Config)
    } | {"distribution", "topologies", "payment_mode"}
    assert reference.documented["data-intervention-parameter"] == {
        field.name for field in fields(Intervention)
    }


def test_saved_interventions_round_trip_without_regenerating_inputs(request_payload, monkeypatch):
    request_payload["interventions"] = [
        {"agent": 0, "round": 5, "giving_preference": -1},
        {"agent": 1, "round": 7, "conditional_ally": True},
        {"agent": 0, "round": 9, "giving_preference": 1},
    ]
    original = explorer.simulation_payload(request_payload)
    saved = json.loads(json.dumps(original["experiment"]))
    assert saved["interventions"] == request_payload["interventions"]
    assert saved["model"] == explorer.model_identity()
    assert all("baseline_scenarios" in network for network in original["topologies"])
    assert "baseline_scenarios" not in saved

    def unexpected_generation(*args):
        pytest.fail("saved experiment must use its exact recorded inputs")

    monkeypatch.setattr(explorer, "make_network_inputs", unexpected_generation)
    restored = explorer.simulation_payload(saved)
    assert restored == original


def test_appended_edit_preserves_prior_history_and_shared_inputs(request_payload):
    baseline = explorer.simulation_payload(request_payload)
    saved = json.loads(json.dumps(baseline["experiment"]))
    saved["interventions"].append({"agent": 0, "round": 6, "conditional_ally": True})
    edited = explorer.simulation_payload(saved)
    assert edited["experiment"]["prepared_inputs"] == baseline["experiment"]["prepared_inputs"]
    for before, after in zip(baseline["topologies"], edited["topologies"], strict=True):
        for mode in explorer.MODES:
            for field in ("beliefs", "wealth", "payments", "recipients"):
                assert after["scenarios"][mode][field][:6] == before["scenarios"][mode][field][:6]
            assert "relationships" in after["scenarios"][mode]


def test_interventions_retain_exact_display_boundaries(request_payload):
    request_payload["config"]["rounds"] = 1000
    request_payload["topologies"] = ["barabasi_albert"]
    request_payload["interventions"] = [{"agent": 0, "round": 51, "giving_preference": 1}]
    result = explorer.simulation_payload(request_payload)
    assert 50 in result["frames"] and 51 in result["frames"]
    assert len(result["frames"]) == 502
    config, distribution, topologies, _ = explorer.parse_request(request_payload)
    inputs = explorer.restore_inputs(result["experiment"]["prepared_inputs"], config, distribution, topologies)
    frames = np.asarray(result["frames"])
    network = result["topologies"][0]
    for mode in explorer.MODES:
        run_config = replace(config, fixed_payment=0) if mode == "no_payment" else config
        expected = simulate(run_config, inputs[network["id"]], "fixed" if mode == "no_payment" else mode)
        baseline = network["baseline_scenarios"][mode]
        assert_array_equal(baseline["beliefs"], expected.beliefs[frames])
        assert_array_equal(baseline["wealth"], expected.wealth[frames])
        assert_array_equal(baseline["payments"][1:], expected.payments[frames[1:] - 1])
        assert_array_equal(baseline["recipients"][1:], expected.recipients[frames[1:] - 1])
        assert len(baseline["beliefs"]) == len(network["scenarios"][mode]["beliefs"])


@pytest.mark.parametrize("event", [
    {"agent": 0, "round": 0, "giving_preference": 1},
    {"agent": -1, "round": 1, "giving_preference": 1},
    {"agent": 8, "round": 1, "giving_preference": 1},
    {"agent": 0, "round": 13, "giving_preference": 1},
    {"agent": True, "round": 1, "giving_preference": 1},
    {"agent": 0, "round": 1, "giving_preference": 2},
    {"agent": 0, "round": 1, "conditional_ally": "true"},
    {"agent": 0, "round": 1},
    {"agent": 0, "round": 1, "social_rate": 0.5},
])
def test_invalid_intervention_rejected_before_preparing_inputs(request_payload, event, monkeypatch):
    request_payload["interventions"] = [event]
    monkeypatch.setattr(explorer, "make_network_inputs", lambda *args: pytest.fail("unexpected allocation"))
    with pytest.raises(ValueError):
        explorer.simulation_payload(request_payload)


def test_intervention_loyalty_histories_count_toward_export_limit():
    with pytest.raises(ValueError, match="relationship values"):
        explorer.simulation_payload({
            "config": {"n_agents": 100, "rounds": 100},
            "interventions": [{"agent": 0, "round": 50, "conditional_ally": True}],
        })


def test_saved_experiment_rejects_version_mismatch_and_invalid_inputs(request_payload):
    saved = explorer.simulation_payload(request_payload)["experiment"]
    changed = dict(saved, model={**saved["model"], "sha256": "different-model"})
    with pytest.raises(ValueError, match="model or dependency version differs"):
        explorer.simulation_payload(changed)
    with pytest.raises(ValueError, match="format_version"):
        explorer.simulation_payload(dict(saved, format_version=True))
    saved["prepared_inputs"]["barabasi_albert"]["initial_beliefs"][0][0] = -1
    with pytest.raises(ValueError, match="nonnegative probability"):
        explorer.simulation_payload(saved)


def test_saved_experiment_endpoint_and_offline_log(server, request_payload, tmp_path):
    request_payload["interventions"] = [{"agent": 2, "round": 3, "giving_preference": -1}]
    initial = explorer.simulation_payload(request_payload)
    status, body, _ = send_request(
        server, "POST", "/api/simulate", json.dumps(initial["experiment"]),
        {"Content-Type": "application/json"},
    )
    assert status == 200
    assert json.loads(body) == initial
    exported = explorer.export_html(tmp_path / "intervention.html", initial["experiment"])
    assert '"interventions":[{"agent":2,"round":3,"giving_preference":-1}]' in exported.read_text()


def test_mixed_interventions_preserve_accounting_connections_and_exact_replay(request_payload):
    request_payload["config"].update(income_rate=0.02, public_evidence=False)
    baseline = explorer.simulation_payload(request_payload)
    saved = baseline["experiment"]
    saved["interventions"] = [
        {"agent": 0, "round": 3, "cash_delta": 25},
        {"agent": 1, "round": 5, "belief_center": 0.8, "belief_width": 0.15, "belief_strength": 0.6},
        {"agent": 0, "round": 6, "neighbor": 1, "connection": 0},
        {"agent": 0, "round": 8, "neighbor": 1, "connection": -1},
    ]
    changed = explorer.simulation_payload(saved)
    assert explorer.simulation_payload(json.loads(json.dumps(changed["experiment"]))) == changed
    for before, after in zip(baseline["topologies"], changed["topologies"], strict=True):
        assert after["baseline_scenarios"] == before["scenarios"]
        for mode in explorer.MODES:
            scenario = after["scenarios"][mode]
            assert scenario["beliefs"][:3] == before["scenarios"][mode]["beliefs"][:3]
            assert scenario["wealth"][:3] == before["scenarios"][mode]["wealth"][:3]
            metrics = scenario["metrics"]
            assert_allclose(np.diff(metrics["wealth_total"]),
                            np.array(metrics["total_income"])[1:] + np.array(metrics["total_external_cash"])[1:])
            assert metrics["total_external_cash"][3] == 25
            assert metrics["cumulative_external_cash"][-1] == 25
            assert scenario["external_cash"][3][0] == 25
            assert np.all(np.array(scenario["relationships"])[6:8, 0, 1] == 0)
            assert scenario["relationships"][8][0][1] == -1
            assert scenario["relationships"][8][1][0] == -1


def test_connection_histories_count_toward_export_limit():
    with pytest.raises(ValueError, match="relationship values"):
        explorer.simulation_payload({
            "config": {"n_agents": 100, "rounds": 100},
            "interventions": [{"agent": 0, "round": 50, "neighbor": 1, "connection": 0}],
        })


def test_withdrawal_error_identifies_scenario_and_preserves_saved_inputs(request_payload):
    saved = explorer.simulation_payload(request_payload)["experiment"]
    original_inputs = json.dumps(saved["prepared_inputs"])
    saved["interventions"] = [{"agent": 0, "round": 1, "cash_delta": -1e6}]
    with pytest.raises(ValueError, match="BA hubs / Fixed payment"):
        explorer.simulation_payload(saved)
    assert json.dumps(saved["prepared_inputs"]) == original_inputs


def test_zero_total_cash_is_valid_with_explicitly_undefined_gini(request_payload):
    saved = explorer.simulation_payload(request_payload)["experiment"]
    wealth = saved["prepared_inputs"]["barabasi_albert"]["initial_wealth"]
    saved["interventions"] = [{"agent": agent, "round": 1, "cash_delta": -value}
                              for agent, value in enumerate(wealth)]
    payload = explorer.simulation_payload(saved)
    json.dumps(payload, allow_nan=False)
    for network in payload["topologies"]:
        for scenario in network["scenarios"].values():
            assert_allclose(scenario["wealth"][1:], 0)
            assert scenario["metrics"]["wealth_gini"][1:] == [None] * 12
            assert_allclose(scenario["metrics"]["cumulative_external_cash"][-1], -sum(wealth))


@pytest.mark.parametrize("event", [
    {"agent": 0, "round": 1, "neighbor": 8, "connection": 1},
    {"agent": 0, "round": 1, "neighbor": 0, "connection": 1},
    {"agent": 0, "round": 1, "cash_delta": "10"},
    {"agent": 0, "round": 1, "belief_center": 0.8, "belief_width": 0.0, "belief_strength": 0.5},
    {"agent": 0, "round": 1, "belief_strength": 0.5},
])
def test_extended_interventions_validate_before_network_generation(request_payload, event, monkeypatch):
    request_payload["interventions"] = [event]
    monkeypatch.setattr(explorer, "make_network_inputs", lambda *args: pytest.fail("unexpected allocation"))
    with pytest.raises(ValueError):
        explorer.simulation_payload(request_payload)


@pytest.mark.parametrize("traits", [{}, {
    "conditional_ally_fraction": 0.5, "ally_giver_fraction": 0.4,
    "opponent_giver_fraction": 0.3,
}])
def test_baseline_reuses_exact_prepared_objects_and_matches_empty_log_everywhere(request_payload, traits, monkeypatch):
    request_payload["config"].update(traits, income_rate=0.02, public_evidence=False)
    request_payload["interventions"] = [
        {"agent": 0, "round": 4, "cash_delta": 25.0},
        {"agent": 1, "round": 5, "belief_center": 0.8, "belief_width": 0.15, "belief_strength": 0.6},
        {"agent": 0, "round": 6, "neighbor": 1, "connection": 0},
        {"agent": 2, "round": 7, "conditional_ally": True},
    ]
    config, distribution, topologies, _ = explorer.parse_request(request_payload)
    prepared = explorer.make_network_inputs(config, distribution, topologies)
    prepared_calls = []
    simulation_calls = []

    def reuse_inputs(*args):
        prepared_calls.append(args)
        return prepared

    def record_simulation(run_config, inputs, mode, *, interventions=()):
        simulation_calls.append((run_config, inputs, mode, bool(interventions)))
        return simulate(run_config, inputs, mode, interventions=interventions)

    monkeypatch.setattr(explorer, "make_network_inputs", reuse_inputs)
    monkeypatch.setattr(explorer, "simulate", record_simulation)
    result = explorer.simulation_payload(request_payload)
    assert len(prepared_calls) == 1
    assert len(simulation_calls) == len(topologies) * len(explorer.MODES) * 2
    for network in result["topologies"]:
        inputs = prepared[network["id"]]
        matched_calls = [call for call in simulation_calls if call[1] is inputs]
        assert len(matched_calls) == len(explorer.MODES) * 2
        for mode in explorer.MODES:
            run_config = replace(config, fixed_payment=0) if mode == "no_payment" else config
            expected = simulate(run_config, inputs, "fixed" if mode == "no_payment" else mode)
            baseline = network["baseline_scenarios"][mode]
            current = network["scenarios"][mode]
            assert_array_equal(baseline["beliefs"], expected.beliefs)
            assert_array_equal(baseline["wealth"], expected.wealth)
            assert_array_equal(baseline["payments"][1:], expected.payments)
            assert_array_equal(baseline["recipients"][1:], expected.recipients)
            assert_array_equal(baseline["external_cash"], 0.0)
            for field in ("beliefs", "wealth", "payments", "recipients", "external_cash"):
                assert current[field][:4] == baseline[field][:4]
            if np.any(inputs.conditional_allies):
                assert_array_equal(baseline["relationships"], expected.relationships)
                assert current["relationships"][:4] == baseline["relationships"][:4]
            else:
                assert "relationships" not in baseline
                assert "relationships" in current


@pytest.mark.parametrize("conditional_fraction, budget", [(0.0, 19_000), (1.0, 35_000)])
def test_baseline_beliefs_cash_and_relationships_are_budgeted_before_generation(
    request_payload, conditional_fraction, budget, monkeypatch,
):
    request_payload["config"]["conditional_ally_fraction"] = conditional_fraction
    monkeypatch.setattr(explorer, "MAX_TRAJECTORY_VALUES", budget)
    explorer.parse_request(request_payload)
    request_payload["interventions"] = [{"agent": 0, "round": 4, "cash_delta": 1.0}]
    monkeypatch.setattr(explorer, "make_network_inputs", lambda *args: pytest.fail("unexpected allocation"))
    with pytest.raises(ValueError, match="including the baseline comparison"):
        explorer.simulation_payload(request_payload)


def test_restored_initial_traits_budget_both_relationship_histories_before_simulation(request_payload, monkeypatch):
    saved = explorer.simulation_payload(request_payload)["experiment"]
    saved["interventions"] = [{"agent": 0, "round": 4, "cash_delta": 1.0}]
    for inputs in saved["prepared_inputs"].values():
        inputs["conditional_allies"][0] = True
    # Config alone has no initial loyalty; the recorded traits require two histories.
    monkeypatch.setattr(explorer, "MAX_TRAJECTORY_VALUES", 35_000)
    explorer.parse_request(saved)
    monkeypatch.setattr(explorer, "simulate", lambda *args, **kwargs: pytest.fail("unexpected simulation"))
    with pytest.raises(ValueError, match="relationship histories and baseline comparison"):
        explorer.simulation_payload(saved)


def test_baseline_failure_names_network_payment_rule_and_empty_log_arm(request_payload, monkeypatch):
    request_payload["interventions"] = [{"agent": 0, "round": 4, "cash_delta": 1.0}]

    def fail_baseline(config, inputs, mode, *, interventions=()):
        if not interventions:
            raise ValueError("baseline numerical failure")
        return simulate(config, inputs, mode, interventions=interventions)

    monkeypatch.setattr(explorer, "simulate", fail_baseline)
    with pytest.raises(ValueError, match="BA hubs / Fixed payment / baseline without interventions: baseline numerical failure"):
        explorer.simulation_payload(request_payload)
