"""Preservation-page boundaries, paired replay, and complete-network semantics."""

from copy import deepcopy
from http.client import HTTPConnection
import json
from threading import Thread

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from bayes_cash import explorer, strategy_explorer


@pytest.fixture
def small_request():
    return {"kind": "belief-preservation", "config": {
        "n_agents": 6, "rounds": 8, "grid_size": 7, "public_evidence": False,
    }}


def test_initialization_and_saved_experiment_reproduce_every_value(small_request):
    result = strategy_explorer.simulation_payload(small_request)
    assert result["baseline"] is None
    assert_array_equal(result["scenario"]["relationships"][0], np.eye(6, dtype=int) - 1)
    assert_allclose(sorted(result["initial_centers"]), np.linspace(0, 1, 6))
    assert_array_equal(result["scenario"]["distance"][0], np.zeros(6))
    assert strategy_explorer.simulation_payload(json.loads(json.dumps(result["experiment"]))) == result


def test_separate_log_replay_pairs_inputs_and_preserves_prefix(small_request):
    original = strategy_explorer.simulation_payload(small_request)
    saved = original["experiment"]
    log = {"kind": "belief-preservation-interventions", "format_version": 2,
           "interventions": [{"agent": 0, "round": 4, "cash_delta": 10},
                             {"agent": 1, "round": 5, "neighbor": 2, "connection": 1}]}
    edited = strategy_explorer.simulation_payload({**saved, "interventions": log["interventions"]})
    assert saved["interventions"] == []
    assert edited["experiment"]["prepared_inputs"] == saved["prepared_inputs"]
    assert edited["baseline"] == original["scenario"]
    assert edited["scenario"]["beliefs"][:4] == original["scenario"]["beliefs"][:4]
    assert_allclose(np.diff(edited["scenario"]["metrics"]["wealth_total"]), [0, 0, 0, 10, 0, 0, 0, 0], atol=1e-12)
    assert strategy_explorer.simulation_payload(edited["experiment"]) == edited


def test_saved_inactive_costs_do_not_change_dynamics(small_request):
    first = strategy_explorer.simulation_payload(small_request)
    changed = deepcopy(first["experiment"])
    changed["settings"].update(incitation_cost=123, reinforcement_scale=456)
    second = strategy_explorer.simulation_payload(changed)
    assert second["scenario"] == first["scenario"]
    assert second["metadata"]["spending_mechanisms"] == "inactive"
    assert second["settings"]["incitation_cost"] == 123


@pytest.mark.parametrize("strategy,count", [("random_one", 1), ("all", 5), ("farthest", 1), ("farthest_random", 2)])
def test_payment_strategies_export_all_transfers_and_replay_exactly(small_request, strategy, count):
    request = deepcopy(small_request)
    request["config"]["conditional_ally_fraction"] = 0.3
    request["settings"] = {"payment_strategy": strategy, "junzi_fraction": 0.3}
    result = strategy_explorer.simulation_payload(request)
    scenario = result["scenario"]
    contacts, transfers = np.asarray(scenario["contacts"]), np.asarray(scenario["transfers"])
    assert_array_equal(contacts[1:].sum(axis=2), np.full((8, 6), count))
    assert not contacts[:, np.arange(6), np.arange(6)].any()
    assert_allclose(transfers.sum(axis=2), scenario["payments"])
    assert_allclose(transfers[1:].sum(axis=2), np.asarray(scenario["wealth"])[:-1] * 0.05)
    assert_allclose(scenario["metrics"]["wealth_total"], np.sum(scenario["wealth"][0]))
    assert "recipients" not in scenario
    assert result["metadata"]["unpaid_defection"] == "active"
    assert strategy_explorer.simulation_payload(json.loads(json.dumps(result["experiment"]))) == result


def test_type_and_strategy_edits_preserve_prefix_and_distance_is_wasserstein(small_request):
    request = deepcopy(small_request)
    request["settings"] = {"junzi_fraction": 1}
    first = strategy_explorer.simulation_payload(request)
    saved = first["experiment"]
    saved["interventions"] = [
        {"agent": 0, "round": 4, "payment_strategy": "all"},
        {"agent": 1, "round": 4, "person_type": "payment_dependent"},
        {"agent": 0, "round": 4, "belief_center": 0.5, "belief_width": 0.001, "belief_strength": 1},
    ]
    result = strategy_explorer.simulation_payload(saved)
    assert result["baseline"] == first["scenario"]
    assert result["scenario"]["beliefs"][:4] == first["scenario"]["beliefs"][:4]
    assert np.asarray(result["scenario"]["contacts"])[4, 0].sum() == 5
    initial = np.asarray(result["scenario"]["beliefs"][0][0])
    current = np.asarray(result["scenario"]["beliefs"][4][0])
    assert_array_equal(current, [0, 0, 0, 1, 0, 0, 0])
    # Every initial mass must move to theta=.5: the transport cost is analytic.
    expected = np.sqrt(np.dot(initial, (np.asarray(result["theta"]) - 0.5) ** 2))
    assert result["scenario"]["distance"][4][0] == pytest.approx(expected)
    assert expected != pytest.approx(0.5 * np.abs(current - initial).sum())
    assert strategy_explorer.simulation_payload(result["experiment"]) == result


@pytest.mark.parametrize("change,match", [
    ({"kind": "different"}, "belief-preservation"),
    ({"config": {"opponent_fraction": 0.5}}, "all initial opponents"),
    ({"config": {"income_rate": 0.1}}, "zero income"),
    ({"config": {"n_agents": 100, "rounds": 1000}}, "trajectory limit"),
    ({"settings": {"components": []}}, "one to three"),
    ({"settings": {"components": [{"mean": 0.5}]}}, "mean, variance, and weight"),
    ({"settings": {"payment_strategy": "unsupported"}}, "payment_strategy"),
    ({"config": {"conditional_ally_fraction": 0.6}, "settings": {"junzi_fraction": 0.5}}, "sum to at most 1"),
    ({"config": {"ally_giver_fraction": 0.2}}, "replace ally/opponent"),
    ({"interventions": [{"agent": 0, "round": 1, "neighbor": 1, "connection": 0}]}, "cannot be removed"),
    ({"interventions": [{"agent": 0, "round": 1, "incitation_spend": 1}]}, "supported manual"),
    ({"interventions": [{"agent": 0, "round": 1, "conditional_ally": True}]}, "supported manual"),
])
def test_invalid_requests_fail_before_input_generation(small_request, change, match, monkeypatch):
    def unexpected(*args):
        pytest.fail("invalid requests must not allocate simulation inputs")

    monkeypatch.setattr(explorer, "make_network_inputs", unexpected)
    with pytest.raises(ValueError, match=match):
        strategy_explorer.simulation_payload({**small_request, **change})


def test_saved_inputs_cannot_change_network_or_model_identity(small_request):
    original = strategy_explorer.simulation_payload(small_request)["experiment"]
    changed = deepcopy(original)
    changed["model"]["sha256"] = "another-model"
    with pytest.raises(ValueError, match="model or dependency"):
        strategy_explorer.simulation_payload(changed)
    changed = deepcopy(original)
    changed["prepared_inputs"]["complete"]["relationships"][0][1] = 1
    changed["prepared_inputs"]["complete"]["relationships"][1][0] = 1
    with pytest.raises(ValueError, match="every pair opposed"):
        strategy_explorer.simulation_payload(changed)


def test_old_format_and_inconsistent_person_types_are_not_silently_reinterpreted(small_request):
    saved = strategy_explorer.simulation_payload(small_request)["experiment"]
    changed = deepcopy(saved)
    changed["format_version"] = 1
    del changed["person_types"]
    with pytest.raises(ValueError, match="original implementation for version 1"):
        strategy_explorer.simulation_payload(changed)
    changed = deepcopy(saved)
    changed["person_types"][0] = "junzi"
    with pytest.raises(ValueError, match="person_types.*population settings"):
        strategy_explorer.simulation_payload(changed)


@pytest.mark.parametrize("change", ["centers", "beliefs", "variance"])
def test_saved_hierarchy_cannot_disagree_with_actual_initial_beliefs(small_request, change):
    saved = strategy_explorer.simulation_payload(small_request)["experiment"]
    if change == "centers":
        saved["initial_centers"].reverse()
    elif change == "beliefs":
        saved["prepared_inputs"]["complete"]["initial_beliefs"][1] = saved["prepared_inputs"]["complete"]["initial_beliefs"][0]
    else:
        saved["settings"]["within_agent_variance"] = 0.5
    with pytest.raises(ValueError, match="hierarchy settings"):
        strategy_explorer.simulation_payload(saved)


def test_strategy_export_chooses_separate_template_and_preserves_no_overwrite(small_request, tmp_path):
    destination = tmp_path / "preservation.html"
    explorer.export_html(destination, small_request)
    rendered = destination.read_text()
    assert "Belief Preservation Lab" in rendered
    assert "window.EXPLORER_DATA = " in rendered
    assert "/*__EXPLORER_DATA__*/" not in rendered
    with pytest.raises(FileExistsError):
        explorer.export_html(destination, small_request)
    dangerous = explorer.render_html({"name": "</script>\u2028"}, template_name="strategy.html")
    assert "\\u003c/script>" in dangerous
    assert "\\u2028" in dangerous


def test_strategy_route_endpoint_and_old_page_navigation(small_request):
    with explorer.make_server(0) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for path, expected in (("/", "strategy.html"), ("/strategy.html", "Belief Preservation Lab")):
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                connection.request("GET", path)
                response = connection.getresponse()
                assert response.status == 200
                assert expected in response.read().decode()
                connection.close()
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            connection.request("POST", "/api/strategy/simulate", json.dumps(small_request), {"Content-Type": "application/json"})
            response = connection.getresponse()
            assert response.status == 200
            assert json.loads(response.read())["experiment"]["kind"] == "belief-preservation"
            connection.close()
        finally:
            server.shutdown()
            thread.join(timeout=3)
