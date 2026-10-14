"""Historical studies must reject traits that invalidate their fixed-state assumptions."""

from dataclasses import replace
from pathlib import Path
import sys

import pytest

from bayes_cash.model import Config, make_inputs


scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
try:
    import check_network_effects as controls
    import compare_extensions as extensions
    import compare_opponents as comparison
    import sweep_opponents as sweep
finally:
    sys.path.remove(str(scripts))


@pytest.mark.parametrize("trait", [
    "ally_giver_fraction", "opponent_giver_fraction", "conditional_ally_fraction",
])
@pytest.mark.parametrize("run", [
    pytest.param(lambda c, i, o: controls.control_inputs(c, "uniform"), id="network-inputs"),
    pytest.param(lambda c, i, o: controls.simulate_cash(c, i, "fixed"), id="cash-trajectory"),
    pytest.param(lambda c, i, o: controls.run_controls(c, 2, o), id="network-study"),
    pytest.param(lambda c, i, o: comparison.paired_results(c, i, "fixed"), id="opponent-pair"),
    pytest.param(lambda c, i, o: comparison.run_comparison(c, 2, o), id="opponent-study"),
    pytest.param(lambda c, i, o: sweep.make_sweep_inputs(c, "uniform"), id="sweep-inputs"),
    pytest.param(lambda c, i, o: sweep.run_trajectory(c, i, "fixed", (8,)), id="sweep-trajectory"),
    pytest.param(lambda c, i, o: sweep.run_sweep(c, 2, o, horizons=(8,)), id="sweep-study"),
    pytest.param(lambda c, i, o: extensions.run_trajectory(c, i, "fixed", (8,)), id="extension-trajectory"),
    pytest.param(lambda c, i, o: extensions.run_comparison(c, 2, o, horizons=(8,)), id="extension-study"),
])
def test_historical_entrypoints_reject_new_traits_before_writing(trait, run, tmp_path):
    config = replace(Config(n_agents=6, attachment=2, rounds=8), **{trait: 1.0})
    inputs = make_inputs(config, "uniform")
    output = tmp_path / "unsupported-study"
    with pytest.raises(ValueError, match="uniform giving and static relationships.*batch command or explorer"):
        run(config, inputs, output)
    assert not output.exists()
