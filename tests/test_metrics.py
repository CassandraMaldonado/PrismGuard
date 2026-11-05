import numpy as np
import pytest
from prism_guard.analysis import analyze
from prism_guard.metrics import calibration, estimate, population_stability
from prism_guard.simulation import simulate_evaluation

def test_simulation_is_deterministic_and_has_candidate_quality_lift():
    first, second = simulate_evaluation(seed=5), simulate_evaluation(seed=5)
    assert np.array_equal(first.human_quality, second.human_quality)
    assert estimate(first.human_quality, first.policy)["effect"] > 0


def test_calibration_and_drift_metrics():
    metric = calibration(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]), bins=2)
    assert metric["ece"] >= 0
    assert population_stability(np.arange(100), np.arange(100)) == pytest.approx(0.0)


def test_analysis_contract():
    result = analyze()
    assert result["decision"]["status"] in {"SHIP", "ITERATE", "HOLD"}
    assert len(result["guardrails"]) == 5
    assert set(result["task_slices"]) == {"research", "support", "coding", "creative"}
