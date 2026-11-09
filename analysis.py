# Where the evaluation comes together. I compare the two policies, check the quality and safety guardrails, look for differences across slices,
# measure calibration and drift, and turn the evidence into a rollout decision.

from .metrics import calibration, estimate, population_stability
from .simulation import COHORTS, TASKS, EvaluationSet, simulate_evaluation


def _slice(data: EvaluationSet, mask) -> dict:
    return {"quality": estimate(data.human_quality[mask], data.policy[mask]), "grounded": estimate(data.grounded[mask], data.policy[mask])}


def analyze(seed: int = 41, samples: int = 3_000) -> dict:
    data = simulate_evaluation(samples, seed)
    quality = estimate(data.human_quality, data.policy)
    grounded = estimate(data.grounded, data.policy)
    safety = estimate(data.unsafe, data.policy)
    latency = estimate(data.latency_ms, data.policy)
    cost = estimate(data.cost_usd, data.policy)
    calibration_result = calibration(data.confidence, data.human_quality)
    # Current traffic is intentionally shifted a little to demonstrate the check.
    drift = population_stability(data.reference_length[~data.policy], data.reference_length[data.policy] + 24)
    task_slices = {str(task): _slice(data, data.task_type == task) for task in TASKS}
    cohort_slices = {str(cohort): _slice(data, data.cohort == cohort) for cohort in COHORTS}
    guardrails = [
        {"name": "Human-rated quality", "passed": quality["ci_low"] > 0, "value": quality["effect"], "threshold": "95% CI above 0"},
        {"name": "Grounded responses", "passed": grounded["ci_low"] > 0, "value": grounded["effect"], "threshold": "95% CI above 0"},
        {"name": "Unsafe response rate", "passed": safety["ci_high"] < 0.005, "value": safety["effect"], "threshold": "increase below 0.5 pp"},
        {"name": "Calibration", "passed": calibration_result["ece"] < 0.08, "value": calibration_result["ece"], "threshold": "ECE below 0.08"},
        {"name": "Traffic drift", "passed": drift < 0.20, "value": drift, "threshold": "PSI below 0.20"},
    ]
    failures = [item for item in guardrails if not item["passed"]]
    if quality["ci_low"] > 0 and not failures:
        status = "SHIP"
    elif quality["ci_high"] <= 0 or len(failures) >= 2:
        status = "HOLD"
    else:
        status = "ITERATE"
    return {
        "experiment": {"name": "Retrieval and Response Policy", "samples": samples, "tasks": len(TASKS), "cohorts": len(COHORTS)},
        "decision": {"status": status, "summary": f"Human-rated quality changed {quality['relative_effect']:+.1%}; candidate latency changed {latency['effect']:+.0f} ms.", "next_action": "Review failing guardrails, validate with live shadow traffic, and keep a human review path active."},
        "metrics": {"quality": quality, "grounded": grounded, "unsafe_rate": safety, "latency_ms": latency, "cost_usd": cost},
        "guardrails": guardrails,
        "calibration": calibration_result,
        "drift": {"psi": drift, "status": "MONITOR" if drift >= 0.10 else "STABLE"},
        "task_slices": task_slices,
        "cohort_slices": cohort_slices,
    }

