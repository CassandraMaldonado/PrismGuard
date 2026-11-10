# I keep the measurement methods here so the assumptions are easy to inspect.
# This includes effect estimates, confidence intervals, calibration error and a population-stability measure for comparing reference and current traffic.

from dataclasses import asdict, dataclass
import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Estimate:
    control_mean: float
    candidate_mean: float
    effect: float
    relative_effect: float
    ci_low: float
    ci_high: float
    p_value: float
    standard_error: float
    n_control: int
    n_candidate: int


def estimate(outcome: np.ndarray, policy: np.ndarray) -> dict:
    control, candidate = np.asarray(outcome)[~policy], np.asarray(outcome)[policy]
    effect = float(np.mean(candidate) - np.mean(control))
    se = float(np.sqrt(np.var(control, ddof=1) / len(control) + np.var(candidate, ddof=1) / len(candidate)))
    numerator = (np.var(control, ddof=1) / len(control) + np.var(candidate, ddof=1) / len(candidate)) ** 2
    denominator = (np.var(control, ddof=1) / len(control)) ** 2 / (len(control) - 1) + (np.var(candidate, ddof=1) / len(candidate)) ** 2 / (len(candidate) - 1)
    dof = numerator / denominator
    critical = float(stats.t.ppf(0.975, dof))
    p_value = float(2 * stats.t.sf(abs(effect / se), dof)) if se else 1.0
    control_mean = float(np.mean(control))
    return asdict(Estimate(control_mean, float(np.mean(candidate)), effect, effect / control_mean if control_mean else 0.0, effect - critical * se, effect + critical * se, p_value, se, len(control), len(candidate)))


def calibration(confidence: np.ndarray, labels: np.ndarray, bins: int = 10) -> dict:
    edges = np.linspace(0, 1, bins + 1)
    indices = np.clip(np.digitize(confidence, edges, right=True) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for index in range(bins):
        mask = indices == index
        if not np.any(mask):
            continue
        avg_confidence, accuracy, share = float(np.mean(confidence[mask])), float(np.mean(labels[mask])), float(np.mean(mask))
        ece += abs(avg_confidence - accuracy) * share
        rows.append({"confidence": avg_confidence, "accuracy": accuracy, "count": int(np.sum(mask))})
    brier = float(np.mean((confidence - labels) ** 2))
    return {"ece": ece, "brier": brier, "bins": rows}


def population_stability(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)
    ref_share = np.maximum(ref_counts / len(reference), 1e-6)
    current_share = np.maximum(current_counts / len(current), 1e-6)
    return float(np.sum((current_share - ref_share) * np.log(current_share / ref_share)))
